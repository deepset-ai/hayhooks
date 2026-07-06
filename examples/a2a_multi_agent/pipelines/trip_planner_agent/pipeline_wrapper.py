import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

from hayhooks import BasePipelineWrapper, async_streaming_generator, log

ACTIVITIES_MCP_URL = os.getenv("ACTIVITIES_MCP_URL", "http://localhost:8002/mcp")
WEATHER_AGENT_A2A_URL = os.getenv("WEATHER_AGENT_A2A_URL", "http://localhost:1418/weather_agent")
TOOL_ROUTES = {
    "ask_weather_agent": "A2A -> weather_agent",
    "suggest_activities": "MCP -> activities_server",
}


def _format_tool_names(tools: list[Tool]) -> str:
    return ", ".join(tool.name for tool in tools)


def _format_card_skills(card: Any) -> str:
    return "; ".join(f"{skill.name} ({', '.join(skill.tags or [])}) - {skill.description}" for skill in card.skills)


def _log_tool_call_start(tool_name: str, arguments: dict[str, Any] | None, _tool_call_id: str | None) -> None:
    route = TOOL_ROUTES.get(tool_name, "tool")
    log.info("trip_planner_agent | {} | calling '{}' with {}", route, tool_name, arguments or {})


def _log_tool_call_end(tool_name: str, arguments: dict[str, Any], result: str, error: bool) -> None:  # noqa: ARG001
    route = TOOL_ROUTES.get(tool_name, "tool")
    status = "failed" if error else "finished"
    log.info("trip_planner_agent | {} | '{}' {} ({} chars)", route, tool_name, status, len(result))


async def _ask_weather_agent_async(question: str) -> str:
    """Send a message to the weather agent over A2A and return its final response text."""
    import httpx
    from a2a.client import A2ACardResolver, ClientConfig, create_client
    from a2a.helpers import get_artifact_text, get_message_text, new_text_message
    from a2a.types import Role, SendMessageRequest

    async with httpx.AsyncClient(timeout=120) as httpx_client:
        # Discover the agent through its card, then send the question as a user message
        log.info("trip_planner_agent | A2A -> weather_agent | discovering agent card at {}", WEATHER_AGENT_A2A_URL)
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=WEATHER_AGENT_A2A_URL)
        card = await resolver.get_agent_card()
        log.info(
            "trip_planner_agent | A2A -> weather_agent | discovered '{}' v{} | skills: {}",
            card.name,
            card.version,
            _format_card_skills(card),
        )

        client = await create_client(agent=card, client_config=ClientConfig(streaming=False, httpx_client=httpx_client))
        try:
            log.info("trip_planner_agent | A2A -> weather_agent | sending weather question: {}", question)
            request = SendMessageRequest(message=new_text_message(question, role=Role.ROLE_USER))
            texts: list[str] = []
            async for response in client.send_message(request):
                if response.HasField("task"):
                    texts = [get_artifact_text(artifact, delimiter="") for artifact in response.task.artifacts]
                elif response.HasField("message"):
                    texts = [get_message_text(response.message)]
            answer = "".join(texts) or "The weather agent returned no response."
            log.info("trip_planner_agent | A2A -> weather_agent | received response ({} chars)", len(answer))
            return answer
        finally:
            await client.close()


def ask_weather_agent(question: str) -> str:
    """
    Ask the remote weather agent a question via the A2A protocol.

    NOTE: this tool is sync on purpose. Haystack's ToolInvoker runs sync tools in a
    worker thread, so `asyncio.run()` here doesn't block the server event loop that
    is concurrently serving the weather agent's incoming A2A request.
    """
    return asyncio.run(_ask_weather_agent_async(question))


weather_agent_tool = Tool(
    name="ask_weather_agent",
    description=(
        "Ask the weather agent about current weather conditions in a city. "
        "Pass a natural language question, e.g. 'What is the weather in Berlin right now?'"
    ),
    parameters={
        "type": "object",
        "properties": {"question": {"type": "string", "description": "Weather question in natural language"}},
        "required": ["question"],
    },
    function=ask_weather_agent,
)


class PipelineWrapper(BasePipelineWrapper):
    # Customize the auto-generated A2A agent card
    a2a_card: ClassVar[dict[str, Any]] = {
        "name": "trip_planner_agent",
        "description": (
            "Plans a day of activities in a city. Checks the current weather by delegating "
            "to the weather agent over A2A, then suggests matching activities."
        ),
        "skills": [
            {
                "id": "plan_trip",
                "name": "Plan a day trip",
                "description": "Suggest a weather-appropriate day plan for a city.",
                "tags": ["travel", "planning"],
                "examples": ["I'm visiting Berlin today - check the weather and plan my day."],
            }
        ],
    }

    def setup(self) -> None:
        log.info("trip_planner_agent | setup | connecting MCP tools at {}", ACTIVITIES_MCP_URL)
        # eager_connect: the toolset is unpacked into a plain list below, so tools
        # must be resolved now (a lazy toolset would freeze into a placeholder)
        activities_toolset = MCPToolset(
            server_info=StreamableHttpServerInfo(url=ACTIVITIES_MCP_URL), eager_connect=True
        )
        activities_tools = [*activities_toolset]
        all_tools = [*activities_tools, weather_agent_tool]
        log.info(
            "trip_planner_agent | setup | tools ready: {} | routes: suggest_activities=MCP, ask_weather_agent=A2A",
            _format_tool_names(all_tools),
        )
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt=(
                "You are a trip planning assistant. To plan a day in a city, ALWAYS follow these steps:\n"
                "1. Call ask_weather_agent to learn the current weather in the city.\n"
                "2. Call suggest_activities, passing the city and the weather description you learned "
                "(e.g. 'Partly cloudy'). Never suggest activities without calling this tool first.\n"
                "3. Combine everything into a short, friendly day plan that mentions the current weather "
                "and is based on the suggested activities."
            ),
            tools=all_tools,
        )

    async def run_chat_completion_async(
        self,
        model: str,  # noqa: ARG002
        messages: list[dict],
        body: dict,  # noqa: ARG002
    ) -> AsyncGenerator:
        log.info("trip_planner_agent | request | planning trip from incoming A2A message")
        chat_messages = [ChatMessage.from_openai_dict_format(message) for message in messages]

        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={"messages": chat_messages},
            on_tool_call_start=_log_tool_call_start,
            on_tool_call_end=_log_tool_call_end,
        )
