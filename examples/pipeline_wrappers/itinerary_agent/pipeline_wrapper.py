import pathlib
from typing import Any, AsyncGenerator, Dict
from haystack.tools import ComponentTool
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator
from haystack_integrations.tools.mcp.mcp_tool import SSEServerInfo
from haystack_integrations.tools.mcp.mcp_toolset import MCPToolset


def load_day_itinerary_system_message():
    """Load the system message from the external file."""
    current_dir = pathlib.Path(__file__).parent
    system_file = current_dir / "day_itinerary_system_prompt.txt"
    with open(system_file, encoding="utf-8") as f:
        return f.read()

def load_lodging_itinerary_system_message():
    """Load the lodging itinerary system message from the external file."""
    current_dir = pathlib.Path(__file__).parent
    system_file = current_dir / "lodging_itinerary_system_prompt.txt"
    with open(system_file, encoding="utf-8") as f:
        return f.read()

def load_macro_itinerary_system_message():
    """Load the system message from the external file."""
    current_dir = pathlib.Path(__file__).parent
    system_file = current_dir / "macro_itinerary_system_prompt.txt"
    with open(system_file, encoding="utf-8") as f:
        return f.read()


maps_toolset = MCPToolset(
    SSEServerInfo(url="http://localhost:8100/sse"),
    tool_names=["maps_search_places", "maps_place_details"],
)

routing_toolset = MCPToolset(
    SSEServerInfo(url="http://localhost:8104/sse"),
    tool_names=["compute_optimal_route", "get_distance_direction"],
)

perplexity_toolset = MCPToolset(
    SSEServerInfo(
        url="http://localhost:8105/sse"
    ),
    tool_names=["perplexity_ask"],
    invocation_timeout=120, # seconds, as perplexity takes time to respond
)

# Combine all tools
all_tools = maps_toolset + routing_toolset + perplexity_toolset

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        llm = OpenAIChatGenerator(model="gpt-4.1")

        day_itinerary_agent = Agent(
            system_prompt=load_day_itinerary_system_message(),
            chat_generator=llm,
            tools=all_tools            
        )

        day_tool = ComponentTool(
            name="daily_itinerary_planning_agent",
            description="Plans a detailed one-day itinerary. Input: 'Plan detailed day [X] for [location(s)], with activities drawn from [preferences], and stay at[accommodation].' Call this tool separately per day.",
            component=day_itinerary_agent,
            # We only care about the last message from the day itinerary agent (the day itinerary), not the intermediate tool calls history
            outputs_to_string={"source": "last_message"}
        )

        lodging_itinerary_agent = Agent(
            system_prompt=load_lodging_itinerary_system_message(),
            chat_generator=llm,
            tools=all_tools,            
        )

        lodging_tool = ComponentTool(
            name="accommodation_strategy_optimizer",
            description="Determines optimal accommodation placement for multi-day travel itineraries. Input: 'Optimize accommodation strategy for [X]-day route: [destination sequence], transportation: [mode], lodging preferences: [preferences and budget].'",
            component=lodging_itinerary_agent,
            # We only care about the last message from the lodging itinerary agent (the lodging itinerary), not the intermediate tool calls history
            outputs_to_string={"source": "last_message"}
        )

        self.agent = Agent(
            system_prompt=load_macro_itinerary_system_message(),
            chat_generator=llm,
            tools=all_tools + [day_tool, lodging_tool]            
        )


    async def run_chat_completion_async(
        self, model: str, messages: list[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        chat_messages = [
            ChatMessage.from_openai_dict_format(message) for message in messages
        ]

        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={
                "messages": chat_messages,
            },
        )
