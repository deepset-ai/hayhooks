import os
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo

from hayhooks import BasePipelineWrapper, async_streaming_generator, log

WEATHER_MCP_URL = os.getenv("WEATHER_MCP_URL", "http://localhost:8001/mcp")


def _format_tool_names(tools: list[Any]) -> str:
    return ", ".join(tool.name for tool in tools)


def _log_tool_call_start(tool_name: str, arguments: dict[str, Any] | None, _tool_call_id: str | None) -> None:
    log.info("weather_agent | MCP -> weather_server | calling '{}' with {}", tool_name, arguments or {})


def _log_tool_call_end(tool_name: str, arguments: dict[str, Any], result: str, error: bool) -> None:  # noqa: ARG001
    status = "failed" if error else "finished"
    log.info("weather_agent | MCP -> weather_server | '{}' {} ({} chars)", tool_name, status, len(result))


class PipelineWrapper(BasePipelineWrapper):
    # Customize the auto-generated A2A agent card
    a2a_card: ClassVar[dict[str, Any]] = {
        "name": "weather_agent",
        "description": "Answers questions about the current weather in any city, using live Open-Meteo data.",
        "skills": [
            {
                "id": "get_current_weather",
                "name": "Get current weather",
                "description": "Report current temperature, humidity, wind and conditions for a city.",
                "tags": ["weather"],
                "examples": ["What's the weather in Berlin right now?"],
            }
        ],
    }

    def setup(self) -> None:
        log.info("weather_agent | setup | connecting MCP tools at {}", WEATHER_MCP_URL)
        weather_toolset = MCPToolset(server_info=StreamableHttpServerInfo(url=WEATHER_MCP_URL), eager_connect=True)
        weather_tools = [*weather_toolset]
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt=(
                "You are a weather assistant. Use the get_weather tool to fetch current conditions "
                "for the requested city, then answer concisely with temperature, conditions, "
                "humidity and wind speed."
            ),
            tools=weather_tools,
        )
        log.info("weather_agent | setup | MCP tools ready: {}", _format_tool_names(weather_tools))

    async def run_chat_completion_async(
        self,
        model: str,  # noqa: ARG002
        messages: list[dict],
        body: dict,  # noqa: ARG002
    ) -> AsyncGenerator:
        log.info("weather_agent | request | answering incoming A2A weather message")
        chat_messages = [ChatMessage.from_openai_dict_format(message) for message in messages]

        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={"messages": chat_messages},
            on_tool_call_start=_log_tool_call_start,
            on_tool_call_end=_log_tool_call_end,
        )
