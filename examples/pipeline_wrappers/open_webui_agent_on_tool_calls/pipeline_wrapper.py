import time
from collections.abc import AsyncGenerator

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from hayhooks import BasePipelineWrapper, async_streaming_generator
from hayhooks.open_webui import OpenWebUIEvent, create_details_tag, create_notification_event, create_status_event


def weather_function(location):
    """
    This function is a mock for a weather API.

    It returns a dictionary with the weather information for a given location.
    It also sleeps for 3 seconds to simulate a real API call.
    """
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    time.sleep(3)
    return weather_info.get(location, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


# Define a Haystack Tool that provides weather information for a given location.
weather_tool = Tool(
    name="weather_tool",
    description="Provides weather information for a given location.",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
    function=weather_function,
)


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Define a Haystack Agent that uses the weather tool.
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt="You're a helpful agent",
            tools=[weather_tool],
        )

    # Define a callback to intercept the "on_tool_call_start" event.
    def on_tool_call_start(
        self,
        tool_name: str,
        arguments: dict,  # noqa: ARG002
        id: str,  # noqa: ARG002, A002
    ) -> list[OpenWebUIEvent]:
        return [
            create_status_event(description=f"Tool call started: {tool_name}"),
            create_notification_event(
                notification_type="info",
                content=f"Tool call started: {tool_name}",
            ),
        ]

    # Define a callback to intercept the "on_tool_call_end" event.
    def on_tool_call_end(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        error: bool,  # noqa: ARG002
    ) -> list[OpenWebUIEvent]:
        return [
            create_status_event(
                description=f"Tool call ended: {tool_name}",
                done=True,
            ),
            create_notification_event(
                notification_type="success",
                content=f"Tool call ended: {tool_name}",
            ),
            create_details_tag(
                tool_name=tool_name,
                summary=f"Tool call result for {tool_name}",
                content=(f"```\nArguments:\n{arguments}\n\nResponse:\n{result}\n```"),
            ),
        ]

    async def run_chat_completion_async(
        self,
        model: str,  # noqa: ARG002
        messages: list[dict],
        body: dict,  # noqa: ARG002
    ) -> AsyncGenerator[str, None]:
        chat_messages = [ChatMessage.from_openai_dict_format(message) for message in messages]

        # Use the `async_streaming_generator` to run the pipeline and intercept the tool calls
        # using the `on_tool_call_start` and `on_tool_call_end` callbacks.
        return async_streaming_generator(
            on_tool_call_start=self.on_tool_call_start,
            on_tool_call_end=self.on_tool_call_end,
            pipeline=self.agent,
            pipeline_run_args={
                "messages": chat_messages,
            },
        )
