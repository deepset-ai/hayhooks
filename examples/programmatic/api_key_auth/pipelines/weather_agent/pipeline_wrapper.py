"""
Sample pipeline wrapper: a weather-info agent.

This wrapper is automatically picked up by Hayhooks because it lives under
the ``pipelines/`` directory.  Hayhooks expects a ``PipelineWrapper`` class
that inherits from ``BasePipelineWrapper``.

Two async entry points are implemented:

* ``run_api_async``  - called by the REST API (``POST /weather_agent``).
  The method signature's type hints and docstring are used by Hayhooks to
  auto-generate the request/response schema.

* ``run_chat_completion_async`` - called by the OpenAI-compatible
  ``/v1/chat/completions`` endpoint, enabling streaming via SSE.
"""

from collections.abc import AsyncGenerator

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from hayhooks import BasePipelineWrapper, async_streaming_generator, log


def weather_function(location: str) -> str:
    return f"The weather in {location} is sunny and 24 degrees Celsius."


# Tools are plain Python functions wrapped with Tool().
# Replace this with real API calls, DB lookups, etc.
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
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt="You're a helpful weather assistant.",
            tools=[weather_tool],
        )

    async def run_api_async(self, question: str) -> str:  # type: ignore[override]
        """
        Ask the weather agent a question.

        Args:
            question: A natural-language question about the weather.
        """
        result = await self.agent.run_async(messages=[ChatMessage.from_user(question)])
        log.trace("Weather agent result: {}", result)
        return result["last_message"].text

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        del model, body
        chat_messages = [ChatMessage.from_openai_dict_format(message) for message in messages]

        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={"messages": chat_messages},
        )
