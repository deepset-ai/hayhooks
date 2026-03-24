"""
Responses API wrapper for Codex-style client-side tools.

Keeps Codex client tools (e.g. ``exec_command``) fully client-side while
optionally enriching weather questions with a server-side Haystack Agent
backed by Open-Meteo.
"""

# ruff: noqa: TID252

from collections.abc import AsyncGenerator

from haystack import AsyncPipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from hayhooks import BasePipelineWrapper, async_streaming_generator, log

from .client_tools import build_generation_kwargs, client_tool_names
from .input_utils import (
    input_items_to_chat_messages,
    is_tool_followup_request,
    last_user_input_text,
)
from .weather import NO_WEATHER_CONTEXT, create_weather_agent, looks_like_weather_request

AGENT_MODEL = "gpt-4.1-mini"


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = AsyncPipeline()
        self.pipeline.add_component("llm", OpenAIChatGenerator(model=AGENT_MODEL))
        self.weather_agent = create_weather_agent()

    def _build_messages(self, input_items: list[dict], body: dict) -> list[ChatMessage]:
        messages: list[ChatMessage] = []

        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            messages.append(ChatMessage.from_system(instructions))

        messages.extend(input_items_to_chat_messages(input_items))
        return messages

    async def _add_server_weather_context(self, input_items: list[dict], messages: list[ChatMessage]) -> None:
        if is_tool_followup_request(input_items):
            log.opt(colors=True).debug("<yellow>[weather]</yellow> Skipping enrichment: tool-followup request")
            return

        last_user_text = last_user_input_text(input_items)
        if not looks_like_weather_request(last_user_text):
            return

        log.opt(colors=True).info("<yellow>[weather]</yellow> Prompt detected; invoking server weather agent")
        weather_result = await self.weather_agent.run_async(
            messages=[ChatMessage.from_user(last_user_text or "")],
            generation_kwargs={"temperature": 0},
        )

        last_message = weather_result.get("last_message")
        weather_context = ""
        if isinstance(last_message, ChatMessage):
            weather_context = (last_message.text or "").strip()

        if weather_context and weather_context != NO_WEATHER_CONTEXT:
            messages.append(
                ChatMessage.from_system(
                    f"Server weather context (from server_get_weather tool):\n{weather_context}"
                )
            )
            log.opt(colors=True).info("<yellow>[weather]</yellow> Injected server context into model messages")
        else:
            log.opt(colors=True).debug("<yellow>[weather]</yellow> Agent returned no context to inject")

    async def run_response_async(self, model: str, input_items: list[dict], body: dict) -> str | AsyncGenerator:
        messages = self._build_messages(input_items, body)
        await self._add_server_weather_context(input_items, messages)

        generation_kwargs = build_generation_kwargs(body)
        chat_tools = generation_kwargs.get("tools", [])
        names = client_tool_names(chat_tools) if isinstance(chat_tools, list) else []
        log.info(
            "Running Codex + weather demo with {} message(s) and {} client tool(s)",
            len(messages),
            len(names),
        )
        if names:
            log.opt(colors=True).debug("<blue>[client-tool]</blue> Forwarding: {}", ", ".join(names))

        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "llm": {
                    "messages": messages,
                    "generation_kwargs": generation_kwargs,
                }
            },
        )
