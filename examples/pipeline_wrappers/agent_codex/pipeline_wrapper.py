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

from hayhooks import (
    BasePipelineWrapper,
    async_streaming_generator,
    chat_messages_from_openai_response,
    get_last_user_input_text,
    log,
)

from .client_tools import build_generation_kwargs
from .weather import NO_WEATHER_CONTEXT, create_weather_agent, looks_like_weather_request

AGENT_MODEL = "gpt-4.1-mini"


def _is_tool_followup(input_items: list[dict]) -> bool:
    """True when the latest meaningful item is a function_call / function_call_output.

    Used to skip server-side weather enrichment during the Codex tool loop.
    """
    for item in reversed(input_items):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("function_call", "function_call_output"):
            return True
        if item_type in ("message", None):
            return False
    return False


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = AsyncPipeline()
        self.pipeline.add_component("llm", OpenAIChatGenerator(model=AGENT_MODEL))
        self.weather_agent = create_weather_agent()

    async def _enrich_with_weather(self, input_items: list[dict], messages: list[ChatMessage]) -> None:
        """Run server-side weather lookup and inject context if relevant.

        Skipped when the request is a tool follow-up (Codex sending back
        function_call_output) to avoid redundant lookups mid-tool-loop.
        """
        if _is_tool_followup(input_items):
            log.opt(colors=True).debug("<yellow>[weather]</yellow> Skipping: tool-followup request")
            return

        last_user_text = get_last_user_input_text(input_items)
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

    async def run_response_async(self, model: str, input_items: list[dict], body: dict) -> str | AsyncGenerator:
        # 1. Build ChatMessages from the Responses API input.
        #    - Prepend system instructions from the request body (if any).
        #    - Convert input_items (messages, function_call, function_call_output)
        #      into Haystack ChatMessage objects so the LLM can process the full
        #      conversation history including past client-side tool calls.
        messages: list[ChatMessage] = []
        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            messages.append(ChatMessage.from_system(instructions))
        messages.extend(chat_messages_from_openai_response(input_items))

        # 2. Server-side weather enrichment (optional).
        #    If the latest user prompt looks weather-related, run a lightweight
        #    Haystack Agent with the Open-Meteo tool and inject its answer as
        #    extra system context. This happens before the main LLM call so the
        #    model can reference real weather data in its response.
        await self._enrich_with_weather(input_items, messages)

        # 3. Forward client-side tool definitions to the LLM.
        #    Codex sends its tool schemas (exec_command, etc.) in the request body.
        #    We convert them from Responses API format to Chat Completions format
        #    and pass them as generation_kwargs so the model can emit function_call
        #    chunks that Codex will execute locally.
        generation_kwargs = build_generation_kwargs(body)
        log.info("Streaming response with {} message(s)", len(messages))

        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "llm": {
                    "messages": messages,
                    "generation_kwargs": generation_kwargs,
                }
            },
        )
