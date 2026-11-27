"""Expose a filesystem Agent through the Hayhooks chat endpoint."""

import os
from pathlib import Path

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage

from hayhooks import BasePipelineWrapper, stream_agent_chat

from .file_tools import build_file_tools  # noqa: TID252

DEFAULT_MODEL = "gpt-6-luna"


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        workspace = os.getenv("EXAMPLE_CHAT_WORKSPACE")
        workspace_error = "Set EXAMPLE_CHAT_WORKSPACE to an existing, accessible directory on the server."
        if not workspace:
            raise ValueError(workspace_error)
        try:
            tools = build_file_tools(Path(workspace))
        except ValueError as exc:
            raise ValueError(workspace_error) from exc

        model = os.getenv("EXAMPLE_AGENT_MODEL", DEFAULT_MODEL)
        generation_kwargs: dict[str, object] = {"store": False}
        if model == DEFAULT_MODEL:
            generation_kwargs["reasoning"] = {"effort": "low", "summary": "auto"}
        chat_generator = OpenAIResponsesChatGenerator(model=model, generation_kwargs=generation_kwargs)

        self.pipeline = Agent(
            chat_generator=chat_generator,
            tools=tools,
            system_prompt=(
                "Use the workspace tools to work with UTF-8 text files. "
                "Use relative paths and explain tool failures. "
                "Confirm a write only after the tool reports success."
            ),
            max_agent_steps=10,
            raise_on_tool_invocation_failure=False,
        )

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict):
        del model, body  # The deployed Agent owns model selection and generation options.
        chat_messages = [ChatMessage.from_openai_dict_format(message) for message in messages]
        return stream_agent_chat(self.pipeline, chat_messages)
