import os
from pathlib import Path

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from hayhooks import BasePipelineWrapper, stream_agent_chat

from .file_tools import build_file_tools  # noqa: TID252


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        workspace = os.environ.get("EXAMPLE_CHAT_WORKSPACE")
        if not workspace:
            message = "Set EXAMPLE_CHAT_WORKSPACE to an existing server-side demo directory before deploying."
            raise ValueError(message)
        try:
            tools = build_file_tools(Path(workspace))
        except ValueError as exc:
            message = "EXAMPLE_CHAT_WORKSPACE must name an existing, accessible server-side directory."
            raise ValueError(message) from exc
        model = os.environ.get("EXAMPLE_AGENT_MODEL", "gpt-6-luna")
        # GPT-6 Luna needs reasoning disabled for tool calls through Chat Completions.
        generation_kwargs = {"reasoning_effort": "none"} if model == "gpt-6-luna" else None
        self.pipeline = Agent(
            chat_generator=OpenAIChatGenerator(model=model, generation_kwargs=generation_kwargs),
            tools=tools,
            system_prompt=(
                "Help the user work with UTF-8 text files in the configured server workspace. "
                "Use list_files, read_file, and write_file for file operations. "
                "Use relative paths and report tool failures clearly. "
                "Never claim a write succeeded without a tool result."
            ),
            max_agent_steps=10,
            raise_on_tool_invocation_failure=False,
        )

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict):
        del model, body
        return stream_agent_chat(self.pipeline, [ChatMessage.from_openai_dict_format(message) for message in messages])
