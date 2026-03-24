"""
Responses API wrapper with a file-reading Haystack Agent.

Combines two ways of working with files:

1. **Agent tool** — the LLM can call ``read_file`` to read any local file
   by path. Works with standard Responses API clients (OpenAI client/curl).
2. **``/v1/files`` upload** — clients can upload files via the Files API;
   ``run_file_upload`` stores them in an in-memory dict and the agent can
   retrieve them by ``file_id`` through the ``read_uploaded_file`` tool.

Requires the ``OPENAI_API_KEY`` environment variable to be set.
"""

import time
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import uuid4

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.tools import Tool

from hayhooks import BasePipelineWrapper, async_streaming_generator, log

_file_store: dict[str, dict] = {}

AGENT_MODEL = "gpt-4.1-mini"


def _content_to_text(content: object) -> str:
    """Flatten a Responses API content field into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def _input_items_to_chat_messages(input_items: list[dict]) -> list[ChatMessage]:
    """Convert Responses API input items to Haystack ChatMessage objects."""
    messages: list[ChatMessage] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = _content_to_text(item.get("content"))
        if not text:
            continue
        if role == "user":
            messages.append(ChatMessage.from_user(text))
        elif role in ("system", "developer"):
            messages.append(ChatMessage.from_system(text))
        elif role == "assistant":
            messages.append(ChatMessage.from_assistant(text))
    return messages


def read_file(path: str) -> str:
    """Read the contents of a text file from disk."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return f"Error: '{path}' does not exist or is not a file."
    try:
        return p.read_text(errors="replace")
    except Exception as e:
        return f"Error reading '{path}': {e}"


def read_uploaded_file(file_id: str) -> str:
    """Retrieve the contents of a file that was uploaded via /v1/files."""
    stored = _file_store.get(file_id)
    if not stored:
        return f"Error: file_id '{file_id}' not found in the upload store."
    return stored["content"].decode("utf-8", errors="replace")


read_file_tool = Tool(
    name="read_file",
    description="Read a text file from disk given its absolute or relative path.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the file to read."}},
        "required": ["path"],
    },
    function=read_file,
)

read_uploaded_file_tool = Tool(
    name="read_uploaded_file",
    description="Retrieve the contents of a file previously uploaded via the /v1/files endpoint, given its file_id.",
    parameters={
        "type": "object",
        "properties": {"file_id": {"type": "string", "description": "The file ID returned by the upload endpoint."}},
        "required": ["file_id"],
    },
    function=read_uploaded_file,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that can read and analyze files. "
    "When the user asks about a file, use the read_file tool with an ABSOLUTE path. "
    "When the user references an uploaded file_id, use the read_uploaded_file tool. "
    "Be concise and accurate."
)


async def _strip_tool_calls(gen: AsyncGenerator) -> AsyncGenerator:
    """
    Filter internal Agent tool calls from the stream.

    The Haystack Agent executes tools server-side, but the OpenAI-compat layer
    translates StreamingChunk.tool_calls into SSE function-call events. Clients
    like Codex would treat those as client-side calls and loop forever.
    """
    async for chunk in gen:
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            if hasattr(chunk, "content") and chunk.content:
                yield StreamingChunk(content=chunk.content)
        else:
            yield chunk


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model=AGENT_MODEL),
            system_prompt=SYSTEM_PROMPT,
            tools=[read_file_tool, read_uploaded_file_tool],
        )

    def run_file_upload(self, filename: str | None, content_type: str | None, content: bytes, purpose: str) -> dict:
        file_id = f"file-{uuid4().hex[:24]}"
        _file_store[file_id] = {
            "filename": filename,
            "content_type": content_type,
            "content": content,
        }
        log.info("Stored file '{}' as {} ({} bytes)", filename, file_id, len(content))
        return {
            "id": file_id,
            "object": "file",
            "bytes": len(content),
            "created_at": int(time.time()),
            "filename": filename or "",
            "purpose": purpose,
        }

    async def run_response_async(self, model: str, input_items: list[dict], body: dict) -> str | AsyncGenerator:
        messages = _input_items_to_chat_messages(input_items)
        log.info("Running file agent with {} message(s)", len(messages))

        gen = async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={"messages": messages},
        )
        return _strip_tool_calls(gen)
