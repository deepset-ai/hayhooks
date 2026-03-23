"""
Chat Completions API with file upload support.

Upload files via ``/v1/files``, then reference them in chat messages using the
standard OpenAI multi-part content format::

    {"type": "file", "file": {"file_id": "file-abc123"}}

The wrapper resolves file references to inline text before passing messages to
a Haystack Agent.  The agent can also read local files by path via a tool.

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


def read_file(path: str) -> str:
    """Read the contents of a text file from disk."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return f"Error: '{path}' does not exist or is not a file."
    try:
        return p.read_text(errors="replace")
    except Exception as e:
        return f"Error reading '{path}': {e}"


def _read_uploaded_file(file_id: str) -> str:
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

SYSTEM_PROMPT = (
    "You are a helpful assistant that can read and analyze files. "
    "When the user asks about a file, use the read_file tool with an ABSOLUTE path. "
    "Be concise and accurate."
)


def _resolve_file_references(messages: list[dict]) -> list[dict]:
    """
    Resolve ``{"type": "file", "file": {"file_id": "..."}}`` content parts.

    OpenAI Chat Completions supports file references in multi-part content
    (see https://platform.openai.com/docs/api-reference/chat/create).
    This replaces each file reference with the actual text content from the
    upload store so the agent sees it inline.
    """
    resolved: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            resolved.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            resolved.append(msg)
            continue

        new_parts: list[dict] = []
        for part in content:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            if part.get("type") == "file":
                file_info = part.get("file", {})
                fid = file_info.get("file_id", "") if isinstance(file_info, dict) else ""
                if fid:
                    text = _read_uploaded_file(fid)
                    filename = _file_store.get(fid, {}).get("filename", fid)
                    new_parts.append({"type": "text", "text": f"[File: {filename}]\n{text}"})
                    log.info("Resolved file reference {} in chat message", fid)
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)
        resolved.append({**msg, "content": new_parts})
    return resolved


async def _strip_tool_calls(gen: AsyncGenerator) -> AsyncGenerator:
    """
    Filter internal Agent tool calls from the stream.

    The Haystack Agent executes tools server-side, but fastapi-openai-compat
    v1.1.0+ translates StreamingChunk.tool_calls into SSE function-call events.
    Agentic clients would treat those as client-side calls and loop forever.
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
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt=SYSTEM_PROMPT,
            tools=[read_file_tool],
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

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        resolved = _resolve_file_references(messages)
        chat_messages = [
            ChatMessage.from_openai_dict_format(m) if isinstance(m, dict) else m
            for m in resolved
        ]
        log.info("Running agent (chat) with {} message(s)", len(chat_messages))

        gen = async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={"messages": chat_messages},
        )
        return _strip_tool_calls(gen)
