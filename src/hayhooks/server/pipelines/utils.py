import json
from typing import Any
from uuid import uuid4

from fastapi_openai_compat import Message
from haystack.dataclasses import ChatMessage, ToolCall

from hayhooks.server.pipelines.streaming import (
    OnPipelineEnd,
    OnToolCallEnd,
    OnToolCallStart,
    ToolCallbackReturn,
    async_streaming_generator,
    find_all_streaming_components,
    is_streaming_component,
    parse_streaming_components_setting,
    streaming_generator,
)

__all__ = [
    "OnPipelineEnd",
    "OnToolCallEnd",
    "OnToolCallStart",
    "ToolCallbackReturn",
    "async_streaming_generator",
    "chat_messages_from_openai_response",
    "find_all_streaming_components",
    "get_content",
    "get_input_files",
    "get_last_user_input_text",
    "get_last_user_message",
    "is_streaming_component",
    "is_user_message",
    "parse_streaming_components_setting",
    "streaming_generator",
]


def is_user_message(msg: Message | dict) -> bool:
    if isinstance(msg, Message):
        return msg.role == "user"
    return msg.get("role") == "user"


def get_content(msg: Message | dict) -> str:
    if isinstance(msg, Message):
        return msg.content or ""
    return msg.get("content", "")


def get_last_user_message(messages: list[Message | dict]) -> str | None:
    user_messages = (msg for msg in reversed(messages) if is_user_message(msg))

    for message in user_messages:
        return get_content(message)

    return None


def get_input_files(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract all ``input_file`` content parts from OpenAI Responses API input items.

    Returns a list of dicts, each containing at least ``"file_id"`` (and any
    other keys present on the content part, e.g. ``"filename"``).  Items that
    are not user messages or content parts that are not ``input_file`` are
    silently skipped.
    """
    files: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        files.extend(
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "input_file" and part.get("file_id")
        )
    return files


def get_last_user_input_text(input_items: list[dict[str, Any]]) -> str | None:
    """
    Extract the last user text from OpenAI Responses API input items.

    Input items use a different structure than chat messages:
    ``[{"role": "user", "type": "message", "content": [{"type": "input_text", "text": "..."}]}]``

    String shorthand (``"content": "Hello"``) is also supported.
    """
    for item in reversed(input_items):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        if item.get("type") not in ("message", None):
            continue

        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for content_part in reversed(content):
                if isinstance(content_part, dict) and content_part.get("type") == "input_text":  # type: ignore[invalid-argument-type]
                    # InputItem is dict[str, Any]; ty narrows Any through isinstance to object, losing key type info
                    return content_part.get("text")  # ty: ignore[invalid-argument-type]
    return None


def _content_to_text(content: object) -> str:
    """Flatten a Responses API content field into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)


def _parse_tool_arguments(raw: object) -> dict:
    """Parse tool-call arguments that may arrive as a dict or a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_call_id(item: dict[str, Any], *, generate_if_missing: bool) -> str | None:
    """Resolve ``call_id`` or ``id`` from a Responses item, optionally generating one."""
    for key in ("call_id", "id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    if generate_if_missing:
        return f"call_{uuid4().hex[:24]}"
    return None


def chat_messages_from_openai_response(input_items: list[dict[str, Any]]) -> list[ChatMessage]:
    """
    Convert OpenAI Responses API input items to Haystack ``ChatMessage`` objects.

    Accepts both ``InputItem`` dicts (from ``fastapi-openai-compat``) and plain
    ``dict[str, Any]`` — they are the same underlying type.

    Handles three item types:

    - **message** (user / system / developer / assistant) → ``ChatMessage``
    - **function_call** → ``ChatMessage.from_assistant`` with a ``ToolCall``
    - **function_call_output** → ``ChatMessage.from_tool`` with matching origin

    This is the Responses API counterpart of building a ``messages`` list from
    the Chat Completions format.  It supports the full Codex multi-turn loop
    where the client sends back ``function_call`` / ``function_call_output``
    items alongside regular messages.
    """
    messages: list[ChatMessage] = []
    tool_calls_by_id: dict[str, ToolCall] = {}

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "function_call":
            arguments = _parse_tool_arguments(item.get("arguments", "{}"))
            call_id = _extract_call_id(item, generate_if_missing=True)
            tool_name = item.get("name") or "function"

            tool_call = ToolCall(tool_name=tool_name, arguments=arguments, id=call_id)
            tool_calls_by_id[call_id] = tool_call  # type: ignore[index]
            messages.append(ChatMessage.from_assistant(tool_calls=[tool_call]))
            continue

        if item_type == "function_call_output":
            call_id = _extract_call_id(item, generate_if_missing=False)
            raw_output = item.get("output", "")
            output = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)

            if isinstance(call_id, str) and call_id in tool_calls_by_id:
                origin = tool_calls_by_id[call_id]
            else:
                origin = ToolCall(tool_name="function", arguments={}, id=str(call_id) if call_id else None)

            messages.append(ChatMessage.from_tool(tool_result=output, origin=origin))
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
