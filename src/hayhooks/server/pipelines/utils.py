from typing import Any

from fastapi_openai_compat import Message

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
