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
