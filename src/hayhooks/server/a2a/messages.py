"""Conversions between A2A SDK messages and Haystack chat messages."""

from __future__ import annotations

from typing import Any

from hayhooks.server.a2a.imports import RequestContext, Role, get_message_text


def build_openai_messages(context: RequestContext) -> list[dict[str, str]]:
    """Map A2A history plus the current message to OpenAI-compatible messages."""
    messages: list[dict[str, str]] = []
    history = list(context.current_task.history) if context.current_task else []
    history_ids = {message.message_id for message in history}
    for message in history:
        text = get_message_text(message)
        if text:
            messages.append({"role": "assistant" if message.role == Role.ROLE_AGENT else "user", "content": text})
    if context.message is not None and context.message.message_id not in history_ids:
        text = get_message_text(context.message)
        if text:
            messages.append({"role": "user", "content": text})
    return messages


def build_haystack_messages(context: RequestContext) -> list[Any]:
    from haystack.dataclasses import ChatMessage

    converted: list[Any] = []
    for message in build_openai_messages(context):
        if message["role"] == "assistant":
            converted.append(ChatMessage.from_assistant(message["content"]))
        else:
            converted.append(ChatMessage.from_user(message["content"]))
    return converted


def build_haystack_resume_messages(context: RequestContext) -> list[Any]:
    """Convert only the follow-up turn; recovered Agent state already contains history."""
    from haystack.dataclasses import ChatMessage

    if context.message is None:
        return []
    text = get_message_text(context.message)
    if not text:
        return []
    if context.message.role == Role.ROLE_AGENT:
        return [ChatMessage.from_assistant(text)]
    return [ChatMessage.from_user(text)]


__all__ = ["build_haystack_messages", "build_haystack_resume_messages", "build_openai_messages"]
