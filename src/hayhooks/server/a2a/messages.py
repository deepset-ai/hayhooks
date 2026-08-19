"""Conversions between A2A SDK messages and Haystack chat messages."""

from __future__ import annotations

from typing import Any

from haystack.dataclasses import ChatMessage

from hayhooks.server.a2a.imports import RequestContext, Role, TaskState, get_message_text


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


def _haystack_message(role: str, text: str) -> Any:
    return ChatMessage.from_assistant(text) if role == "assistant" else ChatMessage.from_user(text)


def build_haystack_messages(context: RequestContext) -> list[Any]:
    return [_haystack_message(message["role"], message["content"]) for message in build_openai_messages(context)]


def build_haystack_task_messages(task: Any) -> list[Any]:
    """Rebuild the original durable input from persisted A2A history."""
    return [
        _haystack_message("assistant" if message.role == Role.ROLE_AGENT else "user", text)
        for message in task.history
        if (text := get_message_text(message))
    ]


def build_haystack_resume_messages(context: RequestContext) -> list[Any]:
    """Convert only the follow-up turn; recovered Agent state already contains history."""
    if context.message is None:
        return []

    text = get_message_text(context.message)
    if not text:
        return []

    role = "assistant" if context.message.role == Role.ROLE_AGENT else "user"
    return [_haystack_message(role, text)]


def task_is_terminal(task: Any) -> bool:
    return task.HasField("status") and task.status.state in {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_REJECTED,
    }


def task_matches_filters(task: Any, params: Any, timestamp_after: Any | None) -> bool:
    if params.context_id and task.context_id != params.context_id:
        return False
    if params.status and task.status.state != params.status:
        return False
    return timestamp_after is None or (
        task.HasField("status")
        and task.status.HasField("timestamp")
        and (task.status.timestamp.seconds, task.status.timestamp.nanos)
        >= (timestamp_after.seconds, timestamp_after.nanos)
    )


__all__ = [
    "build_haystack_messages",
    "build_haystack_resume_messages",
    "build_haystack_task_messages",
    "build_openai_messages",
    "task_is_terminal",
    "task_matches_filters",
]
