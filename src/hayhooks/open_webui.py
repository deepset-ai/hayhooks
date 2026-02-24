from typing import Any

from pydantic import BaseModel

from hayhooks.events import (
    NotificationEventData,
    PipelineEvent,
    StatusEventData,
    create_notification_event,
    create_status_event,
)

# Re-export shared symbols so that existing ``from hayhooks.open_webui import …``
# statements continue to work.
__all__ = [
    "NotificationEventData",
    "OpenWebUIEvent",
    "PipelineEvent",
    "StatusEventData",
    "create_chat_completion_event",
    "create_details_tag",
    "create_message_event",
    "create_notification_event",
    "create_replace_event",
    "create_source_event",
    "create_status_event",
]


class MessageEventData(BaseModel):
    content: str


class OpenWebUIEvent(PipelineEvent):
    """
    Event targeting the Open WebUI frontend.

    Full event documentation: https://docs.openwebui.com/features/plugin/events

    The event system makes the chat experience richer and more interactive by
    enabling real-time communication between your backend logic and the Open
    WebUI interface.
    """

    data: StatusEventData | MessageEventData | NotificationEventData | dict[str, Any]


def create_chat_completion_event(data: dict[str, Any]) -> OpenWebUIEvent:
    """Create a chat completion event to provide completion results."""
    return OpenWebUIEvent(type="chat:completion", data=data)


def create_message_event(content: str) -> OpenWebUIEvent:
    """Create a message event to append content to the current message."""
    return OpenWebUIEvent(type="message", data=MessageEventData(content=content))


def create_replace_event(content: str) -> OpenWebUIEvent:
    """Create a replace event to completely replace the current message content."""
    return OpenWebUIEvent(type="replace", data=MessageEventData(content=content))


def create_source_event(source_data: dict[str, Any]) -> OpenWebUIEvent:
    """Create a source event to add a reference/citation or code execution result."""
    return OpenWebUIEvent(type="source", data=source_data)


def create_details_tag(
    tool_name: str,
    summary: str,
    content: str,
) -> str:
    """
    Create a details event to show tool call results.

    This is not an OpenWebUIEvent, but a string that can be rendered
    by Open WebUI as a details block.
    """
    return f'<details type="{tool_name}" done="true">\n<summary>{summary}</summary>\n\n{content}\n</details>\n\n'
