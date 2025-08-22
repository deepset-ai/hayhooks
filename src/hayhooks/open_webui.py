from typing import Any, Literal, Union

from pydantic import BaseModel


class StatusEventData(BaseModel):
    description: str
    done: bool = False
    hidden: bool = False


class MessageEventData(BaseModel):
    content: str


class NotificationEventData(BaseModel):
    type: Literal["info", "success", "warning", "error"] = "info"
    content: str


class OpenWebUIEvent(BaseModel):
    """
    OpenWebUIEvent is a Pydantic model that represents an event that can be sent to the Open WebUI.

    It is used to send events to the Open WebUI from Hayhooks.

    Full event documentation: https://docs.openwebui.com/features/plugin/events

    The event system makes the chat experience richer and more interactive by enabling real-time
    communication between your backend logic and the Open WebUI interface.
    """

    type: str
    data: Union[
        StatusEventData,
        MessageEventData,
        NotificationEventData,
        dict[str, Any],  # Fallback for custom data
    ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for Open WebUI."""
        return self.model_dump()


def create_status_event(description: str, done: bool = False, hidden: bool = False) -> OpenWebUIEvent:
    """Create a status event to show progress updates in the UI."""
    return OpenWebUIEvent(type="status", data=StatusEventData(description=description, done=done, hidden=hidden))


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


def create_notification_event(
    content: str, notification_type: Literal["info", "success", "warning", "error"] = "info"
) -> OpenWebUIEvent:
    """
    Create a notification event for showing toast notifications.

    Args:
        content: The notification message
        notification_type: Type of notification ("info", "success", "warning", "error")
    """
    return OpenWebUIEvent(type="notification", data=NotificationEventData(type=notification_type, content=content))


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
