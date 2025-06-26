from typing import Dict, Any, List, Union, Literal
from pydantic import BaseModel


class StatusEventData(BaseModel):
    description: str
    done: bool = False
    hidden: bool = False


class MessageEventData(BaseModel):
    content: str


class ChatTitleEventData(BaseModel):
    title: str


class ChatTagsEventData(BaseModel):
    tags: List[str]


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
        ChatTitleEventData,
        ChatTagsEventData,
        NotificationEventData,
        Dict[str, Any],  # Fallback for custom data
    ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for Open WebUI."""
        return self.model_dump()


def create_status_event(description: str, done: bool = False, hidden: bool = False) -> OpenWebUIEvent:
    """Create a status event to show progress updates in the UI."""
    return OpenWebUIEvent(type="status", data=StatusEventData(description=description, done=done, hidden=hidden))


def create_chat_completion_event(data: Dict[str, Any]) -> OpenWebUIEvent:
    """Create a chat completion event to provide completion results."""
    return OpenWebUIEvent(type="chat:completion", data=data)


def create_message_delta_event(content: str) -> OpenWebUIEvent:
    """Create a message delta event to stream/append content to the current message."""
    return OpenWebUIEvent(type="chat:message:delta", data=MessageEventData(content=content))


def create_message_event(content: str) -> OpenWebUIEvent:
    """Create a message event to append content to the current message."""
    return OpenWebUIEvent(type="message", data=MessageEventData(content=content))


def create_message_replace_event(content: str) -> OpenWebUIEvent:
    """Create a message replace event to completely replace the current message content."""
    return OpenWebUIEvent(type="chat:message", data=MessageEventData(content=content))


def create_replace_event(content: str) -> OpenWebUIEvent:
    """Create a replace event to completely replace the current message content."""
    return OpenWebUIEvent(type="replace", data=MessageEventData(content=content))


def create_chat_title_event(title: str) -> OpenWebUIEvent:
    """Create a chat title event to set or update the conversation title."""
    return OpenWebUIEvent(type="chat:title", data=ChatTitleEventData(title=title))


def create_chat_tags_event(tags: List[str]) -> OpenWebUIEvent:
    """Create a chat tags event to update the set of tags for the conversation."""
    return OpenWebUIEvent(type="chat:tags", data=ChatTagsEventData(tags=tags))


def create_source_event(source_data: Dict[str, Any]) -> OpenWebUIEvent:
    """Create a source event to add a reference/citation or code execution result."""
    return OpenWebUIEvent(type="source", data=source_data)


def create_citation_event(citation_data: Dict[str, Any]) -> OpenWebUIEvent:
    """Create a citation event to add a reference/citation to the message."""
    return OpenWebUIEvent(type="citation", data=citation_data)


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
