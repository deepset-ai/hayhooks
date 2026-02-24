from typing import Any, Literal

from pydantic import BaseModel


class StatusEventData(BaseModel):
    description: str
    done: bool = False
    hidden: bool = False


class NotificationEventData(BaseModel):
    type: Literal["info", "success", "warning", "error"] = "info"
    content: str


class PipelineEvent(BaseModel):
    """
    Base event model for all pipeline UI events.

    Subclassed by ``OpenWebUIEvent`` (Open WebUI-specific events) and
    ``ChainlitEvent`` (Chainlit-specific events).  Shared event helpers
    such as ``create_status_event`` and ``create_notification_event`` return
    instances of this base class because they are consumed by both frontends.

    The ``to_event_dict`` method is used by *fastapi-openai-compat* via duck
    typing: any yielded object with a callable ``to_event_dict`` attribute is
    serialised as a custom SSE event rather than an OpenAI content chunk.
    """

    type: str
    data: StatusEventData | NotificationEventData | dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return self.model_dump()

    def to_event_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format for SSE custom event serialization.

        Used by fastapi_openai_compat to detect and serialize custom events
        via duck typing (objects with a to_event_dict method are treated as
        custom SSE events rather than content chunks).
        """
        return self.model_dump()


def create_status_event(description: str, done: bool = False, hidden: bool = False) -> PipelineEvent:
    """Create a status event to show progress updates in the UI."""
    return PipelineEvent(type="status", data=StatusEventData(description=description, done=done, hidden=hidden))


def create_notification_event(
    content: str, notification_type: Literal["info", "success", "warning", "error"] = "info"
) -> PipelineEvent:
    """
    Create a notification event for showing toast notifications.

    Args:
        content: The notification message
        notification_type: Type of notification ("info", "success", "warning", "error")
    """
    return PipelineEvent(type="notification", data=NotificationEventData(type=notification_type, content=content))
