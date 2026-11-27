from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseDelta:
    segment: int
    text: str


@dataclass(frozen=True)
class SideMessage:
    message_id: str
    actor: str
    text: str


@dataclass(frozen=True)
class ReasoningSummary:
    text: str


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    actor: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolFinished:
    call_id: str
    result: str
    error: bool = False


ChatEvent = ResponseDelta | SideMessage | ReasoningSummary | ToolStarted | ToolFinished
