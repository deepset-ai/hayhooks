from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseDelta:
    message_id: str
    actor: str
    text: str


@dataclass(frozen=True)
class ResponseComplete:
    message_id: str
    actor: str
    text: str
    final: bool = False


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


ChatEvent = ResponseDelta | ResponseComplete | ToolStarted | ToolFinished
