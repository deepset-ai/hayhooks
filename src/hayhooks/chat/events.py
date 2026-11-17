from dataclasses import dataclass

Actor = str


@dataclass(frozen=True)
class ResponseDelta:
    message_id: str
    actor: Actor
    text: str


@dataclass(frozen=True)
class ResponseComplete:
    message_id: str
    actor: Actor
    text: str
    final: bool = False


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    actor: Actor
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolFinished:
    call_id: str
    result: str
    error: bool = False


ChatEvent = ResponseDelta | ResponseComplete | ToolStarted | ToolFinished
