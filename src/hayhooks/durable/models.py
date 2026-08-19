"""
Persisted durable-execution model and JSON-safe value helpers.

This module is transport- and storage-neutral.  REST, A2A, Redis, and
Haystack-specific code build on these types rather than extending them.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeAlias, cast

from pydantic import BaseModel, Field

from hayhooks.durable.engine import ExecutionLeaseLostError as _ExecutionLeaseLostError
from hayhooks.durable.engine import ExecutionStatus, normalize_cancellation_reason

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
ExecutionLeaseLostError = _ExecutionLeaseLostError
DEFAULT_MAX_RECORD_BYTES = 1_000_000
DEFAULT_MAX_PROGRESS_EVENTS = 100
DEFAULT_MAX_PROGRESS_BYTES = 8_192
_SENSITIVE_NAME = r"(?:api[_ -]?key|access[_ -]?token|authorization|bearer|password|secret)"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def json_safe(value: Any) -> JsonValue:
    """Convert common application values to a JSON-compatible value."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset | deque):
        return [json_safe(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return json_safe(converter())
    msg = f"{type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def validate_json(value: Any, *, limit: int | None, label: str) -> JsonValue:
    """Normalize a value and, when requested, bound its encoded size."""
    try:
        safe = json_safe(value)
        encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        msg = f"{label} must be JSON serializable"
        raise ValueError(msg) from error
    if limit is not None and len(encoded.encode("utf-8")) > limit:
        msg = f"{label} exceeds the {limit}-byte durable execution limit"
        raise ExecutionRecordSizeError(msg)
    return safe


def _sanitize_error_message(error: BaseException) -> str:
    message = str(error)
    patterns = (
        (r"(?i)(authorization\s*:\s*)bearer\s+[A-Za-z0-9._~+/=-]+", r"\1<redacted>"),
        (rf"(?i)([?&]{_SENSITIVE_NAME}=)[^&#\s]+", r"\1<redacted>"),
        (rf'(?i)(["\']{_SENSITIVE_NAME}["\']\s*:\s*)["\'][^"\']*["\']', r'\1"<redacted>"'),
        (rf"(?i)({_SENSITIVE_NAME})\s*[:=]\s*[^\s,;&]+", r"\1=<redacted>"),
    )
    for pattern, replacement in patterns:
        message = re.sub(pattern, replacement, message)
    return message[:2_000]


class ExecutionStoreError(RuntimeError):
    """A storage or lease-heartbeat operation failed transiently."""


class ExecutionAdmissionError(RuntimeError):
    """A durable store rejected new work because a configured capacity limit is full."""

    retry_after_seconds = 1

    def __init__(self, policy: str) -> None:
        self.policy = policy
        super().__init__(f"Durable execution admission rejected by {policy} capacity limit")


class ExecutionRecordSizeError(ValueError):
    """An execution record cannot fit within its configured persistence limit."""


class ExecutionCanceledError(RuntimeError):
    """Raised at a cooperative cancellation boundary."""


class ExecutionSuspendedError(RuntimeError):
    """Internal signal used after a claim atomically enters ``waiting``."""


class RetryableExecutionError(RuntimeError):
    """Ask the manager to persist retry metadata and redeliver later."""

    def __init__(self, message: str, *, delay: float = 0.0) -> None:
        super().__init__(message)
        self.delay = max(0.0, delay)


class ExecutionProgress(BaseModel):
    """Sanitized client-visible progress event."""

    sequence: int
    kind: str
    message: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Safe durable REST/A2A execution projection."""

    execution_id: str
    status: ExecutionStatus
    attempt: int
    sequence: int
    progress: list[ExecutionProgress]
    result: Any | None = None
    error: dict[str, Any] | None = None
    waiting: dict[str, Any] | None = None
    cancellation_requested_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    links: dict[str, str] = Field(default_factory=dict)


class ExecutionKind(str, Enum):
    """The Haystack object adapter selected for an execution."""

    PIPELINE = "pipeline"
    AGENT = "agent"


@dataclass
class ExecutionError:
    """Sanitized persisted failure metadata."""

    type: str
    message: str
    retryable: bool = False
    code: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {"type": self.type, "message": self.message, "retryable": self.retryable, "code": self.code}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionError:
        return cls(
            type=str(value.get("type", "ExecutionError")),
            message=str(value.get("message", ""))[:2_000],
            retryable=bool(value.get("retryable", False)),
            code=str(value["code"]) if value.get("code") is not None else None,
        )

    @classmethod
    def from_exception(cls, error: BaseException, *, retryable: bool = False) -> ExecutionError:
        code = getattr(error, "code", None)
        return cls(
            type=type(error).__name__,
            message=_sanitize_error_message(error),
            retryable=retryable,
            code=str(code) if code is not None else None,
        )


@dataclass
class ExecutionProgressEvent:
    """A bounded, safe event visible through REST and A2A projections."""

    sequence: int
    message: str
    timestamp: datetime = field(default_factory=utc_now)
    kind: str = "progress"
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1:
            msg = "progress event sequence numbers must start at one"
            raise ValueError(msg)
        self.timestamp = _as_utc(self.timestamp)
        self.kind = str(self.kind)[:128]
        self.message = str(self.message)[:2_000]
        self.metadata = cast(
            dict[str, JsonValue],
            validate_json(self.metadata, limit=None, label="progress metadata"),
        )
        validate_json(self.to_dict(), limit=DEFAULT_MAX_PROGRESS_BYTES, label="progress event")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionProgressEvent:
        timestamp = value.get("timestamp")
        return cls(
            sequence=int(value["sequence"]),
            kind=str(value.get("kind", "progress")),
            message=str(value.get("message", "")),
            timestamp=datetime.fromisoformat(str(timestamp)) if timestamp else utc_now(),
            metadata=dict(cast(Mapping[str, Any], value.get("metadata", {}))),
        )


@dataclass
class ExecutionCheckpoint:
    """Private recovery data, discriminated by the selected adapter."""

    kind: ExecutionKind
    data: dict[str, JsonValue]

    def __post_init__(self) -> None:
        self.kind = ExecutionKind(self.kind)
        self.data = cast(dict[str, JsonValue], validate_json(self.data, limit=None, label="checkpoint"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"kind": self.kind.value, "data": self.data}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionCheckpoint:
        return cls(kind=ExecutionKind(str(value["kind"])), data=dict(cast(Mapping[str, Any], value["data"])))


@dataclass
class ExecutionRecord:
    """Hayhooks' versioned, private durable execution envelope."""

    execution_id: str
    execution_kind: ExecutionKind
    deployment_name: str
    definition_revision: str
    validated_input: dict[str, JsonValue]
    operation_fingerprint: str = ""
    owner_id: str | None = None
    status: ExecutionStatus = ExecutionStatus.QUEUED
    sequence: int = 0
    attempt: int = 0
    checkpoint: ExecutionCheckpoint | None = None
    application_state: dict[str, JsonValue] = field(default_factory=dict)
    wait: dict[str, JsonValue] | None = None
    progress: list[ExecutionProgressEvent] = field(default_factory=list)
    result: JsonValue | None = None
    error: ExecutionError | None = None
    cancel_requested_at: datetime | None = None
    cancel_reason: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    max_progress_events: int = field(default=DEFAULT_MAX_PROGRESS_EVENTS, repr=False, compare=False)
    max_record_bytes: int = field(default=DEFAULT_MAX_RECORD_BYTES, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.execution_id or not self.deployment_name or not self.definition_revision:
            msg = "execution_id, deployment_name, and definition_revision must be non-empty"
            raise ValueError(msg)
        self.execution_kind = ExecutionKind(self.execution_kind)
        self.status = ExecutionStatus(self.status)
        self.created_at = _as_utc(self.created_at)
        self.updated_at = _as_utc(self.updated_at)
        if self.cancel_requested_at is not None:
            self.cancel_requested_at = _as_utc(self.cancel_requested_at)
        self.cancel_reason = normalize_cancellation_reason(self.cancel_reason)
        self.validated_input = self._bounded_dict(self.validated_input, "validated input")
        self.application_state = self._bounded_dict(self.application_state, "application state")
        if self.wait is not None:
            self.wait = self._bounded_dict(self.wait, "wait")
        if self.result is not None:
            self.result = validate_json(self.result, limit=self.max_record_bytes, label="result")
        if self.checkpoint is not None:
            if not isinstance(self.checkpoint, ExecutionCheckpoint):
                self.checkpoint = ExecutionCheckpoint.from_dict(cast(Mapping[str, Any], self.checkpoint))
            self.checkpoint.data = self._bounded_dict(self.checkpoint.data, "checkpoint")
        if self.error is not None and not isinstance(self.error, ExecutionError):
            self.error = ExecutionError.from_dict(cast(Mapping[str, Any], self.error))
        self.progress = [
            event
            if isinstance(event, ExecutionProgressEvent)
            else ExecutionProgressEvent.from_dict(cast(Mapping[str, Any], event))
            for event in self.progress
        ]
        self._trim_progress()

    @property
    def terminal(self) -> bool:
        return self.status.terminal

    def _bounded_dict(self, value: Any, label: str) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], validate_json(value, limit=self.max_record_bytes, label=label))

    def touch(self) -> None:
        self.sequence += 1
        self.updated_at = utc_now()

    def append_progress(
        self, message: str, *, kind: str = "progress", metadata: Mapping[str, Any] | None = None
    ) -> ExecutionProgressEvent:
        event = ExecutionProgressEvent(
            sequence=self.progress[-1].sequence + 1 if self.progress else 1,
            message=message,
            kind=kind,
            metadata=dict(metadata or {}),
        )
        self.progress.append(event)
        self._trim_progress()
        self.touch()
        return event

    def mark_canceled(self) -> None:
        """Mark this execution as canceled unless it is already terminal."""
        if not self.terminal or self.status == ExecutionStatus.CANCELED:
            if self.cancel_requested_at is None:
                self.cancel_requested_at = utc_now()
            self.status = ExecutionStatus.CANCELED
            self.error = None
            self.result = None
            self.wait = None
            self.touch()

    def mark_failed(self, error: BaseException | ExecutionError) -> None:
        self.status = ExecutionStatus.FAILED
        self.error = error if isinstance(error, ExecutionError) else ExecutionError.from_exception(error)
        self.wait = None
        self.touch()

    def safe_view(self, *, links: Mapping[str, str] | None = None) -> dict[str, JsonValue]:
        """Return only the public projection; never expose inputs/checkpoints."""
        waiting = None
        if self.wait is not None:
            waiting = {key: self.wait[key] for key in ("kind", "message", "expected_input_schema") if key in self.wait}
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "attempt": self.attempt,
            "sequence": self.sequence,
            "progress": [event.to_dict() for event in self.progress],
            "result": self.result,
            "error": self.error.to_dict() if self.error else None,
            "waiting": waiting,
            "cancellation_requested_at": (self.cancel_requested_at.isoformat() if self.cancel_requested_at else None),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "links": dict(links or {}),
        }

    def _trim_progress(self) -> None:
        self.max_progress_events = max(1, self.max_progress_events)
        if len(self.progress) > self.max_progress_events:
            self.progress = self.progress[-self.max_progress_events :]
