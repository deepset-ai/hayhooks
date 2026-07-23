"""
Internal, transport-neutral durable execution core.

This module deliberately owns the persisted envelope, worker lifecycle, and
store protocol.  It does not import FastAPI, A2A, Redis, or Haystack so REST
and A2A remain projections of exactly the same execution.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import re
import socket
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

from hayhooks.server.logger import log

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
RecordRunner: TypeAlias = Callable[["DurableContext"], Awaitable[JsonValue]]

RECORD_SCHEMA_VERSION = 2
DEFAULT_MAX_RECORD_BYTES = 1_000_000
DEFAULT_MAX_PROGRESS_EVENTS = 100
DEFAULT_MAX_PROGRESS_BYTES = 8_192
_RESUME_INPUT_KEY = "__hayhooks_resume_input"
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


def validate_json(value: Any, *, limit: int, label: str) -> JsonValue:
    """Normalize and bound a value that will be persisted in an execution."""
    try:
        safe = json_safe(value)
        encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        msg = f"{label} must be JSON serializable"
        raise ValueError(msg) from error
    if len(encoded.encode("utf-8")) > limit:
        msg = f"{label} exceeds the {limit}-byte durable execution limit"
        raise ValueError(msg)
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


class ExecutionLeaseLostError(RuntimeError):
    """A stale worker tried to write a fenced execution record."""


class ExecutionStoreError(RuntimeError):
    """A storage or lease-heartbeat operation failed transiently."""


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


class ExecutionKind(str, Enum):
    """The Haystack object adapter selected for an execution."""

    PIPELINE = "pipeline"
    AGENT = "agent"


class ExecutionStatus(str, Enum):
    """Public lifecycle state for one durable execution."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELED}


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
            validate_json(self.metadata, limit=DEFAULT_MAX_PROGRESS_BYTES, label="progress metadata"),
        )

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
        self.data = cast(
            dict[str, JsonValue], validate_json(self.data, limit=DEFAULT_MAX_RECORD_BYTES, label="checkpoint")
        )

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
    schema_version: int = RECORD_SCHEMA_VERSION
    status: ExecutionStatus = ExecutionStatus.QUEUED
    sequence: int = 0
    attempt: int = 0
    checkpoint: ExecutionCheckpoint | None = None
    application_state: dict[str, JsonValue] = field(default_factory=dict)
    wait: dict[str, JsonValue] | None = None
    progress: list[ExecutionProgressEvent] = field(default_factory=list)
    result: JsonValue | None = None
    error: ExecutionError | None = None
    last_retry_error: ExecutionError | None = None
    retry_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    cancel_reason: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    max_progress_events: int = field(default=DEFAULT_MAX_PROGRESS_EVENTS, repr=False, compare=False)
    max_record_bytes: int = field(default=DEFAULT_MAX_RECORD_BYTES, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != RECORD_SCHEMA_VERSION:
            msg = (
                f"Unsupported durable execution schema version {self.schema_version}. "
                "Remove unreleased prototype records before enabling durable execution."
            )
            raise ValueError(msg)
        if not self.execution_id or not self.deployment_name or not self.definition_revision:
            msg = "execution_id, deployment_name, and definition_revision must be non-empty"
            raise ValueError(msg)
        self.execution_kind = ExecutionKind(self.execution_kind)
        self.status = ExecutionStatus(self.status)
        self.created_at = _as_utc(self.created_at)
        self.updated_at = _as_utc(self.updated_at)
        if self.cancel_requested_at is not None:
            self.cancel_requested_at = _as_utc(self.cancel_requested_at)
        self.cancel_reason = str(self.cancel_reason)[:2_000] if self.cancel_reason else None
        self.validated_input = cast(
            dict[str, JsonValue],
            validate_json(self.validated_input, limit=self.max_record_bytes, label="validated input"),
        )
        self.application_state = cast(
            dict[str, JsonValue],
            validate_json(self.application_state, limit=self.max_record_bytes, label="application state"),
        )
        if self.wait is not None:
            self.wait = cast(dict[str, JsonValue], validate_json(self.wait, limit=self.max_record_bytes, label="wait"))
        if self.result is not None:
            self.result = validate_json(self.result, limit=self.max_record_bytes, label="result")
        if self.checkpoint is not None and not isinstance(self.checkpoint, ExecutionCheckpoint):
            self.checkpoint = ExecutionCheckpoint.from_dict(cast(Mapping[str, Any], self.checkpoint))
        if self.error is not None and not isinstance(self.error, ExecutionError):
            self.error = ExecutionError.from_dict(cast(Mapping[str, Any], self.error))
        if self.last_retry_error is not None and not isinstance(self.last_retry_error, ExecutionError):
            self.last_retry_error = ExecutionError.from_dict(cast(Mapping[str, Any], self.last_retry_error))
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

    def mark_canceled(self, *, force: bool = False) -> None:
        """Mark canceled, optionally allowing an atomic store to win a terminal race."""
        if force or not self.terminal or self.status == ExecutionStatus.CANCELED:
            if self.cancel_requested_at is None:
                self.cancel_requested_at = utc_now()
            self.status = ExecutionStatus.CANCELED
            self.error = None
            self.result = None
            self.retry_at = None
            self.wait = None
            self.touch()

    def request_cancellation(self, reason: str | None = None) -> bool:
        """Persist an idempotent cancellation request on a nonterminal record."""
        if self.terminal:
            return self.status is ExecutionStatus.CANCELED
        if self.cancel_requested_at is not None:
            return True
        self.cancel_requested_at = utc_now()
        self.cancel_reason = str(reason)[:2_000] if reason else None
        self.append_progress("Cancellation requested", kind="cancellation_requested")
        return True

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

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "execution_kind": self.execution_kind.value,
            "deployment_name": self.deployment_name,
            "definition_revision": self.definition_revision,
            "validated_input": self.validated_input,
            "operation_fingerprint": self.operation_fingerprint,
            "owner_id": self.owner_id,
            "status": self.status.value,
            "sequence": self.sequence,
            "attempt": self.attempt,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "application_state": self.application_state,
            "wait": self.wait,
            "bounded_progress": [event.to_dict() for event in self.progress],
            "result": self.result,
            "error": self.error.to_dict() if self.error else None,
            "last_retry_error": self.last_retry_error.to_dict() if self.last_retry_error else None,
            "retry_at": self.retry_at.isoformat() if self.retry_at else None,
            "cancel_requested_at": (self.cancel_requested_at.isoformat() if self.cancel_requested_at else None),
            "cancel_reason": self.cancel_reason,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_json(self) -> str:
        while True:
            payload = self.to_dict()
            try:
                validate_json(payload, limit=self.max_record_bytes, label="execution record")
            except ValueError as error:
                if self.progress:
                    self.progress.pop(0)
                    continue
                msg = f"execution record exceeds the {self.max_record_bytes}-byte durable execution limit"
                raise ExecutionRecordSizeError(msg) from error
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        max_progress_events: int = DEFAULT_MAX_PROGRESS_EVENTS,
        max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
    ) -> ExecutionRecord:
        if "schema_version" not in value:
            msg = "Legacy durable execution records are unsupported; clear unreleased prototype records and resubmit."
            raise ValueError(msg)
        checkpoint = value.get("checkpoint")
        error = value.get("error")
        last_retry_error = value.get("last_retry_error")
        retry_at = value.get("retry_at")
        cancel_requested_at = value.get("cancel_requested_at")
        return cls(
            schema_version=int(value["schema_version"]),
            execution_id=str(value["execution_id"]),
            execution_kind=ExecutionKind(str(value["execution_kind"])),
            deployment_name=str(value["deployment_name"]),
            definition_revision=str(value["definition_revision"]),
            validated_input=dict(cast(Mapping[str, Any], value.get("validated_input", {}))),
            operation_fingerprint=str(value.get("operation_fingerprint", "")),
            owner_id=str(value["owner_id"]) if value.get("owner_id") is not None else None,
            status=ExecutionStatus(str(value.get("status", ExecutionStatus.QUEUED.value))),
            sequence=int(value.get("sequence", 0)),
            attempt=int(value.get("attempt", 0)),
            checkpoint=ExecutionCheckpoint.from_dict(cast(Mapping[str, Any], checkpoint)) if checkpoint else None,
            application_state=dict(cast(Mapping[str, Any], value.get("application_state", {}))),
            wait=dict(cast(Mapping[str, Any], value["wait"])) if value.get("wait") else None,
            progress=[
                ExecutionProgressEvent.from_dict(item)
                for item in cast(list[Mapping[str, Any]], value.get("bounded_progress", []))
            ],
            result=cast(JsonValue, value.get("result")),
            error=ExecutionError.from_dict(cast(Mapping[str, Any], error)) if error else None,
            last_retry_error=(
                ExecutionError.from_dict(cast(Mapping[str, Any], last_retry_error)) if last_retry_error else None
            ),
            retry_at=datetime.fromisoformat(str(retry_at)) if retry_at else None,
            cancel_requested_at=(datetime.fromisoformat(str(cancel_requested_at)) if cancel_requested_at else None),
            cancel_reason=str(value["cancel_reason"]) if value.get("cancel_reason") is not None else None,
            created_at=datetime.fromisoformat(str(value["created_at"])),
            updated_at=datetime.fromisoformat(str(value["updated_at"])),
            max_progress_events=max_progress_events,
            max_record_bytes=max_record_bytes,
        )

    @classmethod
    def from_json(
        cls,
        payload: str | bytes,
        *,
        max_progress_events: int = DEFAULT_MAX_PROGRESS_EVENTS,
        max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
    ) -> ExecutionRecord:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return cls.from_dict(
            cast(Mapping[str, Any], json.loads(payload)),
            max_progress_events=max_progress_events,
            max_record_bytes=max_record_bytes,
        )

    def _trim_progress(self) -> None:
        self.max_progress_events = max(1, self.max_progress_events)
        if len(self.progress) > self.max_progress_events:
            self.progress = self.progress[-self.max_progress_events :]


@runtime_checkable
class ExecutionClaim(Protocol):
    """Fenced ownership of a currently delivered execution."""

    @property
    def record(self) -> ExecutionRecord: ...

    async def checkpoint(self) -> None: ...

    async def cancellation_requested(self) -> bool: ...

    async def complete(self) -> None: ...

    async def suspend(self) -> None: ...

    async def retry(self, error: ExecutionError, *, delay: float) -> None: ...

    async def __aenter__(self) -> ExecutionClaim: ...

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...


@runtime_checkable
class DurableExecutionStore(Protocol):
    """Advanced, storage-neutral contract used by the execution manager."""

    async def initialize(self) -> None: ...

    async def submit(self, record: ExecutionRecord) -> bool: ...

    async def get(self, execution_id: str) -> ExecutionRecord | None: ...

    async def claim_next(self, worker_name: str) -> ExecutionClaim | None: ...

    async def request_cancel(self, execution_id: str, reason: str | None = None) -> bool: ...

    async def resume(self, execution_id: str, update: JsonValue | None = None) -> bool: ...

    async def retire_incompatible(self, definition_revision: str) -> int: ...

    async def operational_counts(self) -> dict[str, int]: ...

    async def close(self) -> None: ...


@runtime_checkable
class DurableExecutionStoreProvider(Protocol):
    """Application-owned factory for deployment-scoped stores."""

    def create_execution_store(self, deployment_name: str) -> DurableExecutionStore: ...

    async def close(self) -> None: ...


_active_context: ContextVar[DurableContext | None] = ContextVar("hayhooks_durable_context", default=None)


def get_current_durable_context() -> DurableContext | None:
    """Return the context active in a durable wrapper, component, hook, or tool."""
    return _active_context.get()


@contextmanager
def execution_context_scope(context: DurableContext):
    token = _active_context.set(context)
    try:
        yield
    finally:
        _active_context.reset(token)


class DurableContext:
    """Execution controls and adapters bound to one claimed record."""

    def __init__(
        self, claim: ExecutionClaim, adapter: Any, *, event_loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        self.claim = claim
        self.record = claim.record
        self.adapter = adapter
        self._event_loop = event_loop or asyncio.get_running_loop()

    @property
    def execution_id(self) -> str:
        return self.record.execution_id

    @property
    def attempt(self) -> int:
        return self.record.attempt

    @property
    def state(self) -> dict[str, JsonValue]:
        return self.record.application_state

    @property
    def resume_input(self) -> JsonValue | None:
        """Return the most recently persisted resume payload without consuming it."""
        return self.record.application_state.get(_RESUME_INPUT_KEY)

    def take_resume_input(self) -> JsonValue | None:
        """Consume the persisted resume payload exactly once within this attempt."""
        return self.record.application_state.pop(_RESUME_INPUT_KEY, None)

    async def checkpoint(self, checkpoint: ExecutionCheckpoint | None = None) -> None:
        if checkpoint is not None:
            if checkpoint.kind is not self.record.execution_kind:
                msg = (
                    f"{checkpoint.kind.value} checkpoint cannot be used for "
                    f"{self.record.execution_kind.value} execution"
                )
                raise ValueError(msg)
            self.record.checkpoint = checkpoint
        await self.claim.checkpoint()

    async def report_progress(
        self, message: str, *, kind: str = "progress", metadata: Mapping[str, Any] | None = None
    ) -> ExecutionProgressEvent:
        event = self.record.append_progress(message, kind=kind, metadata=metadata)
        await self.claim.checkpoint()
        return event

    def report_progress_sync(
        self, message: str, *, kind: str = "progress", metadata: Mapping[str, Any] | None = None
    ) -> ExecutionProgressEvent:
        return self._sync_await(self.report_progress(message, kind=kind, metadata=metadata))

    async def check_cancelled(self) -> None:
        if await self.claim.cancellation_requested():
            msg = "Durable execution cancellation was requested"
            raise ExecutionCanceledError(msg)

    def check_cancelled_sync(self) -> None:
        self._sync_await(self.check_cancelled())

    async def retry(self, message: str, *, delay: float | None = None) -> None:
        """Request a bounded, durable retry from application code."""
        raise RetryableExecutionError(message, delay=delay or 0.0)

    def retry_sync(self, message: str, *, delay: float | None = None) -> None:
        """Synchronous counterpart to :meth:`retry`."""
        raise RetryableExecutionError(message, delay=delay or 0.0)

    async def suspend(self, wait: Mapping[str, Any], *, update: Mapping[str, Any] | None = None) -> None:
        """Atomically checkpoint and move this execution to durable ``waiting``."""
        self.record.wait = cast(
            dict[str, JsonValue], validate_json(dict(wait), limit=DEFAULT_MAX_RECORD_BYTES, label="wait")
        )
        if update is not None:
            self.record.application_state.update(
                cast(
                    dict[str, JsonValue],
                    validate_json(dict(update), limit=DEFAULT_MAX_RECORD_BYTES, label="wait update"),
                )
            )
        self.record.status = ExecutionStatus.WAITING
        self.record.append_progress("Execution is waiting for resume", kind="waiting")
        await self.claim.suspend()
        raise ExecutionSuspendedError()

    def suspend_sync(self, wait: Mapping[str, Any], *, update: Mapping[str, Any] | None = None) -> None:
        self._sync_await(self.suspend(wait, update=update))

    async def run_pipeline_async(
        self, data: Mapping[str, Any], *, checkpoint_at: list[str] | None = None
    ) -> dict[str, Any]:
        return await self.adapter.run_pipeline_async(self, dict(data), checkpoint_at=checkpoint_at or [])

    def run_pipeline(self, data: Mapping[str, Any], *, checkpoint_at: list[str] | None = None) -> dict[str, Any]:
        return self.adapter.run_pipeline(self, dict(data), checkpoint_at=checkpoint_at or [])

    async def run_agent_async(self, *, messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        return await self.adapter.run_agent_async(self, messages=messages, **kwargs)

    def run_agent(self, *, messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        return self.adapter.run_agent(self, messages=messages, **kwargs)

    def _sync_await(self, awaitable: Awaitable[Any]) -> Any:
        """Bridge a synchronous wrapper/component thread to its manager loop."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._event_loop:
            msg = "A synchronous durable context method cannot run on the server event loop"
            raise RuntimeError(msg)

        async def resolve() -> Any:
            return await awaitable

        future = asyncio.run_coroutine_threadsafe(cast(Coroutine[Any, Any, Any], resolve()), self._event_loop)
        return future.result()


class InMemoryExecutionStore:
    """Volatile deterministic store for development and tests only."""

    volatile = True

    def __init__(
        self, *, max_progress_events: int = DEFAULT_MAX_PROGRESS_EVENTS, max_records: int | None = None
    ) -> None:
        self.max_progress_events = max(1, max_progress_events)
        self.max_records = max_records
        self._records: dict[str, ExecutionRecord] = {}
        self._queued: deque[str] = deque()
        self._queued_ids: set[str] = set()
        self._claims: dict[str, _InMemoryExecutionClaim] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        async with self._lock:
            self._initialized = True
            self._closed = False
            for execution_id, record in self._records.items():
                if (
                    not record.terminal
                    and record.status is not ExecutionStatus.WAITING
                    and execution_id not in self._claims
                ):
                    self._enqueue_locked(execution_id)

    async def submit(self, record: ExecutionRecord) -> bool:
        async with self._lock:
            if self._closed:
                msg = "durable execution store is closed"
                raise RuntimeError(msg)
            if record.execution_id in self._records:
                return False
            record.max_progress_events = self.max_progress_events
            record.to_json()
            self._records[record.execution_id] = _copy_record(record)
            self._enqueue_locked(record.execution_id)
            self._evict_locked()
            return True

    async def get(self, execution_id: str) -> ExecutionRecord | None:
        async with self._lock:
            record = self._records.get(execution_id)
            return _copy_record(record) if record else None

    async def claim_next(self, worker_name: str) -> ExecutionClaim | None:
        async with self._lock:
            if self._closed or not self._initialized:
                return None
            now = utc_now()
            attempts = len(self._queued)
            for _ in range(attempts):
                execution_id = self._queued.popleft()
                self._queued_ids.discard(execution_id)
                record = self._records.get(execution_id)
                if (
                    record is None
                    or record.terminal
                    or record.status is ExecutionStatus.WAITING
                    or execution_id in self._claims
                ):
                    continue
                if record.retry_at and record.retry_at > now:
                    self._enqueue_locked(execution_id)
                    continue
                record = _copy_record(record)
                record.status = ExecutionStatus.RUNNING
                record.attempt += 1
                if record.error is not None:
                    record.last_retry_error = record.error
                    record.error = None
                record.retry_at = None
                record.touch()
                claim = _InMemoryExecutionClaim(self, record, f"{worker_name}:{uuid.uuid4().hex}")
                self._claims[execution_id] = claim
                self._records[execution_id] = _copy_record(record)
                return claim
            return None

    async def request_cancel(self, execution_id: str, reason: str | None = None) -> bool:
        async with self._lock:
            persisted = self._records.get(execution_id)
            if persisted is None or persisted.terminal:
                return False
            record = _copy_record(persisted)
            record.request_cancellation(reason)
            if record.status is ExecutionStatus.WAITING:
                record.mark_canceled()
                record.to_json()
                self._records[execution_id] = _copy_record(record)
                self._queued_ids.discard(execution_id)
                return True
            record.to_json()
            self._records[execution_id] = _copy_record(record)
            return True

    async def resume(self, execution_id: str, update: JsonValue | None = None) -> bool:
        async with self._lock:
            persisted = self._records.get(execution_id)
            if persisted is None or persisted.status is not ExecutionStatus.WAITING:
                return False
            record = _copy_record(persisted)
            if record.cancel_requested_at is not None:
                record.mark_canceled()
                self._records[execution_id] = _copy_record(record)
                return False
            if update is not None:
                record.application_state[_RESUME_INPUT_KEY] = validate_json(
                    update, limit=record.max_record_bytes, label="resume update"
                )
            record.wait = None
            record.status = ExecutionStatus.QUEUED
            record.append_progress("Execution resumed", kind="resumed")
            record.to_json()
            self._records[execution_id] = _copy_record(record)
            self._enqueue_locked(execution_id)
            return True

    async def retire_incompatible(self, definition_revision: str) -> int:
        """Fail unclaimed work that can no longer run on the published revision."""
        retired = 0
        async with self._lock:
            for execution_id, persisted in list(self._records.items()):
                if (
                    persisted.terminal
                    or persisted.definition_revision == definition_revision
                    or persisted.status is ExecutionStatus.RUNNING
                    or execution_id in self._claims
                ):
                    continue
                record = _copy_record(persisted)
                record.result = None
                record.retry_at = None
                record.wait = None
                if record.cancel_requested_at is not None:
                    record.mark_canceled(force=True)
                    record.append_progress(
                        "Execution canceled during deployment replacement",
                        kind="canceled",
                    )
                else:
                    record.mark_failed(
                        ExecutionError(
                            type="DefinitionRevisionConflictError",
                            message="Execution cannot continue after its deployment definition was replaced",
                            retryable=False,
                            code="definition_revision_conflict",
                        )
                    )
                    record.append_progress(
                        "Execution retired because its deployment definition was replaced",
                        kind="definition_revision_conflict",
                    )
                record.to_json()
                self._records[execution_id] = _copy_record(record)
                self._queued_ids.discard(execution_id)
                retired += 1
            self._evict_locked()
        return retired

    async def operational_counts(self) -> dict[str, int]:
        async with self._lock:
            counts = {status.value: 0 for status in ExecutionStatus}
            for record in self._records.values():
                counts[record.status.value] += 1
            counts["queue_deliveries"] = len(self._queued_ids)
            counts["active_claims"] = len(self._claims)
            counts["delayed"] = sum(
                record.status is ExecutionStatus.QUEUED and record.retry_at is not None and record.retry_at > utc_now()
                for record in self._records.values()
            )
            return counts

    async def close(self) -> None:
        async with self._lock:
            self._closed = True

    def _enqueue_locked(self, execution_id: str) -> None:
        if execution_id not in self._queued_ids:
            self._queued.append(execution_id)
            self._queued_ids.add(execution_id)

    def _evict_locked(self) -> None:
        if self.max_records is None or len(self._records) <= self.max_records:
            return
        for execution_id in list(self._records):
            if len(self._records) <= self.max_records:
                break
            if self._records[execution_id].terminal and execution_id not in self._claims:
                self._records.pop(execution_id, None)

    async def _checkpoint(self, claim: _InMemoryExecutionClaim) -> None:
        async with self._lock:
            self._assert_owner_locked(claim)
            claim.record.touch()
            persisted = self._records[claim.record.execution_id]
            if persisted.cancel_requested_at is not None:
                claim.record.cancel_requested_at = persisted.cancel_requested_at
                claim.record.cancel_reason = persisted.cancel_reason
            claim.record.to_json()
            self._records[claim.record.execution_id] = _copy_record(claim.record)

    async def _complete(self, claim: _InMemoryExecutionClaim) -> None:
        async with self._lock:
            self._assert_owner_locked(claim)
            record = claim.record
            persisted = self._records[record.execution_id]
            if persisted.cancel_requested_at is not None:
                # A cancellation accepted between the worker's final check and
                # this fenced terminal write wins over a completion candidate.
                record.cancel_requested_at = persisted.cancel_requested_at
                record.cancel_reason = persisted.cancel_reason
                record.mark_canceled(force=True)
            if not record.terminal:
                msg = f"Execution '{record.execution_id}' cannot complete while {record.status.value}"
                raise RuntimeError(msg)
            record.to_json()
            self._records[record.execution_id] = _copy_record(record)
            self._claims.pop(record.execution_id, None)
            claim.finished = True
            self._evict_locked()

    async def _suspend(self, claim: _InMemoryExecutionClaim) -> None:
        async with self._lock:
            self._assert_owner_locked(claim)
            persisted = self._records[claim.record.execution_id]
            if persisted.cancel_requested_at is not None:
                claim.record.cancel_requested_at = persisted.cancel_requested_at
                claim.record.cancel_reason = persisted.cancel_reason
                claim.record.mark_canceled(force=True)
            if (
                claim.record.status is not ExecutionStatus.WAITING
                and claim.record.status is not ExecutionStatus.CANCELED
            ):
                msg = "Execution claims may only suspend waiting records"
                raise RuntimeError(msg)
            claim.record.touch()
            claim.record.to_json()
            self._records[claim.record.execution_id] = _copy_record(claim.record)
            self._claims.pop(claim.record.execution_id, None)
            claim.finished = True

    async def _retry(self, claim: _InMemoryExecutionClaim, error: ExecutionError, delay: float) -> None:
        async with self._lock:
            self._assert_owner_locked(claim)
            record = claim.record
            persisted = self._records[record.execution_id]
            if persisted.cancel_requested_at is not None:
                record.cancel_requested_at = persisted.cancel_requested_at
                record.cancel_reason = persisted.cancel_reason
                record.mark_canceled(force=True)
                record.to_json()
                self._records[record.execution_id] = _copy_record(record)
                self._claims.pop(record.execution_id, None)
                claim.finished = True
                return
            record.status = ExecutionStatus.QUEUED
            record.error = error
            record.retry_at = utc_now() + timedelta(seconds=delay)
            record.append_progress("Execution retry scheduled", kind="retry")
            record.to_json()
            self._records[record.execution_id] = _copy_record(record)
            self._claims.pop(record.execution_id, None)
            claim.finished = True
            self._enqueue_locked(record.execution_id)

    async def _exit(self, claim: _InMemoryExecutionClaim) -> None:
        async with self._lock:
            if self._claims.get(claim.record.execution_id) is not claim:
                return
            self._claims.pop(claim.record.execution_id, None)
            persisted = self._records.get(claim.record.execution_id)
            if persisted is not None and not persisted.terminal and persisted.status is not ExecutionStatus.WAITING:
                persisted.status = ExecutionStatus.QUEUED
                persisted.touch()
                self._records[persisted.execution_id] = _copy_record(persisted)
                self._enqueue_locked(persisted.execution_id)

    async def _cancellation_requested(self, execution_id: str) -> bool:
        async with self._lock:
            record = self._records.get(execution_id)
            return bool(record and record.cancel_requested_at is not None)

    def _assert_owner_locked(self, claim: _InMemoryExecutionClaim) -> None:
        if self._claims.get(claim.record.execution_id) is not claim:
            msg = f"Execution claim for '{claim.record.execution_id}' is no longer owned"
            raise ExecutionLeaseLostError(msg)


class _InMemoryExecutionClaim:
    def __init__(self, store: InMemoryExecutionStore, record: ExecutionRecord, token: str) -> None:
        self.store = store
        self._record = record
        self.token = token
        self.finished = False

    @property
    def record(self) -> ExecutionRecord:
        return self._record

    async def checkpoint(self) -> None:
        await self.store._checkpoint(self)

    async def cancellation_requested(self) -> bool:
        return await self.store._cancellation_requested(self.record.execution_id)

    async def complete(self) -> None:
        await self.store._complete(self)

    async def suspend(self) -> None:
        await self.store._suspend(self)

    async def retry(self, error: ExecutionError, *, delay: float) -> None:
        await self.store._retry(self, error, delay)

    async def __aenter__(self) -> _InMemoryExecutionClaim:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.finished:
            await self.store._exit(self)


class InMemoryExecutionStoreProvider:
    """Create volatile namespaced stores; opt in explicitly for development."""

    def __init__(self, **store_kwargs: Any) -> None:
        self.store_kwargs = store_kwargs
        self.stores: dict[str, InMemoryExecutionStore] = {}

    def create_execution_store(self, deployment_name: str) -> InMemoryExecutionStore:
        return self.stores.setdefault(deployment_name, InMemoryExecutionStore(**self.store_kwargs))

    async def close(self) -> None:
        for store in self.stores.values():
            await store.close()


class DurableExecutionManager:
    """Bounded worker manager shared by REST and A2A adapters."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        store: DurableExecutionStore,
        runner: RecordRunner,
        adapter: Any,
        *,
        concurrency: int = 1,
        poll_interval: float = 0.05,
        shutdown_grace_period: float = 5.0,
        max_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        if concurrency < 1:
            msg = "durable execution concurrency must be at least one"
            raise ValueError(msg)
        self.name = name
        self.store = store
        self.runner = runner
        self.adapter = adapter
        self.concurrency = concurrency
        self.poll_interval = poll_interval
        self.shutdown_grace_period = shutdown_grace_period
        self.max_attempts = max(1, max_attempts)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self.retry_max_delay = max(self.retry_base_delay, retry_max_delay)
        self._workers: list[asyncio.Task[None]] = []
        self._draining_workers: set[asyncio.Task[None]] = set()
        self._prepared = False
        self._started = False
        self._accepting_claims = False
        self._worker_generation = 0
        self._store_error_count = 0
        self._last_successful_claim_at: datetime | None = None
        self._active_claims = 0
        self._metrics = {
            "attempts_started": 0,
            "completed": 0,
            "failed": 0,
            "canceled": 0,
            "suspended": 0,
            "retries_scheduled": 0,
            "retries_exhausted": 0,
            "lease_losses": 0,
            "worker_restarts": 0,
            "record_size_failures": 0,
        }

    async def start(self) -> None:
        if self._started:
            self._accepting_claims = True
            return
        await self.prepare()
        self.activate()

    async def prepare(self) -> None:
        """Initialize storage while keeping workers and submissions disabled."""
        if self._prepared:
            return
        await self.store.initialize()
        self._prepared = True

    def activate(self) -> None:
        """Start workers for an initialized deployment without an await gap."""
        if self._started:
            self._accepting_claims = True
            return
        if not self._prepared:
            msg = "durable execution manager must be prepared before activation"
            raise RuntimeError(msg)
        self._started = True
        self._accepting_claims = True
        self._worker_generation += 1
        generation = self._worker_generation
        identity = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._workers = [self._start_worker(identity, slot, generation) for slot in range(self.concurrency)]

    def deactivate(self) -> None:
        self._accepting_claims = False
        self._worker_generation += 1

    @property
    def started(self) -> bool:
        return self._started

    @property
    def accepting(self) -> bool:
        return self._started and self._accepting_claims

    @property
    def health(self) -> dict[str, JsonValue]:
        """Return a payload-safe worker projection for readiness and diagnostics."""
        running = sum(not worker.done() for worker in self._workers)
        return {
            "healthy": not self._prepared or (self._started and self._accepting_claims and running == self.concurrency),
            "configured_slots": self.concurrency,
            "running_slots": running,
            "draining_slots": sum(not worker.done() for worker in self._draining_workers),
            "store_error_count": self._store_error_count,
            "last_successful_claim_at": (
                self._last_successful_claim_at.isoformat() if self._last_successful_claim_at else None
            ),
            "accepting": self.accepting,
            "active_claims": self._active_claims,
            "metrics": cast(JsonValue, dict(self._metrics)),
        }

    async def health_snapshot(self) -> dict[str, JsonValue]:
        """Add storage-level queue/state counts to the local readiness view."""
        health = self.health
        counts = getattr(self.store, "operational_counts", None)
        if not callable(counts):
            return health
        try:
            health["counts"] = cast(JsonValue, await counts())
        except Exception as error:
            health["healthy"] = False
            health["operational_error"] = type(error).__name__
        return health

    @property
    def draining(self) -> bool:
        return any(not worker.done() for worker in self._draining_workers)

    async def wait_drained(self) -> None:
        """Wait for detached application work before closing shared storage."""
        pending = [worker for worker in self._draining_workers if not worker.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self) -> None:
        self._accepting_claims = False
        if not self._workers:
            self._started = False
            return
        done, pending = await asyncio.wait(self._workers, timeout=self.shutdown_grace_period)
        if pending:
            self._draining_workers.update(pending)
            for worker in pending:
                worker.add_done_callback(self._draining_workers.discard)
            log.warning(
                "{} | {} durable worker slot(s) exceeded the {:.2f}s shutdown grace period; "
                "claims remain fenced and heartbeating until application work exits",
                self.name,
                len(pending),
                self.shutdown_grace_period,
            )
        self._log_worker_failures(done)
        self._workers = []
        self._started = False

    async def submit(self, record: ExecutionRecord) -> bool:
        return await self.store.submit(record)

    def _start_worker(self, identity: str, slot: int, generation: int) -> asyncio.Task[None]:
        worker = asyncio.create_task(
            self._worker(f"{self.name}:{identity}:{slot}", generation),
            name=f"durable:{self.name}:{slot}",
        )
        worker.add_done_callback(lambda completed: self._worker_done(identity, slot, generation, completed))
        return worker

    def _worker_done(
        self,
        identity: str,
        slot: int,
        generation: int,
        worker: asyncio.Task[None],
    ) -> None:
        if self._accepting_claims and generation == self._worker_generation and not worker.cancelled():
            error = worker.exception()
            if error is not None:
                self._metrics["worker_restarts"] += 1
                log.opt(exception=error).error(
                    "{} | durable worker slot {} stopped unexpectedly; restarting",
                    self.name,
                    slot,
                )
                replacement = self._start_worker(identity, slot, generation)
                try:
                    index = self._workers.index(worker)
                except ValueError:
                    replacement.cancel()
                else:
                    self._workers[index] = replacement

    def _log_worker_failures(self, workers: set[asyncio.Task[None]]) -> None:
        for worker in workers:
            if worker.cancelled():
                continue
            error = worker.exception()
            if error is not None:
                log.opt(exception=error).warning(
                    "{} | durable worker '{}' ended with an error during shutdown",
                    self.name,
                    worker.get_name(),
                )

    async def _worker(self, worker_name: str, generation: int) -> None:
        consecutive_store_errors = 0
        while self._accepting_claims and generation == self._worker_generation:
            try:
                claim = await self.store.claim_next(worker_name)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                consecutive_store_errors += 1
                await self._backoff_store_error(error, consecutive_store_errors, operation="claim")
                continue
            if claim is None:
                await asyncio.sleep(self.poll_interval)
                continue
            self._last_successful_claim_at = utc_now()
            if not self._accepting_claims or generation != self._worker_generation:
                try:
                    async with claim:
                        pass
                except Exception as error:
                    consecutive_store_errors += 1
                    await self._backoff_store_error(error, consecutive_store_errors, operation="abandon")
                return
            try:
                self._active_claims += 1
                self._metrics["attempts_started"] += 1
                try:
                    await self._process_claim(claim)
                finally:
                    self._active_claims -= 1
                consecutive_store_errors = 0
            except ExecutionLeaseLostError:
                consecutive_store_errors = 0
                self._metrics["lease_losses"] += 1
                log.warning("{} | lost durable execution claim {}", self.name, claim.record.execution_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                consecutive_store_errors += 1
                await self._backoff_store_error(error, consecutive_store_errors, operation="transition")

    async def _process_claim(  # noqa: C901, PLR0912, PLR0915 - explicit durable transition table
        self,
        claim: ExecutionClaim,
    ) -> None:
        async with claim:
            context = DurableContext(claim, self.adapter)
            try:
                if await claim.cancellation_requested():
                    claim.record.mark_canceled()
                    await claim.complete()
                    self._record_terminal_metric(claim.record)
                    return
                with execution_context_scope(context):
                    result = await self.runner(context)
                if await claim.cancellation_requested():
                    claim.record.mark_canceled()
                else:
                    claim.record.result = validate_json(result, limit=claim.record.max_record_bytes, label="result")
                    claim.record.error = None
                    claim.record.status = ExecutionStatus.COMPLETED
                    claim.record.wait = None
                    claim.record.retry_at = None
                claim.record.touch()
                await claim.complete()
                self._record_terminal_metric(claim.record)
            except ExecutionSuspendedError:
                # ``DurableContext.suspend`` already persisted and released this claim.
                if claim.record.terminal:
                    self._record_terminal_metric(claim.record)
                else:
                    self._metrics["suspended"] += 1
                return
            except ExecutionCanceledError:
                claim.record.mark_canceled()
                await claim.complete()
                self._record_terminal_metric(claim.record)
            except RetryableExecutionError as error:
                if claim.record.attempt >= self.max_attempts:
                    exhausted = ExecutionError(
                        type="RetryExhausted",
                        message=f"Execution exhausted its {self.max_attempts} permitted attempts",
                        retryable=False,
                        code="retry_exhausted",
                    )
                    claim.record.mark_failed(exhausted)
                    claim.record.append_progress("Execution retry limit reached", kind="retry_exhausted")
                    await claim.complete()
                    self._metrics["retries_exhausted"] += 1
                    self._record_terminal_metric(claim.record)
                    return
                delay = self._retry_delay(claim.record.attempt, error.delay)
                await claim.retry(ExecutionError.from_exception(error, retryable=True), delay=delay)
                if claim.record.terminal:
                    self._record_terminal_metric(claim.record)
                else:
                    self._metrics["retries_scheduled"] += 1
            except ExecutionRecordSizeError:
                await self._fail_oversized_record(claim)
            except ExecutionLeaseLostError:
                raise
            except ExecutionStoreError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as error:
                claim.record.mark_failed(error)
                await claim.complete()
                self._record_terminal_metric(claim.record)

    async def _fail_oversized_record(self, claim: ExecutionClaim) -> None:
        """Persist a small terminal record after application state exceeded its bound."""
        record = claim.record
        record.validated_input = {}
        record.checkpoint = None
        record.application_state = {}
        record.wait = None
        record.progress = []
        record.result = None
        record.error = None
        record.last_retry_error = None
        record.retry_at = None
        record.mark_failed(
            ExecutionError(
                type="ExecutionRecordTooLarge",
                message=f"Execution exceeded its {record.max_record_bytes}-byte durable record limit",
                retryable=False,
                code="record_too_large",
            )
        )
        await claim.complete()
        self._metrics["record_size_failures"] += 1
        self._record_terminal_metric(record)

    def _record_terminal_metric(self, record: ExecutionRecord) -> None:
        if record.status is ExecutionStatus.COMPLETED:
            self._metrics["completed"] += 1
        elif record.status is ExecutionStatus.FAILED:
            self._metrics["failed"] += 1
        elif record.status is ExecutionStatus.CANCELED:
            self._metrics["canceled"] += 1

    def _retry_delay(self, attempt: int, requested_delay: float) -> float:
        exponent = min(max(0, attempt - 1), 30)
        delay = requested_delay if requested_delay > 0 else self.retry_base_delay * (2**exponent)
        return min(max(0.0, delay), self.retry_max_delay)

    async def _backoff_store_error(self, error: BaseException, failures: int, *, operation: str) -> None:
        self._store_error_count += 1
        exponent = min(max(0, failures - 1), 10)
        ceiling = min(max(self.poll_interval, 0.01) * (2**exponent), 5.0)
        delay = random.uniform(ceiling / 2, ceiling)  # noqa: S311 - jitter is not security-sensitive
        log.opt(exception=error).warning(
            "{} | durable worker store {} failed; retrying in {:.2f}s: {}",
            self.name,
            operation,
            delay,
            error,
        )
        await asyncio.sleep(delay)


def _copy_record(record: ExecutionRecord | None) -> ExecutionRecord:
    if record is None:
        msg = "execution record is required"
        raise TypeError(msg)
    return copy.deepcopy(record)


__all__ = [
    "DEFAULT_MAX_PROGRESS_EVENTS",
    "DEFAULT_MAX_RECORD_BYTES",
    "RECORD_SCHEMA_VERSION",
    "DurableContext",
    "DurableExecutionManager",
    "DurableExecutionStore",
    "DurableExecutionStoreProvider",
    "ExecutionCanceledError",
    "ExecutionCheckpoint",
    "ExecutionError",
    "ExecutionKind",
    "ExecutionLeaseLostError",
    "ExecutionProgressEvent",
    "ExecutionRecord",
    "ExecutionRecordSizeError",
    "ExecutionStatus",
    "ExecutionStoreError",
    "ExecutionSuspendedError",
    "InMemoryExecutionStore",
    "InMemoryExecutionStoreProvider",
    "JsonValue",
    "RetryableExecutionError",
    "get_current_durable_context",
    "json_safe",
    "utc_now",
    "validate_json",
]
