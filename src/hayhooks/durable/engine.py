"""
Storage-neutral durable execution state machine.

The reducer is the only place that decides an execution lifecycle.  Storage
only persists its effects and derives its indexes from the old and new control.
"""
# ruff: noqa: EM101, EM102, PLR0911, PLR0913

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum

RUN_ID_PATTERN = r"[A-Za-z0-9_-]{1,128}"
MAX_CONTROL_SCALAR_BYTES = 4_096
MAX_CANCELLATION_REASON_LENGTH = 2_000
_RUN_ID_RE = re.compile(f"{RUN_ID_PATTERN}\\Z")


def validate_run_id(run_id: str) -> None:
    """Reject execution IDs that cannot be embedded safely in backend keys."""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id has an invalid key-safe format")


def normalize_cancellation_reason(reason: str | None) -> str | None:
    """Bound cancellation text by both characters and persisted UTF-8 bytes."""
    if not reason:
        return None
    encoded = str(reason)[:MAX_CANCELLATION_REASON_LENGTH].encode()[:MAX_CONTROL_SCALAR_BYTES]
    return encoded.decode(errors="ignore")


class ExecutionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELED}


class PayloadKind(str, Enum):
    INPUT = "input"
    CHECKPOINT = "checkpoint"
    RESULT = "result"
    ERROR = "error"
    WAIT = "wait"


class ExecutionNotFoundError(RuntimeError):
    pass


class ExecutionLeaseLostError(RuntimeError):
    pass


class InvalidExecutionTransitionError(RuntimeError):
    pass


class ExecutionPayloadSizeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionControl:
    """
    The compact, authoritative execution state.

    Times come from Redis ``TIME`` (or the reference-store equivalent), so a
    worker never decides leases from its own wall clock.
    """

    run_id: str
    idempotency_digest: str
    idempotency_binding_digest: str
    deployment: str
    definition_revision: str
    owner_id: str | None
    kind: str
    status: ExecutionStatus = ExecutionStatus.QUEUED
    version: int = 1
    fence: int = 0
    run_attempt: int = 0
    application_retry_count: int = 0
    available_at_ms: int | None = None
    lease_owner: str | None = None
    lease_expires_at_ms: int | None = None
    cancel_requested_at_ms: int | None = None
    cancel_reason: str | None = None
    progress_sequence: int = 0
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "idempotency_digest",
            "idempotency_binding_digest",
            "deployment",
            "definition_revision",
            "kind",
        ):
            value = getattr(self, name)
            if not value or len(value.encode()) > MAX_CONTROL_SCALAR_BYTES:
                raise ValueError(f"{name} must be non-empty and at most {MAX_CONTROL_SCALAR_BYTES} bytes")
        for name in ("owner_id", "lease_owner", "cancel_reason"):
            value = getattr(self, name)
            if value is not None and len(value.encode()) > MAX_CONTROL_SCALAR_BYTES:
                raise ValueError(f"{name} must be at most {MAX_CONTROL_SCALAR_BYTES} bytes")
        validate_run_id(self.run_id)
        for name in (
            "version",
            "fence",
            "run_attempt",
            "application_retry_count",
            "progress_sequence",
            "created_at_ms",
            "updated_at_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.status is ExecutionStatus.RUNNING:
            if not self.lease_owner or self.lease_expires_at_ms is None:
                raise ValueError("running controls require lease_owner and lease_expires_at_ms")
        elif self.lease_owner is not None or self.lease_expires_at_ms is not None:
            raise ValueError("only running controls may hold a lease")

    @property
    def terminal(self) -> bool:
        return self.status.terminal


@dataclass(frozen=True, slots=True)
class PayloadWrite:
    kind: PayloadKind
    data: bytes


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    sequence: int
    data: bytes


@dataclass(frozen=True, slots=True)
class LeaseIndexUpdate:
    deadline_ms: int | None
    fence: int


@dataclass(frozen=True, slots=True)
class TransitionPlan:
    next_control: ExecutionControl
    payload_writes: tuple[PayloadWrite, ...] = ()
    payload_deletes: tuple[PayloadKind, ...] = ()
    progress_events: tuple[ProgressEvent, ...] = ()
    lease_index_update: LeaseIndexUpdate | None = None


@dataclass(frozen=True, slots=True)
class Claim:
    worker_id: str
    now_ms: int
    lease_duration_ms: int
    max_run_attempts: int
    worker_revision: str


@dataclass(frozen=True, slots=True)
class ReleaseClaim:
    fence: int
    worker_id: str
    now_ms: int = 0
    lease_commit_safety_ms: int = 0


@dataclass(frozen=True, slots=True)
class Heartbeat:
    fence: int
    worker_id: str
    now_ms: int
    lease_duration_ms: int
    lease_commit_safety_ms: int = 0


@dataclass(frozen=True, slots=True)
class Checkpoint:
    fence: int
    worker_id: str
    now_ms: int
    lease_duration_ms: int
    payload: bytes
    progress_events: tuple[bytes, ...] = ()
    lease_commit_safety_ms: int = 0


@dataclass(frozen=True, slots=True)
class RequestCancellation:
    now_ms: int
    reason: str | None = None
    progress_events: tuple[bytes, ...] = ()


@dataclass(frozen=True, slots=True)
class ScheduleRetry:
    fence: int
    worker_id: str
    now_ms: int
    delay_ms: int
    max_application_retries: int
    error: bytes = b""
    lease_commit_safety_ms: int = 0


@dataclass(frozen=True, slots=True)
class Suspend:
    fence: int
    worker_id: str
    now_ms: int
    checkpoint: bytes
    wait: bytes
    progress_events: tuple[bytes, ...] = ()
    lease_commit_safety_ms: int = 0


@dataclass(frozen=True, slots=True)
class Resume:
    now_ms: int
    worker_revision: str
    checkpoint: bytes | None = None
    progress_events: tuple[bytes, ...] = ()
    expected_version: int | None = None


@dataclass(frozen=True, slots=True)
class Complete:
    fence: int
    worker_id: str
    now_ms: int
    result: bytes
    progress_events: tuple[bytes, ...] = ()
    lease_commit_safety_ms: int = 0


@dataclass(frozen=True, slots=True)
class Fail:
    fence: int
    worker_id: str
    now_ms: int
    error: bytes
    progress_events: tuple[bytes, ...] = ()
    lease_commit_safety_ms: int = 0


@dataclass(frozen=True, slots=True)
class RecoverExpiredLease:
    now_ms: int
    indexed_fence: int
    indexed_deadline_ms: int
    max_run_attempts: int
    worker_revision: str


ExecutionCommand = (
    Claim
    | ReleaseClaim
    | Heartbeat
    | Checkpoint
    | RequestCancellation
    | ScheduleRetry
    | Suspend
    | Resume
    | Complete
    | Fail
    | RecoverExpiredLease
)


def initial_control(
    *,
    run_id: str,
    idempotency_digest: str,
    idempotency_binding_digest: str,
    deployment: str,
    definition_revision: str,
    owner_id: str | None,
    kind: str,
    now_ms: int,
) -> ExecutionControl:
    """Create the payload-free, version-one control record for a new queued execution."""
    return ExecutionControl(
        run_id=run_id,
        idempotency_digest=idempotency_digest,
        idempotency_binding_digest=idempotency_binding_digest,
        deployment=deployment,
        definition_revision=definition_revision,
        owner_id=owner_id,
        kind=kind,
        created_at_ms=now_ms,
        updated_at_ms=now_ms,
    )


def submission_plan(control: ExecutionControl, input_payload: bytes) -> TransitionPlan:
    """Validate a new control and return the initial input-persistence plan."""
    if control.status is not ExecutionStatus.QUEUED or control.version != 1:
        raise InvalidExecutionTransitionError("only a new queued control can be submitted")
    return TransitionPlan(control, payload_writes=(PayloadWrite(PayloadKind.INPUT, input_payload),))


def decide(control: ExecutionControl, command: ExecutionCommand) -> TransitionPlan:  # noqa: C901
    """
    Reduce one command into the next control and its atomic persistence effects.

    The reducer performs no I/O and never mutates ``control``. Invalid lifecycle
    or lease transitions raise ``InvalidExecutionTransitionError`` or
    ``ExecutionLeaseLostError`` before any effects can be persisted.
    """
    if isinstance(command, Claim):
        return _claim(control, command)
    if isinstance(command, ReleaseClaim):
        return _release_claim(control, command)
    if isinstance(command, Heartbeat):
        _owned(control, command.fence, command.worker_id, command.now_ms, command.lease_commit_safety_ms)
        return TransitionPlan(
            replace(control, lease_expires_at_ms=command.now_ms + command.lease_duration_ms),
            lease_index_update=LeaseIndexUpdate(command.now_ms + command.lease_duration_ms, control.fence),
        )
    if isinstance(command, Checkpoint):
        _owned(control, command.fence, command.worker_id, command.now_ms, command.lease_commit_safety_ms)
        next_control = _business(
            control,
            command.now_ms,
            progress_sequence=control.progress_sequence + len(command.progress_events),
            lease_expires_at_ms=command.now_ms + command.lease_duration_ms,
        )
        return TransitionPlan(
            next_control,
            payload_writes=(PayloadWrite(PayloadKind.CHECKPOINT, command.payload),),
            progress_events=_progress_events(control.progress_sequence, command.progress_events),
            lease_index_update=LeaseIndexUpdate(next_control.lease_expires_at_ms, control.fence),
        )
    if isinstance(command, RequestCancellation):
        return _cancel(control, command)
    if isinstance(command, ScheduleRetry):
        return _retry(control, command)
    if isinstance(command, Suspend):
        return _suspend(control, command)
    if isinstance(command, Resume):
        return _resume(control, command)
    if isinstance(command, Complete):
        _owned(control, command.fence, command.worker_id, command.now_ms, command.lease_commit_safety_ms)
        return _terminal_or_canceled(
            control,
            command.now_ms,
            ExecutionStatus.COMPLETED,
            PayloadKind.RESULT,
            command.result,
            command.progress_events,
        )
    if isinstance(command, Fail):
        _owned(control, command.fence, command.worker_id, command.now_ms, command.lease_commit_safety_ms)
        return _terminal_or_canceled(
            control, command.now_ms, ExecutionStatus.FAILED, PayloadKind.ERROR, command.error, command.progress_events
        )
    if isinstance(command, RecoverExpiredLease):
        return _recover(control, command)
    raise TypeError(f"unsupported execution command {type(command).__name__}")


def _claim(control: ExecutionControl, command: Claim) -> TransitionPlan:
    if control.status is not ExecutionStatus.QUEUED:
        raise InvalidExecutionTransitionError("execution is not queued")
    if control.available_at_ms is not None and control.available_at_ms > command.now_ms:
        raise InvalidExecutionTransitionError("queued execution is not due")
    if control.definition_revision != command.worker_revision:
        return _terminal(
            control, command.now_ms, ExecutionStatus.FAILED, PayloadKind.ERROR, b"definition revision is incompatible"
        )
    if control.cancel_requested_at_ms is not None:
        return _terminal(control, command.now_ms, ExecutionStatus.CANCELED, None, None)
    if control.run_attempt >= command.max_run_attempts:
        return _terminal(control, command.now_ms, ExecutionStatus.FAILED, PayloadKind.ERROR, b"run attempts exhausted")
    deadline = command.now_ms + command.lease_duration_ms
    next_control = _business(
        control,
        command.now_ms,
        status=ExecutionStatus.RUNNING,
        fence=control.fence + 1,
        run_attempt=control.run_attempt + 1,
        available_at_ms=None,
        lease_owner=command.worker_id,
        lease_expires_at_ms=deadline,
    )
    return TransitionPlan(next_control, lease_index_update=LeaseIndexUpdate(deadline, next_control.fence))


def _release_claim(control: ExecutionControl, command: ReleaseClaim) -> TransitionPlan:
    _owned(control, command.fence, command.worker_id, command.now_ms, command.lease_commit_safety_ms)
    next_control = _business(
        control,
        command.now_ms,
        status=ExecutionStatus.QUEUED,
        run_attempt=control.run_attempt - 1,
        lease_owner=None,
        lease_expires_at_ms=None,
    )
    return TransitionPlan(next_control, lease_index_update=LeaseIndexUpdate(None, control.fence))


def _cancel(control: ExecutionControl, command: RequestCancellation) -> TransitionPlan:
    if control.terminal or control.cancel_requested_at_ms is not None:
        return TransitionPlan(control)
    if control.status is ExecutionStatus.RUNNING:
        return TransitionPlan(
            _business(
                control,
                command.now_ms,
                cancel_requested_at_ms=command.now_ms,
                cancel_reason=normalize_cancellation_reason(command.reason),
                progress_sequence=control.progress_sequence + len(command.progress_events),
            ),
            progress_events=_progress_events(control.progress_sequence, command.progress_events),
        )
    return _terminal(
        _business(
            control,
            command.now_ms,
            cancel_requested_at_ms=command.now_ms,
            cancel_reason=normalize_cancellation_reason(command.reason),
            progress_sequence=control.progress_sequence + len(command.progress_events),
        ),
        command.now_ms,
        ExecutionStatus.CANCELED,
        None,
        None,
        increment_version=False,
        progress_events=_progress_events(control.progress_sequence, command.progress_events),
    )


def _retry(control: ExecutionControl, command: ScheduleRetry) -> TransitionPlan:
    _owned(control, command.fence, command.worker_id, command.now_ms, command.lease_commit_safety_ms)
    if control.cancel_requested_at_ms is not None:
        return _terminal(control, command.now_ms, ExecutionStatus.CANCELED, None, None)
    if control.application_retry_count >= command.max_application_retries:
        return _terminal(
            control,
            command.now_ms,
            ExecutionStatus.FAILED,
            PayloadKind.ERROR,
            command.error or b"application retries exhausted",
        )
    due = command.now_ms + max(0, command.delay_ms)
    next_control = _business(
        control,
        command.now_ms,
        status=ExecutionStatus.QUEUED,
        application_retry_count=control.application_retry_count + 1,
        available_at_ms=due,
        lease_owner=None,
        lease_expires_at_ms=None,
    )
    return TransitionPlan(
        next_control,
        payload_writes=((PayloadWrite(PayloadKind.ERROR, command.error),) if command.error else ()),
        payload_deletes=((PayloadKind.ERROR,) if not command.error else ()),
        lease_index_update=LeaseIndexUpdate(None, control.fence),
    )


def _suspend(control: ExecutionControl, command: Suspend) -> TransitionPlan:
    _owned(control, command.fence, command.worker_id, command.now_ms, command.lease_commit_safety_ms)
    if control.cancel_requested_at_ms is not None:
        return _terminal(control, command.now_ms, ExecutionStatus.CANCELED, None, None)
    next_control = _business(
        control,
        command.now_ms,
        status=ExecutionStatus.WAITING,
        progress_sequence=control.progress_sequence + len(command.progress_events),
        lease_owner=None,
        lease_expires_at_ms=None,
    )
    return TransitionPlan(
        next_control,
        payload_writes=(
            PayloadWrite(PayloadKind.CHECKPOINT, command.checkpoint),
            PayloadWrite(PayloadKind.WAIT, command.wait),
        ),
        progress_events=_progress_events(control.progress_sequence, command.progress_events),
        lease_index_update=LeaseIndexUpdate(None, control.fence),
    )


def _resume(control: ExecutionControl, command: Resume) -> TransitionPlan:
    if command.expected_version is not None and control.version != command.expected_version:
        raise InvalidExecutionTransitionError("execution changed before it could resume")
    if control.status is not ExecutionStatus.WAITING:
        raise InvalidExecutionTransitionError("only waiting executions can resume")
    if control.definition_revision != command.worker_revision:
        raise InvalidExecutionTransitionError("definition revision is incompatible")
    if control.cancel_requested_at_ms is not None:
        return _terminal(control, command.now_ms, ExecutionStatus.CANCELED, None, None)
    writes = (PayloadWrite(PayloadKind.CHECKPOINT, command.checkpoint),) if command.checkpoint is not None else ()
    next_control = _business(
        control,
        command.now_ms,
        status=ExecutionStatus.QUEUED,
        progress_sequence=control.progress_sequence + len(command.progress_events),
    )
    return TransitionPlan(
        next_control,
        payload_writes=writes,
        payload_deletes=(PayloadKind.WAIT,),
        progress_events=_progress_events(control.progress_sequence, command.progress_events),
    )


def _recover(control: ExecutionControl, command: RecoverExpiredLease) -> TransitionPlan:
    if control.status is not ExecutionStatus.RUNNING or control.fence != command.indexed_fence:
        return TransitionPlan(control, lease_index_update=LeaseIndexUpdate(None, command.indexed_fence))
    assert control.lease_expires_at_ms is not None
    if control.lease_expires_at_ms != command.indexed_deadline_ms:
        return TransitionPlan(control, lease_index_update=LeaseIndexUpdate(control.lease_expires_at_ms, control.fence))
    if control.lease_expires_at_ms > command.now_ms:
        return TransitionPlan(control)
    if control.cancel_requested_at_ms is not None:
        return _terminal(control, command.now_ms, ExecutionStatus.CANCELED, None, None)
    if control.definition_revision != command.worker_revision:
        return _terminal(
            control, command.now_ms, ExecutionStatus.FAILED, PayloadKind.ERROR, b"definition revision is incompatible"
        )
    if control.run_attempt >= command.max_run_attempts:
        return _terminal(control, command.now_ms, ExecutionStatus.FAILED, PayloadKind.ERROR, b"run attempts exhausted")
    next_control = _business(
        control, command.now_ms, status=ExecutionStatus.QUEUED, lease_owner=None, lease_expires_at_ms=None
    )
    return TransitionPlan(next_control, lease_index_update=LeaseIndexUpdate(None, control.fence))


def _terminal_or_canceled(
    control: ExecutionControl,
    now_ms: int,
    status: ExecutionStatus,
    payload_kind: PayloadKind,
    payload: bytes,
    progress_values: tuple[bytes, ...] = (),
) -> TransitionPlan:
    progress_events = _progress_events(control.progress_sequence, progress_values)
    if control.cancel_requested_at_ms is not None:
        return _terminal(control, now_ms, ExecutionStatus.CANCELED, None, None, progress_events=progress_events)
    return _terminal(control, now_ms, status, payload_kind, payload, progress_events=progress_events)


def _terminal(
    control: ExecutionControl,
    now_ms: int,
    status: ExecutionStatus,
    payload_kind: PayloadKind | None,
    payload: bytes | None,
    *,
    increment_version: bool = True,
    progress_events: tuple[ProgressEvent, ...] = (),
) -> TransitionPlan:
    if control.terminal:
        raise InvalidExecutionTransitionError("terminal execution cannot transition")
    kwargs: dict[str, object] = {
        "status": status,
        "available_at_ms": None,
        "lease_owner": None,
        "lease_expires_at_ms": None,
    }
    if progress_events:
        kwargs["progress_sequence"] = progress_events[-1].sequence
    next_control = (
        _business(control, now_ms, **kwargs) if increment_version else replace(control, updated_at_ms=now_ms, **kwargs)
    )
    writes = (PayloadWrite(payload_kind, payload or b""),) if payload_kind is not None else ()
    deletes: tuple[PayloadKind, ...] = ()
    if payload_kind is PayloadKind.RESULT:
        deletes = (PayloadKind.ERROR,)
    elif payload_kind is PayloadKind.ERROR:
        deletes = (PayloadKind.RESULT,)
    elif status is ExecutionStatus.CANCELED:
        deletes = (PayloadKind.RESULT, PayloadKind.ERROR)
    return TransitionPlan(
        next_control,
        payload_writes=writes,
        payload_deletes=(*deletes, PayloadKind.WAIT),
        progress_events=progress_events,
        lease_index_update=LeaseIndexUpdate(None, control.fence),
    )


def _owned(control: ExecutionControl, fence: int, worker_id: str, now_ms: int, safety_margin_ms: int) -> None:
    if (
        control.status is not ExecutionStatus.RUNNING
        or control.fence != fence
        or control.lease_owner != worker_id
        or control.lease_expires_at_ms is None
        or safety_margin_ms < 0
        or now_ms >= control.lease_expires_at_ms - safety_margin_ms
    ):
        raise ExecutionLeaseLostError("execution is no longer owned by this worker fence")


def _business(control: ExecutionControl, now_ms: int, **changes: object) -> ExecutionControl:
    return replace(control, version=control.version + 1, updated_at_ms=now_ms, **changes)


def _progress_events(sequence: int, values: tuple[bytes, ...]) -> tuple[ProgressEvent, ...]:
    return tuple(ProgressEvent(sequence + index, value) for index, value in enumerate(values, start=1))
