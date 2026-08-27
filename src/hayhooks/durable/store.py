"""Durable persistence contract and in-memory reference implementation."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from loguru import logger as log

from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionCommand,
    ExecutionControl,
    ExecutionLeaseLostError,
    ExecutionNotFoundError,
    ExecutionPayloadSizeError,
    ExecutionStatus,
    Fail,
    Heartbeat,
    InvalidExecutionTransitionError,
    PayloadKind,
    ProgressEvent,
    RecoverExpiredLease,
    ReleaseClaim,
    ScheduleRetry,
    Suspend,
    TransitionPlan,
    decide,
    require_owned,
    submission_plan,
)

CHUNK_CURSOR_START = "0-0"
MAINTENANCE_BATCH_SIZE = 100
MAX_CHUNK_READ_BYTES = 4_000_000
MAX_CHUNK_READ_COUNT = 1_000
_CHUNK_CURSOR = re.compile(r"^\d{1,20}-\d{1,20}$")
_MAX_CURSOR_PART = 2**64 - 1
_LEASE_COMMANDS = (ReleaseClaim, Heartbeat, Checkpoint, ScheduleRetry, Suspend, Complete, Fail)


class ExecutionStoreError(RuntimeError):
    """A durable store operation failed."""


class ExecutionStoreCorruptionError(ExecutionStoreError):
    """Persisted state cannot be decoded without violating durable invariants."""


class ExecutionContentionError(ExecutionStoreError):
    """A bounded optimistic transaction could not obtain a stable snapshot."""


class ExecutionAdmissionError(RuntimeError):
    """The configured nonterminal execution limit has been reached."""


class ExecutionIdempotencyConflictError(RuntimeError):
    """An idempotency key was reused for different work."""


class ChunkCursorExpiredError(RuntimeError):
    """A requested chunk cursor is no longer retained."""


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """Limits shared by every durable store implementation."""

    lease_commit_safety_ms: int = 1_500
    terminal_ttl_seconds: int = 604_800
    max_nonterminal_executions: int = 1_000
    max_payload_bytes: int = 1_000_000
    max_progress_events: int = 100
    max_progress_event_bytes: int = 8_192
    max_stream_chunks: int = 100
    max_stream_chunk_bytes: int = 64_000

    def __post_init__(self) -> None:
        if min(self.lease_commit_safety_ms, self.max_nonterminal_executions, self.max_stream_chunks) < 0:
            raise ValueError("durable store limits cannot be negative")
        for name in (
            "terminal_ttl_seconds",
            "max_payload_bytes",
            "max_progress_events",
            "max_progress_event_bytes",
            "max_stream_chunk_bytes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    created: bool
    control: ExecutionControl


@dataclass(frozen=True, slots=True)
class StoredExecution:
    """One stable control and payload snapshot."""

    control: ExecutionControl
    payloads: Mapping[PayloadKind, bytes]
    progress: tuple[ProgressEvent, ...]


@dataclass(frozen=True, slots=True)
class StreamChunk:
    cursor: str
    attempt: int
    data: bytes


class ExecutionStore(Protocol):
    """Atomic persistence required by the durable runtime."""

    config: StoreConfig
    deployment: str

    async def initialize(self) -> None: ...

    async def submit(self, control: ExecutionControl, input_payload: bytes) -> SubmissionResult: ...

    async def read(self, run_id: str) -> StoredExecution | None: ...

    async def transition(self, run_id: str, command: ExecutionCommand) -> TransitionPlan: ...

    async def claim(self, command: Claim) -> TransitionPlan | None: ...

    async def maintain(
        self,
        *,
        max_run_attempts: int,
        attempts_error: bytes,
    ) -> None: ...

    async def append_chunk(self, run_id: str, attempt: int, fence: int, worker_id: str, data: bytes) -> None: ...

    async def read_chunks(self, run_id: str, after: str) -> tuple[StreamChunk, ...]: ...

    async def operational_counts(self) -> dict[str, int]: ...


class MemoryExecutionStore:
    """Single-process reference store for development and contract tests."""

    def __init__(
        self,
        deployment: str,
        *,
        config: StoreConfig | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if not deployment:
            raise ValueError("deployment cannot be empty")
        self.deployment = deployment
        self.config = config or StoreConfig()
        self._clock = clock or (lambda: time.time_ns() // 1_000_000)
        self._controls: dict[str, ExecutionControl] = {}
        self._payloads: dict[str, dict[PayloadKind, bytes]] = {}
        self._progress: dict[str, list[ProgressEvent]] = {}
        self._chunks: dict[str, deque[StreamChunk]] = {}
        self._chunk_sequence = 0
        self._runnable: dict[str, int] = {}
        self._lease_expiry: dict[tuple[str, int], int] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._terminal_cleanup: dict[str, tuple[int, str, tuple[str, str]]] = {}
        self._nonterminal = 0

    async def initialize(self) -> None:
        return None

    async def submit(self, control: ExecutionControl, input_payload: bytes) -> SubmissionResult:
        if control.deployment != self.deployment:
            raise ValueError("control deployment does not match this store")
        validate_payload_size("input", input_payload, self.config.max_payload_bytes)
        self._cleanup_terminal(self._clock())
        binding = (control.run_id, control.idempotency_binding_digest)
        if existing := self._idempotency.get(control.idempotency_digest):
            if existing[1] != binding[1]:
                raise ExecutionIdempotencyConflictError("idempotency key is bound to different work")
            existing_control = self._controls.get(existing[0])
            if existing_control is None:
                raise ExecutionStoreError("idempotency binding points to a missing execution")
            return SubmissionResult(created=False, control=existing_control)
        if control.run_id in self._controls:
            raise ExecutionIdempotencyConflictError("run ID is bound to a different idempotency key")
        if self.config.max_nonterminal_executions and self._nonterminal >= self.config.max_nonterminal_executions:
            raise ExecutionAdmissionError("nonterminal execution limit reached")

        now_ms = self._clock()
        control = replace(control, created_at_ms=now_ms, updated_at_ms=now_ms)
        plan = submission_plan(control, input_payload)
        self._idempotency[control.idempotency_digest] = binding
        self._apply(control, plan, new_submission=True)
        log.bind(run_id=control.run_id, deployment=control.deployment).debug("Submitted durable execution")
        return SubmissionResult(created=True, control=control)

    async def read(self, run_id: str) -> StoredExecution | None:
        self._cleanup_terminal(self._clock())
        control = self._controls.get(run_id)
        if control is None:
            return None
        stored = StoredExecution(
            control,
            dict(self._payloads.get(run_id, {})),
            tuple(self._progress.get(run_id, ())),
        )
        validate_stored_execution(stored)
        return stored

    async def transition(self, run_id: str, command: ExecutionCommand) -> TransitionPlan:
        current = self._controls.get(run_id)
        if current is None:
            raise ExecutionNotFoundError(f"execution '{run_id}' was not found")
        command = bind_store_command(command, self._clock(), self.config)
        plan = decide(current, command)
        validate_transition_plan(plan, self.config)
        self._apply(current, plan)
        if not isinstance(command, Heartbeat) and (
            plan.next_control != current
            or plan.payload_writes
            or plan.payload_deletes
            or plan.progress_events
            or plan.lease_index_update
        ):
            log.bind(
                run_id=run_id,
                command=type(command).__name__,
                from_status=current.status.value,
                to_status=plan.next_control.status.value,
                version=plan.next_control.version,
                fence=plan.next_control.fence,
            ).debug("Committed durable execution transition")
        return plan

    async def claim(self, command: Claim) -> TransitionPlan | None:
        if command.lease_duration_ms <= self.config.lease_commit_safety_ms:
            raise ValueError("lease duration must exceed the commit safety margin")
        now_ms = self._clock()
        due = (
            (score, run_id)
            for run_id, score in self._runnable.items()
            if score <= now_ms and self._controls[run_id].definition_revision == command.worker_revision
        )
        try:
            run_id = min(due)[1]
        except ValueError:
            return None
        try:
            return await self.transition(run_id, command)
        except (ExecutionNotFoundError, InvalidExecutionTransitionError):
            self._runnable.pop(run_id, None)
            control = self._controls.get(run_id)
            if control is not None and control.status is ExecutionStatus.QUEUED:
                self._runnable[run_id] = (
                    control.available_at_ms if control.available_at_ms is not None else control.updated_at_ms
                )
            return None

    async def maintain(
        self,
        *,
        max_run_attempts: int,
        attempts_error: bytes,
    ) -> None:
        now_ms = self._clock()
        for (run_id, fence), deadline in sorted(self._lease_expiry.items(), key=lambda item: item[1])[
            :MAINTENANCE_BATCH_SIZE
        ]:
            if deadline > now_ms:
                break
            try:
                await self.transition(
                    run_id,
                    RecoverExpiredLease(
                        0,
                        fence,
                        deadline,
                        max_run_attempts,
                        attempts_error,
                    ),
                )
            except ExecutionNotFoundError:
                self._lease_expiry.pop((run_id, fence), None)
        self._cleanup_terminal(now_ms)

    async def append_chunk(self, run_id: str, attempt: int, fence: int, worker_id: str, data: bytes) -> None:
        if not self.config.max_stream_chunks:
            return
        if attempt < 0:
            raise ValueError("stream chunk attempt cannot be negative")
        validate_payload_size("stream chunk", data, self.config.max_stream_chunk_bytes)
        control = self._controls.get(run_id)
        if control is None or control.run_attempt != attempt:
            raise ExecutionLeaseLostError("execution is no longer owned by this worker fence")
        require_owned(control, fence, worker_id, self._clock(), self.config.lease_commit_safety_ms)
        self._chunk_sequence += 1
        chunks = self._chunks.setdefault(run_id, deque(maxlen=self.config.max_stream_chunks))
        chunks.append(StreamChunk(f"0-{self._chunk_sequence}", attempt, data))

    async def read_chunks(self, run_id: str, after: str) -> tuple[StreamChunk, ...]:
        _, sequence = parse_chunk_cursor(after)
        chunks = tuple(self._chunks.get(run_id, ()))
        if after != CHUNK_CURSOR_START and not any(chunk.cursor == after for chunk in chunks):
            raise ChunkCursorExpiredError(after)
        return tuple(chunk for chunk in chunks if int(chunk.cursor.partition("-")[2]) > sequence)[
            : chunk_read_count(self.config)
        ]

    async def operational_counts(self) -> dict[str, int]:
        self._cleanup_terminal(self._clock())
        return {
            "nonterminal": self._nonterminal,
            "runnable": len(self._runnable),
            "lease_expiry": len(self._lease_expiry),
        }

    def _apply(self, current: ExecutionControl, plan: TransitionPlan, *, new_submission: bool = False) -> None:
        control = plan.next_control
        self._controls[control.run_id] = control
        payloads = self._payloads.setdefault(control.run_id, {})
        for write in plan.payload_writes:
            payloads[write.kind] = write.data
        for kind in plan.payload_deletes:
            payloads.pop(kind, None)

        if plan.progress_events:
            progress = self._progress.setdefault(control.run_id, [])
            progress.extend(plan.progress_events)
            del progress[: -self.config.max_progress_events]

        self._runnable.pop(control.run_id, None)
        if control.status is ExecutionStatus.QUEUED:
            self._runnable[control.run_id] = runnable_score(control)

        if lease := plan.lease_index_update:
            member = (control.run_id, lease.fence)
            if lease.deadline_ms is None:
                self._lease_expiry.pop(member, None)
            else:
                self._lease_expiry[member] = lease.deadline_ms

        if new_submission:
            self._nonterminal += 1
        elif not current.terminal and control.terminal:
            self._nonterminal -= 1
            if self._nonterminal < 0:
                raise ExecutionStoreError("nonterminal execution counter underflow")
            binding = (control.run_id, control.idempotency_binding_digest)
            self._terminal_cleanup[control.run_id] = (
                control.updated_at_ms + self.config.terminal_ttl_seconds * 1_000,
                control.idempotency_digest,
                binding,
            )

    def _cleanup_terminal(self, now_ms: int) -> None:
        for run_id, (expires_at, digest, binding) in tuple(self._terminal_cleanup.items()):
            if expires_at > now_ms:
                continue
            self._terminal_cleanup.pop(run_id)
            if self._idempotency.get(digest) == binding:
                self._idempotency.pop(digest)
            self._controls.pop(run_id, None)
            self._payloads.pop(run_id, None)
            self._progress.pop(run_id, None)
            self._chunks.pop(run_id, None)


def parse_chunk_cursor(value: str) -> tuple[int, int]:
    """Validate the untrusted cursor accepted by streaming transports."""
    if not _CHUNK_CURSOR.fullmatch(value):
        raise ValueError("chunk cursor must be a '<time>-<sequence>' ID")
    left, right = value.split("-", maxsplit=1)
    cursor = (int(left), int(right))
    if max(cursor) > _MAX_CURSOR_PART:
        raise ValueError("chunk cursor parts must be 64-bit unsigned integers")
    return cursor


def chunk_read_count(config: StoreConfig) -> int:
    """Return the shared bounded chunk page size."""
    return max(
        1,
        min(
            MAX_CHUNK_READ_COUNT,
            config.max_stream_chunks or 1,
            MAX_CHUNK_READ_BYTES // config.max_stream_chunk_bytes,
        ),
    )


def runnable_score(control: ExecutionControl) -> int:
    """Return the due-time score shared by every runnable index."""
    return control.available_at_ms if control.available_at_ms is not None else control.updated_at_ms


def bind_store_command(command: ExecutionCommand, now_ms: int, config: StoreConfig) -> ExecutionCommand:
    """Bind a store clock and lease policy before reduction."""
    changes = {"now_ms": now_ms}
    if isinstance(command, _LEASE_COMMANDS):
        changes["lease_commit_safety_ms"] = config.lease_commit_safety_ms
    bound = replace(command, **changes)
    if isinstance(bound, (Claim, Heartbeat, Checkpoint)) and (bound.lease_duration_ms <= config.lease_commit_safety_ms):
        raise ValueError("lease duration must exceed the commit safety margin")
    return bound


def validate_transition_plan(plan: TransitionPlan, config: StoreConfig) -> None:
    """Enforce shared byte limits on effects that the reducer chose to persist."""
    for write in plan.payload_writes:
        validate_payload_size(write.kind.value, write.data, config.max_payload_bytes)
    for event in plan.progress_events:
        validate_payload_size("progress event", event.data, config.max_progress_event_bytes)


def validate_stored_execution(stored: StoredExecution) -> None:
    """Reject payload snapshots that contradict their authoritative lifecycle state."""
    status = stored.control.status
    required = {PayloadKind.INPUT}
    if status is ExecutionStatus.COMPLETED:
        required.add(PayloadKind.RESULT)
    elif status is ExecutionStatus.FAILED:
        required.add(PayloadKind.ERROR)
    elif status is ExecutionStatus.WAITING:
        required.update((PayloadKind.CHECKPOINT, PayloadKind.WAIT))
    if missing := required - stored.payloads.keys():
        names = ", ".join(sorted(kind.value for kind in missing))
        raise ExecutionStoreCorruptionError(f"{status.value} execution is missing required payload: {names}")
    if PayloadKind.RESULT in stored.payloads and status is not ExecutionStatus.COMPLETED:
        raise ExecutionStoreCorruptionError("result payload contradicts execution status")
    if PayloadKind.ERROR in stored.payloads and status in (ExecutionStatus.COMPLETED, ExecutionStatus.CANCELED):
        raise ExecutionStoreCorruptionError("error payload contradicts execution status")
    if PayloadKind.WAIT in stored.payloads and status is not ExecutionStatus.WAITING:
        raise ExecutionStoreCorruptionError("wait payload contradicts execution status")


def validate_payload_size(label: str, payload: bytes, limit: int) -> None:
    if len(payload) > limit:
        raise ExecutionPayloadSizeError(f"{label} exceeds its configured byte limit")
