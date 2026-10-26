"""Durable persistence contract and in-memory reference implementation."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import asyncio
import base64
import bisect
import hashlib
import heapq
import json
import re
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

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
    became_terminal,
    decide,
    plan_changes,
    require_owned,
    submission_plan,
    validate_run_id,
)

if TYPE_CHECKING:
    from hayhooks.durable.removal import RemovalScope, StoreRemovalSnapshot

CHUNK_CURSOR_START = "0-0"
MAINTENANCE_BATCH_SIZE = 100
MAINTENANCE_MAX_BATCHES = 10
MAX_CHUNK_READ_BYTES = 4_000_000
MAX_CHUNK_READ_COUNT = 1_000
MAX_EXECUTION_PAGE_SIZE = 100
EXECUTION_CANDIDATE_LIMIT = 200
EXECUTION_CURSOR_STAMP_WIDTH = 20
MAX_EXECUTION_SEARCH_LENGTH = 128
MAX_EXECUTION_REVISION_LENGTH = 512
MAX_EXECUTION_CURSOR_LENGTH = 1024
_CHUNK_CURSOR = re.compile(r"^\d{1,20}-\d{1,20}$")
_MAX_CURSOR_PART = 2**64 - 1
LEASE_COMMANDS = (ReleaseClaim, Heartbeat, Checkpoint, ScheduleRetry, Suspend, Complete, Fail)
PUBLIC_PAYLOAD_KINDS = (PayloadKind.RESULT, PayloadKind.ERROR, PayloadKind.WAIT)


class ExecutionStoreError(RuntimeError):
    """A durable store operation failed."""


class ExecutionStoreCorruptionError(ExecutionStoreError):
    """Persisted state cannot be decoded without violating durable invariants."""


class ExecutionProgressCorruptionError(ExecutionStoreCorruptionError):
    """Retained progress cannot be decoded consistently with its control."""


class ExecutionContentionError(ExecutionStoreError):
    """A bounded optimistic transaction could not obtain a stable snapshot."""


class ExecutionAdmissionError(RuntimeError):
    """Admission was refused: the nonterminal limit is reached or the deployment stopped accepting submissions."""


class ExecutionIdempotencyConflictError(RuntimeError):
    """An idempotency key was reused for different work."""


class ChunkCursorExpiredError(RuntimeError):
    """A requested chunk cursor is no longer retained."""


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """Limits shared by every durable store implementation."""

    lease_commit_safety_ms: int = 1_500
    terminal_ttl_seconds: int = 604_800
    stream_ttl_seconds: int = 3_600
    max_nonterminal_executions: int = 1_000
    max_payload_bytes: int = 1_000_000
    max_progress_events: int = 100
    max_progress_event_bytes: int = 8_192
    max_stream_chunks: int = 100
    max_stream_chunk_bytes: int = 64_000
    monitoring_enabled: bool = False

    def __post_init__(self) -> None:
        if min(self.lease_commit_safety_ms, self.max_nonterminal_executions, self.max_stream_chunks) < 0:
            raise ValueError("durable store limits cannot be negative")
        for name in (
            "terminal_ttl_seconds",
            "stream_ttl_seconds",
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
    """One display chunk, terminal marker, or undecodable entry that readers skip."""

    cursor: str
    attempt: int
    data: bytes
    terminal: bool = False
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class ControlPage:
    controls: tuple[ExecutionControl, ...]
    next_cursor: str | None
    has_more: bool
    skipped_controls: int = 0


def monitoring_scope(owner_id: str | None) -> str:
    # Separate unscoped access from an explicitly empty owner.
    return hashlib.sha256(json.dumps(owner_id).encode()).hexdigest()


def creation_member(control: ExecutionControl) -> str:
    return f"{control.created_at_ms:0{EXECUTION_CURSOR_STAMP_WIDTH}d}:{control.run_id}"


@dataclass(frozen=True, slots=True)
class ControlQuery:
    """Bounded, owner-scoped creation pagination shared by both stores."""

    owner_id: str | None = None
    limit: int = 50
    cursor: str | None = None
    status: ExecutionStatus | None = None
    revision: str | None = None
    search: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.limit <= MAX_EXECUTION_PAGE_SIZE:
            raise ValueError("limit must be between 1 and 100")
        if len(self.search) > MAX_EXECUTION_SEARCH_LENGTH or (
            self.revision is not None and not 1 <= len(self.revision) <= MAX_EXECUTION_REVISION_LENGTH
        ):
            raise ValueError("execution filter is too long or empty")
        self.after()

    @property
    def scope(self) -> str:
        return monitoring_scope(self.owner_id)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps([self.scope, self.status, self.revision, self.search]).encode()).hexdigest()

    def after(self) -> str:
        if self.cursor is None:
            return ""
        try:
            if len(self.cursor) > MAX_EXECUTION_CURSOR_LENGTH:
                raise ValueError
            fingerprint, member = json.loads(base64.b64decode(self.cursor, altchars=b"-_", validate=True))
            if not isinstance(fingerprint, str) or not isinstance(member, str):
                raise ValueError
            stamp, run_id = member.split(":", 1)
            validate_run_id(run_id)
            if (
                fingerprint != self.fingerprint
                or len(stamp) != EXECUTION_CURSOR_STAMP_WIDTH
                or not stamp.isascii()
                or not stamp.isdigit()
            ):
                raise ValueError
            return member
        except (ValueError, TypeError, UnicodeError) as error:
            raise ValueError("invalid execution cursor for this owner and query") from error

    def encode_cursor(self, member: str) -> str:
        return base64.urlsafe_b64encode(json.dumps([self.fingerprint, member]).encode()).decode()

    def matches(self, control: ExecutionControl) -> bool:
        return (
            control.owner_id == self.owner_id
            and (self.status is None or control.status == self.status)
            and (self.revision is None or control.definition_revision == self.revision)
            and self.search.lower() in control.run_id.lower()
        )


class ExecutionStore(Protocol):
    """Atomic persistence required by the durable runtime."""

    config: StoreConfig
    deployment: str

    async def initialize(self) -> None: ...

    async def submit(self, control: ExecutionControl, input_payload: bytes) -> SubmissionResult: ...

    async def read(self, run_id: str) -> StoredExecution | None: ...

    async def read_public(self, run_id: str) -> StoredExecution | None:
        """Read control, progress, and public payloads without input or checkpoint bytes."""
        ...

    async def read_control(self, run_id: str) -> ExecutionControl | None:
        """Read only the decoded control, or ``None`` when the execution does not exist."""
        ...

    async def list_controls(self, query: ControlQuery) -> ControlPage:
        """Read at most 200 indexed candidates, never payloads or another owner's index."""
        ...

    async def transition(self, run_id: str, command: ExecutionCommand) -> TransitionPlan: ...

    async def claim(self, command: Claim) -> TransitionPlan | None: ...

    async def maintain(
        self,
        *,
        max_run_attempts: int,
        attempts_error: bytes,
    ) -> int:
        """Recover due expired leases and return how many executions were requeued."""
        ...

    async def append_chunks(
        self, run_id: str, attempt: int, fence: int, worker_id: str, chunks: Sequence[bytes]
    ) -> None: ...

    async def read_chunks(self, run_id: str, after: str) -> tuple[StreamChunk, ...]: ...

    async def wait_chunks(self, run_id: str, after: str, timeout: float) -> tuple[StreamChunk, ...]:
        """Like ``read_chunks``, but block up to ``timeout`` seconds for a first entry; empty on timeout."""
        ...

    async def operational_counts(self, *, revision: str) -> dict[str, int]:
        """Return deployment and revision nonterminal, runnable, and lease counts."""
        ...


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
        self._chunks_written = asyncio.Condition()
        self._runnable: dict[str, int] = {}
        self._lease_expiry: dict[tuple[str, int], int] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._terminal_cleanup: dict[str, tuple[int, int, str, tuple[str, str]]] = {}
        self._terminal_expiry: list[tuple[int, str]] = []
        self._chunk_expiry: list[tuple[int, str]] = []
        self._nonterminal = 0
        self._removal_fences: set[RemovalScope] = set()
        # ponytail: sorted lists cost O(n) per insertion/removal; use Redis for large monitored collections.
        self._execution_index: dict[str, list[str]] = {}

    async def initialize(self) -> None:
        return None

    async def submit(self, control: ExecutionControl, input_payload: bytes) -> SubmissionResult:
        if control.deployment != self.deployment:
            raise ValueError("control deployment does not match this store")
        validate_payload_size("input", input_payload, self.config.max_payload_bytes)
        self._cleanup_terminal(self._clock())
        self._expire_control(control.run_id, self._clock())
        if existing := self._idempotency.get(control.idempotency_digest):
            self._expire_control(existing[0], self._clock())
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
        if self.config.monitoring_enabled:
            bisect.insort(
                self._execution_index.setdefault(monitoring_scope(control.owner_id), []), creation_member(control)
            )
        log.bind(run_id=control.run_id, deployment=control.deployment).debug("Submitted durable execution")
        return SubmissionResult(created=True, control=control)

    async def read(self, run_id: str) -> StoredExecution | None:
        return self._read(run_id, private=True)

    async def read_public(self, run_id: str) -> StoredExecution | None:
        return self._read(run_id, private=False)

    async def read_control(self, run_id: str) -> ExecutionControl | None:
        self._cleanup_terminal(self._clock())
        self._expire_control(run_id, self._clock())
        return self._controls.get(run_id)

    async def list_controls(self, query: ControlQuery) -> ControlPage:
        if not self.config.monitoring_enabled:
            raise ExecutionStoreError("execution monitoring is disabled")
        self._cleanup_terminal(self._clock())
        index = self._execution_index.get(query.scope, [])
        start = bisect.bisect_right(index, query.after())
        candidates = index[start : start + EXECUTION_CANDIDATE_LIMIT + 1]
        controls = []
        examined = 0
        for member in candidates[:EXECUTION_CANDIDATE_LIMIT]:
            examined += 1
            self._expire_control(member.split(":", 1)[1], self._clock())
            control = self._controls.get(member.split(":", 1)[1])
            if control is not None and query.matches(control):
                controls.append(control)
                if len(controls) == query.limit:
                    break
        more = examined < len(candidates)
        cursor = query.encode_cursor(candidates[examined - 1]) if more and examined else None
        return ControlPage(tuple(controls), cursor, more)

    async def transition(self, run_id: str, command: ExecutionCommand) -> TransitionPlan:
        from hayhooks.durable.removal import worker_scope

        current = self._controls.get(run_id)
        if current is None:
            raise ExecutionNotFoundError(f"execution '{run_id}' was not found")
        if isinstance(command, Claim) and worker_scope(command.worker_id) in self._removal_fences:
            raise InvalidExecutionTransitionError("the worker ownership scope is fenced for removal")
        command = bind_store_command(command, max(self._clock(), current.updated_at_ms), self.config)
        plan = decide(current, command)
        validate_transition_plan(plan, self.config)
        self._apply(current, plan)
        if became_terminal(current, plan):
            await self._write_chunks(run_id, plan.next_control.run_attempt, (b"",), terminal=True)
        if not isinstance(command, Heartbeat) and plan_changes(current, plan):
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
        validate_lease_duration(command.lease_duration_ms, self.config)
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
                self._runnable[run_id] = runnable_score(control)
            return None

    async def maintain(
        self,
        *,
        max_run_attempts: int,
        attempts_error: bytes,
    ) -> int:
        now_ms = self._clock()
        requeued = 0
        for (run_id, fence), deadline in sorted(self._lease_expiry.items(), key=lambda item: item[1])[
            : MAINTENANCE_BATCH_SIZE * MAINTENANCE_MAX_BATCHES
        ]:
            if deadline > now_ms:
                break
            before = self._controls.get(run_id)
            try:
                plan = await self.transition(
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
            except InvalidExecutionTransitionError:
                continue
            else:
                requeued += (
                    before is not None
                    and before.status is ExecutionStatus.RUNNING
                    and plan.next_control.status is ExecutionStatus.QUEUED
                )
        self._cleanup_terminal(now_ms)
        return requeued

    async def append_chunks(
        self, run_id: str, attempt: int, fence: int, worker_id: str, chunks: Sequence[bytes]
    ) -> None:
        if not self.config.max_stream_chunks:
            return
        if attempt < 0:
            raise ValueError("stream chunk attempt cannot be negative")
        for data in chunks:
            validate_payload_size("stream chunk", data, self.config.max_stream_chunk_bytes)
        control = self._controls.get(run_id)
        if control is None or control.run_attempt != attempt:
            raise ExecutionLeaseLostError("execution is no longer owned by this worker fence")
        require_owned(control, fence, worker_id, self._clock(), self.config.lease_commit_safety_ms)
        await self._write_chunks(run_id, attempt, chunks)

    async def read_chunks(self, run_id: str, after: str) -> tuple[StreamChunk, ...]:
        self._cleanup_terminal(self._clock())
        return self._chunks_after(run_id, after)

    async def wait_chunks(self, run_id: str, after: str, timeout: float) -> tuple[StreamChunk, ...]:
        self._cleanup_terminal(self._clock())
        async with self._chunks_written:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._chunks_written.wait_for(lambda: self._chunks_after(run_id, after)),
                    timeout,
                )
        return self._chunks_after(run_id, after)

    async def owner_scope_fenced(self, scope: RemovalScope) -> bool:
        return scope in self._removal_fences

    async def fence_owner_scope(self, scope: RemovalScope) -> None:
        """Permanently reject new claims for this scope, including other deployment instances."""
        self._removal_fences.add(scope)

    async def removal_snapshot(self, *, scope: RemovalScope | None = None) -> StoreRemovalSnapshot:
        """Read namespace counts and inspect at most 200 live lease controls for scoped proof."""
        from hayhooks.durable.removal import MAX_REMOVAL_LEASE_CONTROLS, StoreRemovalSnapshot, worker_scope

        now = self._clock()
        live = [(key, deadline) for key, deadline in self._lease_expiry.items() if deadline > now]
        blockers = []
        owned = 0
        if scope is not None:
            if len(live) > MAX_REMOVAL_LEASE_CONTROLS:
                blockers.append("lease_probe_saturated")
            else:
                for (run_id, fence), deadline in live:
                    control = self._controls.get(run_id)
                    if (
                        control is None
                        or control.status is not ExecutionStatus.RUNNING
                        or control.fence != fence
                        or control.lease_expires_at_ms != deadline
                        or control.lease_owner is None
                    ):
                        blockers.append("lease_index_inconsistent")
                        continue
                    candidate = worker_scope(control.lease_owner)
                    if candidate == scope:
                        owned += 1
                    elif scope.kind == "owner_family" and (candidate is None or candidate.kind != "owner_family"):
                        blockers.append("legacy_lease_ownership_unknown")
        return StoreRemovalSnapshot(
            now,
            self._nonterminal,
            len(live),
            None if scope is None or blockers else owned,
            tuple(sorted(set(blockers))),
        )

    async def operational_counts(self, *, revision: str) -> dict[str, int]:
        self._cleanup_terminal(self._clock())
        return {
            "nonterminal": self._nonterminal,
            "revision_nonterminal": sum(
                not control.terminal and control.definition_revision == revision for control in self._controls.values()
            ),
            "revision_runnable": sum(
                self._controls[run_id].definition_revision == revision for run_id in self._runnable
            ),
            "lease_expiry": len(self._lease_expiry),
        }

    def _apply(  # noqa: C901
        self, current: ExecutionControl, plan: TransitionPlan, *, new_submission: bool = False
    ) -> None:
        control = plan.next_control
        self._controls[control.run_id] = control
        payloads = self._payloads.setdefault(control.run_id, {})
        for write in plan.payload_writes:
            payloads[write.kind] = write.data
        for kind in plan.payload_deletes:
            payloads.pop(kind, None)

        if plan.discard_progress:
            self._progress.pop(control.run_id, None)
        elif plan.progress_events:
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
        elif became_terminal(current, plan):
            self._nonterminal -= 1
            if self._nonterminal < 0:
                raise ExecutionStoreError("nonterminal execution counter underflow")
            binding = (control.run_id, control.idempotency_binding_digest)
            self._terminal_cleanup[control.run_id] = (
                control.updated_at_ms + self.config.terminal_ttl_seconds * 1_000,
                control.updated_at_ms + min(self.config.stream_ttl_seconds, self.config.terminal_ttl_seconds) * 1_000,
                control.idempotency_digest,
                binding,
            )
            expiry, chunk_expiry, _, _ = self._terminal_cleanup[control.run_id]
            heapq.heappush(self._terminal_expiry, (expiry, control.run_id))
            heapq.heappush(self._chunk_expiry, (chunk_expiry, control.run_id))

    def _read(self, run_id: str, *, private: bool) -> StoredExecution | None:
        self._cleanup_terminal(self._clock())
        self._expire_control(run_id, self._clock())
        control = self._controls.get(run_id)
        if control is None:
            return None
        payloads = self._payloads.get(run_id, {})
        stored = StoredExecution(
            control,
            dict(payloads) if private else {kind: payloads[kind] for kind in PUBLIC_PAYLOAD_KINDS if kind in payloads},
            tuple(self._progress.get(run_id, ())),
        )
        validate_stored_execution(stored, private=private)
        return stored

    def _chunks_after(self, run_id: str, after: str) -> tuple[StreamChunk, ...]:
        now_ms = self._clock()
        self._expire_control(run_id, now_ms)
        if (entry := self._terminal_cleanup.get(run_id)) is not None and entry[1] <= now_ms:
            self._chunks.pop(run_id, None)
        _, sequence = parse_chunk_cursor(after)
        chunks = tuple(self._chunks.get(run_id, ()))
        if after != CHUNK_CURSOR_START and not any(chunk.cursor == after for chunk in chunks):
            raise ChunkCursorExpiredError(after)
        return tuple(chunk for chunk in chunks if int(chunk.cursor.partition("-")[2]) > sequence)[
            : chunk_read_count(self.config)
        ]

    async def _write_chunks(
        self, run_id: str, attempt: int, chunks: Sequence[bytes], *, terminal: bool = False
    ) -> None:
        stream = self._chunks.setdefault(run_id, deque(maxlen=max(self.config.max_stream_chunks, 1)))
        for data in chunks:
            self._chunk_sequence += 1
            stream.append(StreamChunk(f"0-{self._chunk_sequence}", attempt, data, terminal))
        async with self._chunks_written:
            self._chunks_written.notify_all()

    def _cleanup_terminal(self, now_ms: int) -> None:
        for queue, chunks_only in ((self._chunk_expiry, True), (self._terminal_expiry, False)):
            for _ in range(MAINTENANCE_BATCH_SIZE):
                if not queue or queue[0][0] > now_ms:
                    break
                _, run_id = heapq.heappop(queue)
                if chunks_only:
                    if (entry := self._terminal_cleanup.get(run_id)) is not None and entry[1] <= now_ms:
                        self._chunks.pop(run_id, None)
                else:
                    self._expire_control(run_id, now_ms)

    def _expire_control(self, run_id: str, now_ms: int) -> None:
        entry = self._terminal_cleanup.get(run_id)
        if entry is None or entry[0] > now_ms:
            return
        _, _, digest, binding = self._terminal_cleanup.pop(run_id)
        if self._idempotency.get(digest) == binding:
            self._idempotency.pop(digest)
        control = self._controls.pop(run_id, None)
        if control is not None:
            index = self._execution_index.get(monitoring_scope(control.owner_id), [])
            member = creation_member(control)
            position = bisect.bisect_left(index, member)
            if position < len(index) and index[position] == member:
                index.pop(position)
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
    if isinstance(command, LEASE_COMMANDS):
        changes["lease_commit_safety_ms"] = config.lease_commit_safety_ms
    bound = replace(command, **changes)
    if isinstance(bound, (Claim, Heartbeat, Checkpoint)):
        validate_lease_duration(bound.lease_duration_ms, config)
    return bound


def validate_lease_duration(lease_duration_ms: int, config: StoreConfig) -> None:
    if lease_duration_ms <= config.lease_commit_safety_ms:
        raise ValueError("lease duration must exceed the commit safety margin")


def validate_transition_plan(plan: TransitionPlan, config: StoreConfig) -> None:
    """Enforce shared byte limits on effects that the reducer chose to persist."""
    for write in plan.payload_writes:
        validate_payload_size(write.kind.value, write.data, config.max_payload_bytes)
    for event in plan.progress_events:
        validate_payload_size("progress event", event.data, config.max_progress_event_bytes)


def validate_stored_execution(stored: StoredExecution, *, private: bool = True) -> None:
    """
    Reject payload snapshots that contradict their authoritative lifecycle state.

    Public snapshots carry no input or checkpoint, so only full reads require them.
    """
    status = stored.control.status
    required = {PayloadKind.INPUT} if private else set()
    if status is ExecutionStatus.COMPLETED:
        required.add(PayloadKind.RESULT)
    elif status is ExecutionStatus.FAILED:
        required.add(PayloadKind.ERROR)
    elif status is ExecutionStatus.WAITING:
        required.update((PayloadKind.CHECKPOINT, PayloadKind.WAIT) if private else (PayloadKind.WAIT,))
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
