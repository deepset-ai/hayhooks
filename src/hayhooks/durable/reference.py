"""Deterministic in-memory implementation of the lean durable backend."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable

from hayhooks.durable.backend import (
    CHUNK_READ_COUNT,
    MAINTENANCE_BATCH_SIZE,
    ExecutionIdempotencyConflictError,
    ExecutionStoreConfig,
    SubmissionResult,
    bind_command,
    bind_progress_sequences,
    check_admission,
    parse_chunk_cursor,
    parse_idempotency_binding,
    parse_lease_member,
    runnable_score,
    validate_command_payloads,
)
from hayhooks.durable.engine import (
    ExecutionCommand,
    ExecutionControl,
    ExecutionNotFoundError,
    ExecutionStatus,
    InvalidExecutionTransitionError,
    PayloadKind,
    TransitionPlan,
    decide,
    submission_plan,
)


class InMemoryExecutionStore:
    """Single-process reference model used by the shared backend contract."""

    def __init__(self, *, deployment: str, config: ExecutionStoreConfig | None = None) -> None:
        self.deployment = deployment
        self.config = config or ExecutionStoreConfig()
        self._controls: dict[str, ExecutionControl] = {}
        self._payloads: dict[str, dict[PayloadKind, bytes]] = {}
        self._progress: dict[str, list[bytes]] = {}
        self._chunks: dict[str, deque[tuple[int, int, bytes]]] = {}
        self._chunk_sequence = 0
        self._runnable: dict[str, int] = {}
        self._lease_expiry: dict[str, int] = {}
        self._capacity = {"nonterminal": 0}
        self._idempotency: dict[str, str] = {}
        self._terminal_cleanup: dict[str, tuple[int, str, str]] = {}

    @staticmethod
    def _now_ms() -> int:
        return round(time.time() * 1_000)

    async def initialize(self) -> None:
        return None

    async def submit(self, control: ExecutionControl, input_payload: bytes, *, binding_digest: str) -> SubmissionResult:
        if control.deployment != self.deployment:
            raise ValueError("control deployment does not match this store")
        if len(input_payload) > self.config.max_input_bytes:
            raise ValueError("input payload exceeds configured size")
        existing = self._idempotency.get(control.idempotency_digest)
        if existing is not None:
            existing_run, existing_binding = parse_idempotency_binding(existing)
            existing_control = self._controls.get(existing_run)
            if existing_binding != binding_digest:
                raise ExecutionIdempotencyConflictError("idempotency key is bound to another request")
            if existing_control is not None:
                return SubmissionResult(created=False, control=existing_control)
        check_admission(self._capacity, self.config)
        self._idempotency[control.idempotency_digest] = f"{control.run_id}|{binding_digest}"
        self._apply_plan(control, submission_plan(control, input_payload))
        self._capacity["nonterminal"] += 1
        return SubmissionResult(created=True, control=control)

    async def get(self, run_id: str) -> ExecutionControl | None:
        return self._controls.get(run_id)

    async def read_payloads(self, run_id: str, kinds: tuple[PayloadKind, ...]) -> dict[PayloadKind, bytes | None]:
        return {kind: self._payloads.get(run_id, {}).get(kind) for kind in kinds}

    async def read_progress(self, run_id: str) -> list[bytes]:
        return list(self._progress.get(run_id, ()))

    async def append_chunk(self, run_id: str, attempt: int, chunk: bytes) -> None:
        # A stale append from a worker that lost its lease must not resurrect a log
        # whose terminal cleanup already ran; the control is dropped by that cleanup,
        # so its absence is the same signal the Redis backend gets from a missing key.
        if run_id not in self._controls:
            return
        self._chunk_sequence += 1
        entries = self._chunks.setdefault(run_id, deque(maxlen=self.config.max_stream_chunks))
        # Sequences only have to be orderable, not Redis-shaped; a counter cannot
        # regress the way a wall clock can. Entry IDs are formatted on the way out.
        entries.append((self._chunk_sequence, attempt, chunk))
        # A stale append before that cleanup still lands, and the terminal cleanup
        # entry already registered for the run removes it at the terminal TTL,
        # mirroring the Redis backend's EXPIRE.

    async def read_chunks(self, run_id: str, after: str, *, block_ms: int) -> list[tuple[str, int, bytes]]:
        # ponytail: a 20 ms poll instead of an asyncio.Condition, so a dev-mode SSE
        # client sees up to 20 ms of extra per-token latency. Swap in a Condition if
        # that ever shows; the Redis backend already blocks natively.
        deadline = time.monotonic() + block_ms / 1_000
        _, cursor = parse_chunk_cursor(after)
        while True:
            fresh = [entry for entry in self._chunks.get(run_id, ()) if entry[0] > cursor][:CHUNK_READ_COUNT]
            if fresh or time.monotonic() >= deadline:
                return [(f"0-{sequence}", attempt, chunk) for sequence, attempt, chunk in fresh]
            await asyncio.sleep(0.02)

    async def transition(self, run_id: str, command: ExecutionCommand, *, candidate: bool = False) -> TransitionPlan:
        current = self._controls.get(run_id)
        if current is None:
            raise ExecutionNotFoundError(f"execution '{run_id}' was not found")
        command = bind_command(
            command, now_ms=self._now_ms(), lease_commit_safety_ms=self.config.lease_commit_safety_ms
        )
        validate_command_payloads(command, self.config)
        try:
            plan = bind_progress_sequences(decide(current, command), self.config)
        except InvalidExecutionTransitionError:
            if not candidate:
                raise
            self._runnable.pop(run_id, None)
            if current.status is ExecutionStatus.QUEUED:
                self._runnable[run_id] = runnable_score(current)
            return TransitionPlan(current)
        self._apply_plan(current, plan)
        return plan

    async def read_candidate(self) -> str | None:
        now_ms = self._now_ms()
        due = ((score, run_id) for run_id, score in self._runnable.items() if score <= now_ms)
        try:
            return min(due)[1]
        except ValueError:
            return None

    async def maintain(self, command_factory: Callable[[int, int], ExecutionCommand]) -> int:
        now_ms = self._now_ms()
        recovered = 0
        for member, deadline in sorted(self._lease_expiry.items(), key=lambda item: item[1])[:MAINTENANCE_BATCH_SIZE]:
            if deadline > now_ms:
                break
            run_id, fence = parse_lease_member(member)
            try:
                await self.transition(run_id, command_factory(fence, deadline))
                recovered += 1
            except ExecutionNotFoundError:
                self._lease_expiry.pop(member, None)
        self._cleanup_terminal(now_ms)
        return recovered

    async def operational_counts(self) -> dict[str, int]:
        return {
            "nonterminal": self._capacity["nonterminal"],
            "runnable": len(self._runnable),
            "lease_expiry": len(self._lease_expiry),
        }

    def _cleanup_terminal(self, now_ms: int) -> None:
        for run_id, (expires_at, idem_digest, idem_value) in tuple(self._terminal_cleanup.items()):
            if expires_at > now_ms:
                continue
            self._terminal_cleanup.pop(run_id)
            if self._idempotency.get(idem_digest) == idem_value:
                self._idempotency.pop(idem_digest)
            self._controls.pop(run_id, None)
            self._payloads.pop(run_id, None)
            self._progress.pop(run_id, None)
            self._chunks.pop(run_id, None)

    def _apply_plan(self, current: ExecutionControl, plan: TransitionPlan) -> None:
        next_control = plan.next_control
        self._controls[next_control.run_id] = next_control
        payloads = self._payloads.setdefault(next_control.run_id, {})
        for write in plan.payload_writes:
            payloads[write.kind] = write.data
        for kind in plan.payload_deletes:
            payloads.pop(kind, None)
        if plan.progress_events:
            progress = self._progress.setdefault(next_control.run_id, [])
            progress.extend(event.data for event in plan.progress_events)
            del progress[: -self.config.max_progress_events]

        self._runnable.pop(next_control.run_id, None)
        if next_control.status is ExecutionStatus.QUEUED:
            self._runnable[next_control.run_id] = runnable_score(next_control)

        if plan.lease_index_update is not None:
            member = f"{next_control.run_id}|{plan.lease_index_update.fence}"
            if plan.lease_index_update.deadline_ms is None:
                self._lease_expiry.pop(member, None)
            else:
                self._lease_expiry[member] = plan.lease_index_update.deadline_ms

        if not current.terminal and next_control.terminal:
            self._capacity["nonterminal"] -= 1
            if self._capacity["nonterminal"] < 0:
                raise RuntimeError("reference capacity counter underflow")
            self._terminal_cleanup[next_control.run_id] = (
                next_control.updated_at_ms + self.config.terminal_ttl_seconds * 1_000,
                next_control.idempotency_digest,
                f"{next_control.run_id}|{next_control.idempotency_binding_digest}",
            )
