"""Application context bound to one fenced durable execution claim."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Coroutine, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, TypeVar

from loguru import logger as log

from hayhooks.durable.engine import (
    Checkpoint,
    ExecutionCommand,
    ExecutionControl,
    ExecutionLeaseLostError,
    ExecutionNotFoundError,
    ExecutionStatus,
    Heartbeat,
    Suspend,
    TransitionPlan,
)
from hayhooks.durable.models import CheckpointEnvelope, ExecutionProgress, JsonValue, encode_json
from hayhooks.durable.store import ExecutionStore

_T = TypeVar("_T")


class DurableExecutionCancelledError(RuntimeError):
    """Cooperative cancellation was requested for the active execution."""


class _RetryRequestedError(Exception):
    def __init__(self, message: str, delay: float, progress_events: tuple[bytes, ...]) -> None:
        super().__init__(message)
        self.message = message
        self.delay = delay
        self.progress_events = progress_events


class _ExecutionSuspendedError(Exception):
    pass


class _ClaimedExecution:
    """Fenced store handle and heartbeat owned by one runtime worker."""

    def __init__(
        self,
        store: ExecutionStore,
        control: ExecutionControl,
        worker_id: str,
        lease_duration_ms: int,
        checkpoint: CheckpointEnvelope,
    ) -> None:
        heartbeat_interval = max(0.01, lease_duration_ms / 3_000)
        safe_duration = (lease_duration_ms - store.config.lease_commit_safety_ms) / 1_000
        if control.status is not ExecutionStatus.RUNNING or control.lease_owner != worker_id:
            raise ValueError("a claimed execution requires its running control and lease owner")
        if checkpoint.adapter_kind.value != control.kind:
            raise ValueError("checkpoint kind does not match the claimed execution")
        if safe_duration <= heartbeat_interval:
            raise ValueError("lease duration must leave more than one safe heartbeat interval")
        self.store = store
        self.control = control
        self.worker_id = worker_id
        self.lease_duration_ms = lease_duration_ms
        self.checkpoint = checkpoint
        self.lease_lost = asyncio.Event()
        self.event_loop = asyncio.get_running_loop()
        self._heartbeat_interval = heartbeat_interval
        self._safe_duration = safe_duration
        self._confirmed_until = time.monotonic() + safe_duration
        self._transition_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._finished = False

    async def __aenter__(self) -> _ClaimedExecution:
        await self.transition(Heartbeat(self.control.fence, self.worker_id, 0, self.lease_duration_ms))
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"durable-heartbeat:{self.control.run_id}",
        )
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self._finished = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task

    async def transition(self, command: ExecutionCommand) -> TransitionPlan:
        async with self._transition_lock:
            self.require_owned()
            confirmed_at = time.monotonic()
            try:
                plan = await self.store.transition(self.control.run_id, command)
            except ExecutionLeaseLostError:
                self.mark_lost()
                raise
            except ExecutionNotFoundError as error:
                self.mark_lost()
                raise ExecutionLeaseLostError(f"execution '{self.control.run_id}' no longer exists") from error
            self.control = plan.next_control
            self._confirmed_until = confirmed_at + self._safe_duration
            self._finished = self.control.status is not ExecutionStatus.RUNNING
            return plan

    def require_owned(self) -> None:
        if self.lease_lost.is_set() or self._finished or self.control.status is not ExecutionStatus.RUNNING:
            raise ExecutionLeaseLostError(f"execution lease for '{self.control.run_id}' was lost")

    def mark_lost(self) -> None:
        self.lease_lost.set()

    async def _heartbeat_loop(self) -> None:
        while not self._finished and not self.lease_lost.is_set():
            await asyncio.sleep(self._heartbeat_interval)
            try:
                await self.transition(Heartbeat(self.control.fence, self.worker_id, 0, self.lease_duration_ms))
            except ExecutionLeaseLostError:
                return
            except Exception:
                if time.monotonic() >= self._confirmed_until:
                    self.mark_lost()
                    return


class DurableContext:
    """Checkpoint, progress, cancellation, suspension, retry, and streaming controls."""

    def __init__(self, claim: _ClaimedExecution) -> None:
        self._claim = claim
        self._state = dict(claim.checkpoint.application_state)
        self._resume_input = claim.checkpoint.resume_input
        self._resume_input_consumed = False
        self._pending_progress: list[bytes] = []
        self._operation_lock = asyncio.Lock()
        self._chunk_drop_reported = False

    @property
    def execution_id(self) -> str:
        return self._claim.control.run_id

    @property
    def attempt(self) -> int:
        return self._claim.control.run_attempt

    @property
    def owner_id(self) -> str | None:
        return self._claim.control.owner_id

    @property
    def state(self) -> dict[str, JsonValue]:
        return self._state

    @property
    def resume_input(self) -> JsonValue:
        """Consume the persisted resume value once in this local attempt."""
        value = None if self._resume_input_consumed else self._resume_input
        self._resume_input_consumed = True
        return value

    async def checkpoint(self, adapter_checkpoint: JsonValue = None) -> None:
        async with self._operation_lock:
            self._claim.require_owned()
            snapshot = self._snapshot(adapter_checkpoint)
            plan = await self._claim.transition(
                Checkpoint(
                    self._claim.control.fence,
                    self._claim.worker_id,
                    0,
                    self._claim.lease_duration_ms,
                    encode_json(snapshot.model_dump(mode="json"), max_bytes=self._claim.store.config.max_payload_bytes),
                    tuple(self._pending_progress),
                )
            )
            self._claim.checkpoint = snapshot
            del self._pending_progress[: len(plan.progress_events)]

    async def report_progress(
        self,
        message: str,
        *,
        kind: str = "progress",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        async with self._operation_lock:
            self._claim.require_owned()
            event = ExecutionProgress.model_validate(
                {
                    "sequence": self._claim.control.progress_sequence + len(self._pending_progress) + 1,
                    "kind": kind,
                    "message": message,
                    "timestamp": datetime.now(timezone.utc),
                    "metadata": dict(metadata or {}),
                }
            )
            self._pending_progress.append(
                encode_json(
                    event.model_dump(mode="json", exclude={"sequence"}),
                    max_bytes=self._claim.store.config.max_progress_event_bytes,
                )
            )

    async def check_cancelled(self) -> None:
        self._claim.require_owned()
        control = (
            await self._claim.transition(
                Heartbeat(self._claim.control.fence, self._claim.worker_id, 0, self._claim.lease_duration_ms)
            )
        ).next_control
        if control.cancel_requested_at_ms is not None:
            raise DurableExecutionCancelledError("durable execution cancellation was requested")

    async def retry(self, message: str, *, delay: float | None = None) -> None:
        async with self._operation_lock:
            self._claim.require_owned()
            delay = 0.0 if delay is None else delay
            if delay < 0 or not math.isfinite(delay):
                raise ValueError("retry delay must be a finite non-negative number")
            raise _RetryRequestedError(str(message), delay, tuple(self._pending_progress))

    async def suspend(
        self,
        wait: Mapping[str, object],
        *,
        update: Mapping[str, object] | None = None,
        adapter_checkpoint: JsonValue = None,
    ) -> None:
        async with self._operation_lock:
            self._claim.require_owned()
            snapshot = self._snapshot(adapter_checkpoint, {**self._state, **dict(update or {})})
            plan = await self._claim.transition(
                Suspend(
                    self._claim.control.fence,
                    self._claim.worker_id,
                    0,
                    encode_json(snapshot.model_dump(mode="json"), max_bytes=self._claim.store.config.max_payload_bytes),
                    encode_json(dict(wait), max_bytes=self._claim.store.config.max_payload_bytes),
                    tuple(self._pending_progress),
                )
            )
            self._claim.checkpoint = snapshot
            self._state = dict(snapshot.application_state)
            del self._pending_progress[: len(plan.progress_events)]
            raise _ExecutionSuspendedError

    async def stream_chunk(self, payload: object) -> None:
        self._claim.require_owned()
        try:
            converter = getattr(payload, "to_dict", None)
            if callable(converter):
                payload = converter()
            data = encode_json(payload, max_bytes=self._claim.store.config.max_stream_chunk_bytes)
            await self._claim.store.append_chunk(self.execution_id, self.attempt, data)
        except Exception as error:
            if not self._chunk_drop_reported:
                self._chunk_drop_reported = True
                log.bind(run_id=self.execution_id, exception_type=type(error).__name__).debug(
                    "Dropped a durable display chunk"
                )

    def checkpoint_sync(self, adapter_checkpoint: JsonValue = None) -> None:
        self._sync(self.checkpoint(adapter_checkpoint))

    def report_progress_sync(
        self,
        message: str,
        *,
        kind: str = "progress",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._sync(self.report_progress(message, kind=kind, metadata=metadata))

    def check_cancelled_sync(self) -> None:
        self._sync(self.check_cancelled())

    def retry_sync(self, message: str, *, delay: float | None = None) -> None:
        self._sync(self.retry(message, delay=delay))

    def suspend_sync(
        self,
        wait: Mapping[str, object],
        *,
        update: Mapping[str, object] | None = None,
        adapter_checkpoint: JsonValue = None,
    ) -> None:
        self._sync(self.suspend(wait, update=update, adapter_checkpoint=adapter_checkpoint))

    def stream_chunk_sync(self, payload: object) -> None:
        self._sync(self.stream_chunk(payload))

    def _snapshot(
        self,
        adapter_checkpoint: JsonValue,
        application_state: Mapping[str, object] | None = None,
    ) -> CheckpointEnvelope:
        return CheckpointEnvelope.model_validate(
            {
                "adapter_kind": self._claim.checkpoint.adapter_kind,
                "adapter_checkpoint": (
                    self._claim.checkpoint.adapter_checkpoint if adapter_checkpoint is None else adapter_checkpoint
                ),
                "application_state": dict(self._state if application_state is None else application_state),
                "resume_input": None if self._resume_input_consumed else self._resume_input,
            }
        )

    def _sync(self, coroutine: Coroutine[Any, Any, _T]) -> _T:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._claim.event_loop:
            coroutine.close()
            raise RuntimeError("a synchronous durable context method cannot run on the runtime event loop")
        if self._claim.event_loop.is_closed():
            coroutine.close()
            raise RuntimeError("the durable runtime event loop is closed")
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self._claim.event_loop)
        except RuntimeError:
            coroutine.close()
            raise
        return future.result()


_active_context: ContextVar[DurableContext | None] = ContextVar("hayhooks_durable_context", default=None)


def current_durable_context() -> DurableContext | None:
    """Return the durable context active in this task or worker thread."""
    return _active_context.get()


@contextmanager
def durable_context_scope(context: DurableContext) -> Iterator[None]:
    token = _active_context.set(context)
    try:
        yield
    finally:
        _active_context.reset(token)


def durable_streaming_callback(payload: object) -> None:
    """Forward a synchronous component callback to its active execution."""
    if context := current_durable_context():
        context.stream_chunk_sync(payload)
