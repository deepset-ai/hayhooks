"""Store contracts and the context exposed to durable application code."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Coroutine, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from hayhooks.durable.models import (
    ExecutionCanceledError,
    ExecutionCheckpoint,
    ExecutionProgressEvent,
    ExecutionStatus,
    ExecutionSuspendedError,
    JsonValue,
    RetryableExecutionError,
    validate_json,
)

if TYPE_CHECKING:
    from hayhooks.durable.adapters import HaystackDurableAdapter
    from hayhooks.durable.store import ExecutionClaim

RESUME_INPUT_KEY = "__hayhooks_resume_input"


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
        self,
        claim: ExecutionClaim,
        adapter: HaystackDurableAdapter,
        *,
        event_loop: asyncio.AbstractEventLoop | None = None,
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
    def owner_id(self) -> str | None:
        """Return the stable owner identity persisted with this execution."""
        return self.record.owner_id

    @property
    def state(self) -> dict[str, JsonValue]:
        return self.record.application_state

    @property
    def resume_input(self) -> JsonValue | None:
        """Return the most recently persisted resume payload without consuming it."""
        return self.record.application_state.get(RESUME_INPUT_KEY)

    def take_resume_input(self) -> JsonValue | None:
        """Consume the persisted resume payload exactly once within this attempt."""
        return self.record.application_state.pop(RESUME_INPUT_KEY, None)

    async def checkpoint(self, checkpoint: ExecutionCheckpoint | None = None) -> None:
        if checkpoint is not None:
            if checkpoint.kind is not self.record.execution_kind:
                msg = (
                    f"{checkpoint.kind.value} checkpoint cannot be used for "
                    f"{self.record.execution_kind.value} execution"
                )
                raise ValueError(msg)
            checkpoint.data = cast(
                dict[str, JsonValue],
                validate_json(checkpoint.data, limit=self.record.max_record_bytes, label="checkpoint"),
            )
            self.record.checkpoint = checkpoint
        await self.claim.checkpoint()

    async def report_progress(
        self, message: str, *, kind: str = "progress", metadata: Mapping[str, Any] | None = None
    ) -> ExecutionProgressEvent:
        event = self.record.append_progress(message, kind=kind, metadata=metadata)
        await self.claim.checkpoint()
        return event

    async def stream_chunk(self, payload: Any) -> None:
        """Append one best-effort display chunk outside the durable fence."""
        await self.claim.stream_chunk(payload.to_dict() if hasattr(payload, "to_dict") else payload)

    def stream_chunk_sync(self, payload: Any) -> None:
        """Synchronous counterpart for sync wrappers running in a worker thread."""
        # ponytail: this blocks the pipeline thread on one Redis round trip per chunk
        # via _sync_await, so one run's token rate is capped at roughly 1/RTT -- about
        # 50 tokens/s against a managed Redis 20 ms away. Batch through an asyncio.Queue
        # drained by a single task if sync-wrapper generation latency shows up.
        with suppress(Exception):
            # The bridge itself fails once the manager loop is gone, which a wrapper
            # still generating past the shutdown grace period can reach. A display
            # chunk is never worth failing the run it describes.
            self._sync_await(self.stream_chunk(payload))

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
            dict[str, JsonValue], validate_json(dict(wait), limit=self.record.max_record_bytes, label="wait")
        )
        if update is not None:
            self.record.application_state.update(
                cast(
                    dict[str, JsonValue],
                    validate_json(dict(update), limit=self.record.max_record_bytes, label="wait update"),
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
