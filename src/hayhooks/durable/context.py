"""Store contracts and the context exposed to durable application code."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Coroutine, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, cast, runtime_checkable

from hayhooks.durable.models import (
    DEFAULT_MAX_RECORD_BYTES,
    ExecutionCanceledError,
    ExecutionCheckpoint,
    ExecutionError,
    ExecutionProgressEvent,
    ExecutionRecord,
    ExecutionStatus,
    ExecutionSuspendedError,
    JsonValue,
    RetryableExecutionError,
    validate_json,
)

RESUME_INPUT_KEY = "__hayhooks_resume_input"


class DurableAdapter(Protocol):
    """Haystack operations made available through a durable context."""

    async def run_pipeline_async(
        self, context: DurableContext, data: dict[str, Any], *, checkpoint_at: list[str]
    ) -> dict[str, Any]: ...

    def run_pipeline(
        self, context: DurableContext, data: dict[str, Any], *, checkpoint_at: list[str]
    ) -> dict[str, Any]: ...

    async def run_agent_async(
        self, context: DurableContext, *, messages: list[Any], **kwargs: Any
    ) -> dict[str, Any]: ...

    def run_agent(self, context: DurableContext, *, messages: list[Any], **kwargs: Any) -> dict[str, Any]: ...


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
        self, claim: ExecutionClaim, adapter: DurableAdapter, *, event_loop: asyncio.AbstractEventLoop | None = None
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
