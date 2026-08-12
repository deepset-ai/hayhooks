"""Advanced durable execution contracts and safe public result models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from hayhooks.durable.context import get_current_durable_context
from hayhooks.durable.mode import DurableAuthoringMode, durable_authoring_mode
from hayhooks.durable.models import ExecutionStatus

if TYPE_CHECKING:
    from hayhooks.durable.runtime import DurableRuntime, ExecutionStoreProvider
    from hayhooks.durable.store import ExecutionStore, InMemoryExecutionStoreProvider, RedisExecutionStoreProvider


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


def current_execution_id() -> str | None:
    """Return the active durable execution ID for hooks and idempotent tools."""
    context = get_current_durable_context()
    return context.execution_id if context is not None else None


def current_durable_context() -> Any | None:
    """Return the active context for advanced hooks and tools."""
    return get_current_durable_context()


def __getattr__(name: str) -> Any:
    """Lazily expose durable infrastructure without eager optional imports."""
    if name in {"DurableRuntime", "ExecutionStoreProvider", "durable_runtime"}:
        from hayhooks.durable.runtime import DurableRuntime, ExecutionStoreProvider, durable_runtime

        return {
            "DurableRuntime": DurableRuntime,
            "ExecutionStoreProvider": ExecutionStoreProvider,
            "durable_runtime": durable_runtime,
        }[name]
    if name in {"ExecutionStore", "InMemoryExecutionStoreProvider", "RedisExecutionStoreProvider"}:
        from hayhooks.durable.store import ExecutionStore, InMemoryExecutionStoreProvider, RedisExecutionStoreProvider

        return {
            "ExecutionStore": ExecutionStore,
            "InMemoryExecutionStoreProvider": InMemoryExecutionStoreProvider,
            "RedisExecutionStoreProvider": RedisExecutionStoreProvider,
        }[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "DurableAuthoringMode",
    "DurableRuntime",
    "ExecutionProgress",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionStore",
    "ExecutionStoreProvider",
    "InMemoryExecutionStoreProvider",
    "RedisExecutionStoreProvider",
    "current_durable_context",
    "current_execution_id",
    "durable_authoring_mode",
    "durable_runtime",
]
