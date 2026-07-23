"""Advanced durable execution contracts and safe public result models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from hayhooks.execution import (
    DurableExecutionStore,
    DurableExecutionStoreProvider,
    ExecutionProgressEvent,
    ExecutionStatus,
    InMemoryExecutionStore,
    InMemoryExecutionStoreProvider,
    get_current_durable_context,
)


@dataclass(frozen=True)
class DurableOptions:
    """Optional explicit definition revision for non-serializable deployments."""

    revision: str | None = None


class ExecutionProgress(BaseModel):
    """Sanitized client-visible progress event."""

    sequence: int
    kind: str
    message: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_event(cls, event: ExecutionProgressEvent) -> ExecutionProgress:
        return cls.model_validate(event.to_dict())


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


__all__ = [
    "DurableExecutionStore",
    "DurableExecutionStoreProvider",
    "DurableOptions",
    "ExecutionProgress",
    "ExecutionResult",
    "ExecutionStatus",
    "InMemoryExecutionStore",
    "InMemoryExecutionStoreProvider",
    "current_durable_context",
    "current_execution_id",
]
