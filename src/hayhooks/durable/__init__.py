"""Public durable execution API."""

from __future__ import annotations

from hayhooks.durable.context import DurableContext, get_current_durable_context
from hayhooks.durable.fastapi import create_durable_router
from hayhooks.durable.mode import DurableAuthoringMode, durable_authoring_mode
from hayhooks.durable.models import (
    ExecutionAdmissionError,
    ExecutionCanceledError,
    ExecutionProgress,
    ExecutionRecordSizeError,
    ExecutionResult,
    ExecutionStatus,
    ExecutionStoreError,
    ExecutionSuspendedError,
    RetryableExecutionError,
)
from hayhooks.durable.runtime import (
    DefinitionRevisionConflictError,
    DurableDeployment,
    DurableRuntime,
    ExecutionStoreProvider,
    IdempotencyConflictError,
    durable_runtime,
)
from hayhooks.durable.settings import DurableSettings
from hayhooks.durable.store import ExecutionStore, InMemoryExecutionStoreProvider, RedisExecutionStoreProvider


def current_execution_id() -> str | None:
    """Return the active durable execution ID for hooks and idempotent tools."""
    context = get_current_durable_context()
    return context.execution_id if context is not None else None


current_durable_context = get_current_durable_context

__all__ = [
    "DefinitionRevisionConflictError",
    "DurableAuthoringMode",
    "DurableContext",
    "DurableDeployment",
    "DurableRuntime",
    "DurableSettings",
    "ExecutionAdmissionError",
    "ExecutionCanceledError",
    "ExecutionProgress",
    "ExecutionRecordSizeError",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionStore",
    "ExecutionStoreError",
    "ExecutionStoreProvider",
    "ExecutionSuspendedError",
    "IdempotencyConflictError",
    "InMemoryExecutionStoreProvider",
    "RedisExecutionStoreProvider",
    "RetryableExecutionError",
    "create_durable_router",
    "current_durable_context",
    "current_execution_id",
    "durable_authoring_mode",
    "durable_runtime",
]
