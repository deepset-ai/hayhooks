"""
Compatibility imports for the durable execution package.

New internal code should import the narrow module it needs from
``hayhooks.durable``.  This facade preserves the original advanced API for
applications and custom store implementations.
"""

from hayhooks.durable import models as _models
from hayhooks.durable.context import (
    DurableAdapter,
    DurableContext,
    DurableExecutionStore,
    DurableExecutionStoreProvider,
    ExecutionClaim,
    execution_context_scope,
    get_current_durable_context,
)
from hayhooks.durable.manager import DurableExecutionManager
from hayhooks.durable.memory import InMemoryExecutionStore, InMemoryExecutionStoreProvider
from hayhooks.durable.models import (
    DEFAULT_MAX_PROGRESS_EVENTS,
    DEFAULT_MAX_RECORD_BYTES,
    RECORD_SCHEMA_VERSION,
    ExecutionCanceledError,
    ExecutionCheckpoint,
    ExecutionError,
    ExecutionKind,
    ExecutionLeaseLostError,
    ExecutionProgressEvent,
    ExecutionRecord,
    ExecutionRecordSizeError,
    ExecutionRetiredError,
    ExecutionStatus,
    ExecutionStoreError,
    ExecutionSuspendedError,
    JsonValue,
    RetryableExecutionError,
    json_safe,
    utc_now,
    validate_json,
)

# Kept as module attributes for existing test and instrumentation patch points.
json = _models.json

__all__ = [
    "DEFAULT_MAX_PROGRESS_EVENTS",
    "DEFAULT_MAX_RECORD_BYTES",
    "RECORD_SCHEMA_VERSION",
    "DurableAdapter",
    "DurableContext",
    "DurableExecutionManager",
    "DurableExecutionStore",
    "DurableExecutionStoreProvider",
    "ExecutionCanceledError",
    "ExecutionCheckpoint",
    "ExecutionClaim",
    "ExecutionError",
    "ExecutionKind",
    "ExecutionLeaseLostError",
    "ExecutionProgressEvent",
    "ExecutionRecord",
    "ExecutionRecordSizeError",
    "ExecutionRetiredError",
    "ExecutionStatus",
    "ExecutionStoreError",
    "ExecutionSuspendedError",
    "InMemoryExecutionStore",
    "InMemoryExecutionStoreProvider",
    "JsonValue",
    "RetryableExecutionError",
    "execution_context_scope",
    "get_current_durable_context",
    "json_safe",
    "utc_now",
    "validate_json",
]
