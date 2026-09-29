"""Portable durable execution primitives, loaded on first use so integrations stay optional."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hayhooks.durable.context import (
        DurableContext,
        DurableExecutionCancelledError,
        current_durable_context,
        durable_context_scope,
        durable_streaming_callback,
    )
    from hayhooks.durable.fastapi import OwnerIdDependency, create_durable_router
    from hayhooks.durable.runtime import DurableDeployment, DurableRuntime, RuntimeConfig
    from hayhooks.durable.store import ExecutionStore, MemoryExecutionStore, StoreConfig

__all__ = [
    "DurableContext",
    "DurableDeployment",
    "DurableExecutionCancelledError",
    "DurableRuntime",
    "ExecutionStore",
    "MemoryExecutionStore",
    "OwnerIdDependency",
    "RuntimeConfig",
    "StoreConfig",
    "create_durable_router",
    "current_durable_context",
    "durable_context_scope",
    "durable_streaming_callback",
]

_EXPORT_MODULES = {
    "DurableContext": "hayhooks.durable.context",
    "DurableDeployment": "hayhooks.durable.runtime",
    "DurableExecutionCancelledError": "hayhooks.durable.context",
    "DurableRuntime": "hayhooks.durable.runtime",
    "ExecutionStore": "hayhooks.durable.store",
    "MemoryExecutionStore": "hayhooks.durable.store",
    "OwnerIdDependency": "hayhooks.durable.fastapi",
    "RuntimeConfig": "hayhooks.durable.runtime",
    "StoreConfig": "hayhooks.durable.store",
    "create_durable_router": "hayhooks.durable.fastapi",
    "current_durable_context": "hayhooks.durable.context",
    "durable_context_scope": "hayhooks.durable.context",
    "durable_streaming_callback": "hayhooks.durable.context",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
