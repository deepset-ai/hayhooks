from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext
    from a2a.server.tasks import TaskStore
    from a2a.types import Task

    from hayhooks.server.a2a.redis_task_store import RedisTaskStore, RedisTaskStoreProvider


class A2APipelineWrapper(BasePipelineWrapper):
    """Base class for wrappers that expose a managed durable A2A Agent."""

    durable: bool = True


def default_a2a_owner(context: ServerCallContext) -> str:
    """Return the built-in stable owner for an A2A request."""
    user = context.user
    if not user.is_authenticated:
        return "anonymous"
    if not user.user_name:
        msg = "Authenticated A2A users must have a non-empty user name"
        raise ValueError(msg)
    return f"user:{user.user_name}"


def validate_a2a_owner(owner_id: str) -> str:
    """Reject owner resolvers that cannot isolate persisted tasks."""
    if not isinstance(owner_id, str) or not owner_id:
        msg = "A2A owner resolvers must return a non-empty string"
        raise ValueError(msg)
    return owner_id


class TaskStoreProvider(ABC):
    """Create A2A SDK task stores for the agents mounted by the server."""

    @abstractmethod
    def create_task_store(self, agent_name: str) -> TaskStore:
        """Return the task store for an exposed agent."""
        raise NotImplementedError

    async def initialize(self) -> None:
        """Validate provider resources during A2A application startup."""
        return None

    async def health(self) -> dict[str, Any]:
        """Return a payload-safe readiness projection for provider resources."""
        return {"healthy": True, "provider": type(self).__name__}

    async def close(self) -> None:
        """Release resources owned by the provider when the A2A server stops."""
        return None


@runtime_checkable
class RecoverableTaskStore(Protocol):
    """Optional Redis operations used to recover durable A2A projections."""

    def owner_id_for_context(self, context: ServerCallContext) -> str: ...

    async def recoverable_task_batch(
        self, cursor: int, limit: int
    ) -> tuple[list[tuple[Task, str, int]], int | None]: ...

    async def save_projection(self, task: Task, owner: str, expected_version: int) -> bool:
        """Persist a projected task through its optimistic version fence."""
        ...


def __getattr__(name: str) -> Any:
    """Lazily expose optional Redis task-store types without importing A2A runtime."""
    if name in {"RedisTaskStore", "RedisTaskStoreProvider"}:
        from hayhooks.server.a2a.redis_task_store import RedisTaskStore, RedisTaskStoreProvider

        return {"RedisTaskStore": RedisTaskStore, "RedisTaskStoreProvider": RedisTaskStoreProvider}[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "A2APipelineWrapper",
    "RecoverableTaskStore",
    "RedisTaskStore",
    "RedisTaskStoreProvider",
    "TaskStoreProvider",
    "default_a2a_owner",
    "validate_a2a_owner",
]
