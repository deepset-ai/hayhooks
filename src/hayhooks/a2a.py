from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

if TYPE_CHECKING:
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.tasks import TaskStore
    from a2a.types import Task

    from hayhooks.server.a2a.redis_task_store import RedisTaskStore, RedisTaskStoreProvider


class A2APipelineWrapper(BasePipelineWrapper):
    """Base class for wrappers that provide A2A or durable Agent behavior."""

    # Set to True for the built-in durable Haystack 3 Agent mapping. A native
    # create_a2a_agent_executor remains the advanced escape hatch.
    durable: bool = False

    def create_a2a_agent_executor(self) -> AgentExecutor:
        """Create the SDK executor used for this deployed agent."""
        raise NotImplementedError


class TaskStoreProvider(ABC):
    """Create A2A SDK task stores for the agents mounted by the server."""

    @abstractmethod
    def create_task_store(self, agent_name: str) -> TaskStore:
        """Return the task store for an exposed agent."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release resources owned by the provider when the A2A server stops."""
        return None


@runtime_checkable
class RecoverableTaskStore(Protocol):
    """Optional task-store extension used for durable execution recovery."""

    async def get_for_execution(self, task_id: str) -> Task | None:
        """Load a task without a request context after process restart."""
        ...

    async def save_for_execution(self, task: Task) -> None:
        """Persist a task projection without a request context."""
        ...

    async def recoverable_tasks(self) -> list[Task]:
        """Return nonterminal tasks whose durable projection may need recovery."""
        ...


@runtime_checkable
class LifecycleAgentExecutor(Protocol):
    """Optional lifecycle implemented by native executors that own background work."""

    async def start(self) -> None:
        """Start executor-owned resources after the A2A app event loop is running."""
        ...

    async def close(self) -> None:
        """Stop background work before shared A2A runtime resources are closed."""
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
    "LifecycleAgentExecutor",
    "RecoverableTaskStore",
    "RedisTaskStore",
    "RedisTaskStoreProvider",
    "TaskStoreProvider",
]
