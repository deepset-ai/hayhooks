import asyncio
from contextlib import suppress
from typing import Any

from hayhooks.a2a import TaskStoreProvider
from hayhooks.durable.runtime import DurableRuntime
from hayhooks.server.a2a.durable_executor import DurableAgentExecutor
from hayhooks.server.a2a.imports import (
    InMemoryTaskStore,
    InvalidParamsError,
    RequestContext,
    RequestContextBuilder,
    SimpleRequestContextBuilder,
    TaskStore,
)
from hayhooks.server.logger import log
from hayhooks.settings import settings


class TaskAwareRequestContextBuilder(RequestContextBuilder):
    """Infer and validate context identity for messages that continue a task."""

    def __init__(self, task_store: "TaskStore") -> None:
        self._task_store = task_store
        self._delegate = SimpleRequestContextBuilder(
            should_populate_referred_tasks=False,
            task_store=task_store,
        )

    async def build(
        self,
        context: Any,
        params: Any | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Any | None = None,
    ) -> "RequestContext":
        """
        Build a request context while preserving the context of an existing task.

        A2A permits a follow-up message to provide only ``task_id``. In that
        case the server must infer ``context_id`` from the stored task. If the
        client provides both identifiers, they must refer to the same task.
        """
        existing_task = task
        if task_id is not None and existing_task is None:
            existing_task = await self._task_store.get(task_id, context)

        if existing_task is not None:
            if context_id is not None and context_id != existing_task.context_id:
                msg = (
                    f"Message context_id '{context_id}' does not match context_id "
                    f"'{existing_task.context_id}' for task '{task_id}'"
                )
                raise InvalidParamsError(message=msg)
            context_id = existing_task.context_id

        # Preserve the SDK's concurrency behavior: ActiveTask refreshes
        # current_task immediately before invoking the executor, so do not pass
        # the independently loaded copy into the request context here.
        return await self._delegate.build(
            context=context,
            params=params,
            task_id=task_id,
            context_id=context_id,
            task=task,
        )


class InMemoryTaskStoreProvider(TaskStoreProvider):
    """Provide an independent in-memory task store for each exposed agent."""

    def create_task_store(self, agent_name: str) -> "TaskStore":  # noqa: ARG002
        return InMemoryTaskStore()


def create_task_store_provider(
    *,
    backend: str = "auto",
    redis_url: str | None = None,
    redis_key_prefix: str | None = None,
    redis: Any | None = None,
    close_redis: bool = True,
) -> TaskStoreProvider:
    """Create a built-in A2A task-store provider."""
    if backend == "redis":
        from hayhooks.server.a2a.redis_task_store import RedisTaskStoreProvider

        return RedisTaskStoreProvider(
            redis_url=redis_url,
            key_prefix=redis_key_prefix,
            redis=redis,
            close_redis=close_redis,
        )
    if backend in {"auto", "memory"}:
        return InMemoryTaskStoreProvider()
    msg = f"Unsupported A2A task-store backend '{backend}'; expected 'auto', 'memory', or 'redis'"
    raise ValueError(msg)


class A2ARuntime:
    """Owns A2A server resources shared by mounted agents."""

    def __init__(
        self,
        task_store_provider: TaskStoreProvider | None = None,
        *,
        durable_runtime: DurableRuntime | None = None,
    ) -> None:
        self.task_store_provider = task_store_provider or InMemoryTaskStoreProvider()
        self._executors: list[DurableAgentExecutor] = []
        self._started_executors: list[DurableAgentExecutor] = []
        self._task_stores: list[TaskStore] = []
        self._maintenance_task: asyncio.Task[None] | None = None
        self._started = False
        self.durable_runtime = durable_runtime

    def register_agent_executor(self, executor: Any) -> None:
        if isinstance(executor, DurableAgentExecutor):
            self._executors.append(executor)

    async def start(self) -> None:
        """Start lifecycle-aware executors after the application event loop is available."""
        try:
            await self.task_store_provider.initialize()
            for executor in self._executors:
                self._started_executors.append(executor)
                await executor.start()
            if any(callable(getattr(store, "cleanup_expired_tasks", None)) for store in self._task_stores):
                self._maintenance_task = asyncio.create_task(
                    self._maintain_task_stores(),
                    name="a2a-task-store-maintenance",
                )
            self._started = True
        except BaseException:
            self._started = False
            await self._close_executors()
            raise

    def create_task_store(self, agent_name: str) -> "TaskStore":
        task_store = self.task_store_provider.create_task_store(agent_name)
        if not isinstance(task_store, TaskStore):
            msg = (
                f"Task store provider {type(self.task_store_provider).__name__} returned "
                f"{type(task_store).__name__} for agent '{agent_name}'; expected a2a.server.tasks.TaskStore"
            )
            raise TypeError(msg)
        if task_store not in self._task_stores:
            self._task_stores.append(task_store)
        return task_store

    async def close(self) -> None:
        """Stop executor work before releasing shared task-store resources."""
        self._started = False
        try:
            if self._maintenance_task is not None:
                self._maintenance_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._maintenance_task
                self._maintenance_task = None
            await self._close_executors()
        finally:
            await self.task_store_provider.close()

    async def health(self) -> dict[str, Any]:
        """Report operational readiness without changing the A2A protocol surface."""
        provider = await self._provider_health()
        executor_health = {
            f"{type(executor).__name__}:{index}": executor.health() for index, executor in enumerate(self._executors)
        }
        maintenance = self._maintenance_health()
        components: dict[str, Any] = {
            "task_store": provider,
            "executors": executor_health,
            "maintenance": maintenance,
        }
        if self._executors:
            components["durable_execution"] = await self._durable_health()

        healthy = self._started and bool(provider.get("healthy", False)) and bool(maintenance["healthy"])
        healthy = healthy and all(bool(value.get("healthy", False)) for value in executor_health.values())
        durable = components.get("durable_execution")
        if isinstance(durable, dict):
            healthy = healthy and bool(durable.get("healthy", False))
        return {
            "healthy": healthy,
            "started": self._started,
            "components": components,
        }

    async def _provider_health(self) -> dict[str, Any]:
        try:
            value = await self.task_store_provider.health()
            if isinstance(value, dict):
                return value
            return {
                "healthy": False,
                "provider": type(self.task_store_provider).__name__,
                "error": "InvalidHealthPayload",
            }
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return {
                "healthy": False,
                "provider": type(self.task_store_provider).__name__,
                "error": type(error).__name__,
            }

    async def _durable_health(self) -> dict[str, Any]:
        try:
            if self.durable_runtime is None:
                return {"healthy": False, "error": "DurableRuntimeUnavailable"}
            return await self.durable_runtime.health()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return {"healthy": False, "error": type(error).__name__}

    def _maintenance_health(self) -> dict[str, Any]:
        task = self._maintenance_task
        health: dict[str, Any] = {
            "healthy": task is None or not task.done(),
            "enabled": task is not None,
        }
        if task is None or not task.done():
            return health
        if task.cancelled():
            health["error"] = "CancelledError"
        elif error := task.exception():
            health["error"] = type(error).__name__
        return health

    async def _maintain_task_stores(self) -> None:
        """Expire terminal tasks even when no later A2A request arrives."""
        intervals = [
            max(1.0, min(60.0, float(getattr(store, "terminal_ttl_seconds", 60)) / 10))
            for store in self._task_stores
            if callable(getattr(store, "cleanup_expired_tasks", None))
        ]
        interval = min(intervals, default=60.0)
        while True:
            await asyncio.sleep(interval)
            for store in self._task_stores:
                cleanup = getattr(store, "cleanup_expired_tasks", None)
                if not callable(cleanup):
                    continue
                try:
                    await cleanup(limit=settings.a2a_list_scan_batch_size)
                except Exception as error:
                    log.opt(exception=error).warning("A2A terminal-task cleanup failed: {}", error)

    async def _close_executors(self) -> None:
        for executor in reversed(self._started_executors):
            try:
                await executor.close()
            except Exception as error:
                log.opt(exception=True).warning(
                    "Error closing A2A executor lifecycle '{}': {}",
                    type(executor).__name__,
                    error,
                )
        self._started_executors.clear()
