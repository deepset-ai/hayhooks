import asyncio
import importlib
from contextlib import suppress
from typing import Any

from hayhooks.a2a import LifecycleAgentExecutor, TaskStoreProvider
from hayhooks.server.a2a.imports import (
    InMemoryTaskStore,
    InvalidParamsError,
    RequestContext,
    RequestContextBuilder,
    SimpleRequestContextBuilder,
    TaskStore,
    a2a_import,
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
        a2a_import.check()
        return InMemoryTaskStore()


def create_task_store_provider(  # noqa: PLR0913 - public backend factory
    *,
    backend: str = "auto",
    custom_provider: str = "",
    redis_url: str | None = None,
    redis_key_prefix: str | None = None,
    redis: Any | None = None,
    close_redis: bool = True,
) -> TaskStoreProvider:
    """Create the configured built-in or custom A2A task-store provider."""
    if custom_provider:
        if backend not in {"auto", "memory"}:
            msg = "--task-store and --task-store-provider cannot be used together"
            raise ValueError(msg)
        return load_task_store_provider(custom_provider)
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
    msg = f"Unsupported A2A task-store backend '{backend}'; expected 'memory' or 'redis'"
    raise ValueError(msg)


def load_task_store_provider(import_path: str) -> TaskStoreProvider:
    """Load and instantiate a task store provider class from ``module:ClassName``."""
    module_name, separator, class_name = import_path.partition(":")
    if not separator or not module_name or not class_name:
        msg = f"A2A task store provider must use the format 'module:ClassName'; received '{import_path}'"
        raise ValueError(msg)

    module = importlib.import_module(module_name)
    try:
        provider_class = getattr(module, class_name)
    except AttributeError as error:
        msg = f"A2A task store provider class '{class_name}' was not found in module '{module_name}'"
        raise ValueError(msg) from error

    if not isinstance(provider_class, type) or not issubclass(provider_class, TaskStoreProvider):
        msg = f"A2A task store provider '{import_path}' must be a TaskStoreProvider subclass"
        raise TypeError(msg)

    return provider_class()


class A2ARuntime:
    """Owns A2A server resources shared by mounted agents."""

    def __init__(
        self,
        task_store_provider: TaskStoreProvider | None = None,
    ) -> None:
        self.task_store_provider = task_store_provider or InMemoryTaskStoreProvider()
        self._executor_lifecycles: list[LifecycleAgentExecutor] = []
        self._started_executor_lifecycles: list[LifecycleAgentExecutor] = []
        self._task_stores: list[TaskStore] = []
        self._maintenance_task: asyncio.Task[None] | None = None

    def register_agent_executor(self, executor: Any) -> None:
        """Register the optional lifecycle exposed by a native agent executor."""
        if isinstance(executor, LifecycleAgentExecutor):
            self._executor_lifecycles.append(executor)

    async def start(self) -> None:
        """Start lifecycle-aware executors after the application event loop is available."""
        try:
            for executor in self._executor_lifecycles:
                self._started_executor_lifecycles.append(executor)
                await executor.start()
            if any(callable(getattr(store, "cleanup_expired_tasks", None)) for store in self._task_stores):
                self._maintenance_task = asyncio.create_task(
                    self._maintain_task_stores(),
                    name="a2a-task-store-maintenance",
                )
        except BaseException:
            await self._close_executors()
            raise

    def create_task_store(self, agent_name: str) -> "TaskStore":
        a2a_import.check()
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

    def create_request_context_builder(self, task_store: "TaskStore") -> TaskAwareRequestContextBuilder:
        return TaskAwareRequestContextBuilder(task_store)

    async def close(self) -> None:
        """Stop executor work before releasing shared task-store resources."""
        try:
            if self._maintenance_task is not None:
                self._maintenance_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._maintenance_task
                self._maintenance_task = None
            await self._close_executors()
        finally:
            await self.task_store_provider.close()

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
                    await cleanup(limit=settings.a2a_projection_batch_size)
                except Exception as error:
                    log.opt(exception=error).warning("A2A terminal-task cleanup failed: {}", error)

    async def _close_executors(self) -> None:
        for executor in reversed(self._started_executor_lifecycles):
            try:
                await executor.close()
            except Exception as error:
                log.opt(exception=True).warning(
                    "Error closing A2A executor lifecycle '{}': {}",
                    type(executor).__name__,
                    error,
                )
        self._started_executor_lifecycles.clear()

    @property
    def push_notifications_enabled(self) -> bool:
        return False
