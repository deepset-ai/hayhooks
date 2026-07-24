"""A2A's persisted projection of managed durable-Agent executions."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

from hayhooks.a2a import RecoverableTaskStore
from hayhooks.durable.models import ExecutionStatus
from hayhooks.durable.runtime import DurableDeployment, durable_runtime
from hayhooks.server.a2a.imports import (
    AgentExecutor,
    EventQueue,
    RequestContext,
    TaskUpdater,
    new_task_from_user_message,
    new_text_part,
)
from hayhooks.server.a2a.messages import build_haystack_messages, build_haystack_resume_messages
from hayhooks.server.logger import log
from hayhooks.server.tracing import SPAN_A2A_DURABLE_PROJECT, build_trace_tags, trace_operation
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings

DURABLE_PROGRESS_ARTIFACT_NAME = "durable-progress"
DURABLE_RESULT_ARTIFACT_NAME = "durable-result"


class _ProjectionConflictError(RuntimeError):
    """A projection event was based on a stale version or lost lease."""


_RECORD_NOT_LOADED = object()


class _RecoveryEventQueue:
    """Apply watcher events directly to a persisted task when no request queue exists."""

    def __init__(
        self,
        task_store: RecoverableTaskStore,
        task: Any,
        *,
        lease_token: str | None = None,
        expected_version: int = 0,
    ) -> None:
        self.task_store = task_store
        self.task = task
        self.lease_token = lease_token
        self.expected_version = expected_version

    async def enqueue_event(self, event: Any) -> None:
        from a2a.server.tasks.task_manager import append_artifact_to_task
        from a2a.types import TaskArtifactUpdateEvent, TaskStatusUpdateEvent

        previous = type(self.task)()
        previous.CopyFrom(self.task)
        try:
            if isinstance(event, TaskStatusUpdateEvent):
                if self.task.status.HasField("message"):
                    self.task.history.append(self.task.status.message)
                if event.metadata:
                    self.task.metadata.MergeFrom(event.metadata)
                self.task.status.CopyFrom(event.status)
            elif isinstance(event, TaskArtifactUpdateEvent):
                append_artifact_to_task(self.task, event)
            save_projection = getattr(self.task_store, "save_projection", None)
            if self.lease_token is not None and callable(save_projection):
                version = await save_projection(self.task, self.lease_token, self.expected_version)
                if version < 0:
                    msg = f"Projection lease or version was lost for task '{self.task.id}'"
                    raise _ProjectionConflictError(msg)
                self.expected_version = version
            else:
                await self.task_store.save_for_execution(self.task)
        except BaseException:
            # TaskUpdater can retry the event only if the local task is put
            # back at the exact snapshot that preceded the rejected write.
            self.task.CopyFrom(previous)
            raise

@dataclass
class _ProjectionCursor:
    updater: TaskUpdater
    task: Any | None = None
    last_sequence: int = 0
    record_sequence: int = -1
    lease_token: str | None = None
    lease_renew_at: float = 0.0
    failures: int = 0
    retry_at: float = 0.0


class DurableAgentExecutor(AgentExecutor):
    """Project a built-in durable Haystack Agent execution onto its A2A task."""

    def __init__(
        self,
        wrapper: BasePipelineWrapper,
        pipeline_name: str,
        task_store: Any,
        *,
        deployment: DurableDeployment | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.pipeline_name = pipeline_name
        self.task_store = task_store
        self._deployment = deployment
        # Futures keep blocking/streaming SDK requests open; one bounded
        # reconciler owns all polling rather than one coroutine per task.
        self._watchers: dict[str, asyncio.Future[None]] = {}
        self._projections: dict[str, _ProjectionCursor] = {}
        self._reconciler: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._closed = False
        self._recovery_cursor: int | None = 0
        self._next_recovery_scan_at = 0.0

    async def start(self) -> None:
        """Recover projections for persisted tasks whose executions outlived the process."""
        self._closed = False
        if not isinstance(self.task_store, RecoverableTaskStore):
            self._ensure_reconciler()
            return
        batch = getattr(self.task_store, "recoverable_task_batch", None)
        if callable(batch):
            await self._discover_recovery_batch()
            self._ensure_reconciler()
            return
        deployment = self._durable_deployment()
        async for task in self._recovery_tasks():
            try:
                await deployment.get(task.id)
            except (KeyError, RuntimeError):
                continue
            recovery_queue = cast(EventQueue, _RecoveryEventQueue(self.task_store, task))
            updater = TaskUpdater(recovery_queue, task.id, task.context_id)
            projection = _ProjectionCursor(updater=updater, task=task)
            if await self._own_projection(task.id, projection):
                self._projections[task.id] = projection
        self._recovery_cursor = 0
        self._next_recovery_scan_at = time.monotonic() + settings.a2a_projection_lease_ms / 1_000
        self._ensure_reconciler()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = await self._task(context, event_queue)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        deployment = self._durable_deployment()
        record = None
        if context.current_task is not None:
            # An existing A2A task may predate durable execution or be restored
            # without its durable record; treat it as a new durable submission.
            with suppress(KeyError):
                record = await deployment.get(task.id)
        if record is not None and record.status is ExecutionStatus.WAITING:
            await deployment.resume(
                task.id,
                {"messages": [message.to_dict() for message in build_haystack_resume_messages(context)]},
            )
            await updater.start_work()
        elif record is None:
            await deployment.submit_agent_messages(build_haystack_messages(context), execution_id=task.id)
            await updater.start_work()
        else:
            # Repeated A2A delivery is idempotent and must not create a second Agent invocation.
            await updater.start_work()
        # Keep the SDK's active-task dispatcher alive for this execution. A
        # return-immediately request stops waiting for events independently,
        # while blocking and streaming requests continue consuming this queue.
        await self._ensure_watcher(task.id, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            return
        deployment = self._durable_deployment()
        accepted = await deployment.request_cancel(task.id)
        if not accepted:
            return
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.add_artifact(
            [new_text_part("Cancellation requested")],
            artifact_id=f"{task.id}-{DURABLE_PROGRESS_ARTIFACT_NAME}",
            name=DURABLE_PROGRESS_ARTIFACT_NAME,
            append=False,
        )
        self._register_projection(task.id, updater, task=task)
        projection = self._projections[task.id]
        await self._reconcile_one(task.id, projection)

    async def close(self) -> None:
        self._closed = True
        if self._reconciler is not None:
            self._reconciler.cancel()
            await asyncio.gather(self._reconciler, return_exceptions=True)
            self._reconciler = None
        await asyncio.gather(
            *(
                self._release_projection(execution_id, projection)
                for execution_id, projection in self._projections.items()
            ),
            return_exceptions=True,
        )
        for watcher in self._watchers.values():
            if not watcher.done():
                watcher.cancel()
        self._watchers.clear()
        self._projections.clear()

    def _ensure_watcher(self, execution_id: str, updater: TaskUpdater) -> asyncio.Future[None]:
        existing = self._watchers.get(execution_id)
        if existing is not None and not existing.done():
            return existing
        watcher = asyncio.get_running_loop().create_future()
        self._watchers[execution_id] = watcher
        self._register_projection(execution_id, updater)
        return watcher

    def _discard_watcher(self, execution_id: str, completed: asyncio.Future[None]) -> None:
        if self._watchers.get(execution_id) is completed:
            self._watchers.pop(execution_id, None)

    def _register_projection(self, execution_id: str, updater: TaskUpdater, *, task: Any | None = None) -> None:
        projection = self._projections.get(execution_id)
        if projection is None:
            self._projections[execution_id] = _ProjectionCursor(updater=updater, task=task)
        else:
            projection.updater = updater
            projection.task = task or projection.task
        self._ensure_reconciler()
        self._wake.set()

    def _ensure_reconciler(self) -> None:
        if self._closed or (self._reconciler is not None and not self._reconciler.done()):
            return
        self._reconciler = asyncio.create_task(
            self._reconcile_loop(),
            name=f"a2a-reconciler:{self.pipeline_name}",
        )

    async def _task(self, context: RequestContext, event_queue: EventQueue) -> Any:
        if context.current_task is not None:
            return context.current_task
        if context.message is None:
            msg = "A2A request has neither a current task nor a message"
            raise ValueError(msg)
        task = new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        return task

    async def _reconcile_loop(self) -> None:
        while not self._closed:
            self._wake.clear()
            now = time.monotonic()
            if now >= self._next_recovery_scan_at:
                try:
                    await self._discover_recovery_batch()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._next_recovery_scan_at = time.monotonic() + min(
                        settings.a2a_projection_interval * 2,
                        5.0,
                    )
                    log.opt(exception=error).warning(
                        "A2A durable recovery discovery failed for '{}': {}",
                        self.pipeline_name,
                        error,
                    )
                now = time.monotonic()
            due = [
                (execution_id, projection)
                for execution_id, projection in list(self._projections.items())
                if projection.retry_at <= now
            ]
            batch = due[: settings.a2a_projection_batch_size]
            await self._reconcile_batch(batch)
            if len(due) > len(batch):
                await asyncio.sleep(0)
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=settings.a2a_projection_interval)

    async def _reconcile_batch(self, batch: list[tuple[str, _ProjectionCursor]]) -> None:
        """Renew ownership, then fetch only records changed since the last projection."""
        owned = await self._owned_projections(batch)
        if not owned:
            return

        deployment = self._durable_deployment()
        try:
            records = await deployment.get_changed(
                {execution_id: projection.record_sequence for execution_id, projection in owned}
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            for execution_id, projection in owned:
                self._backoff_projection(execution_id, projection, error)
            return

        for execution_id, projection in owned:
            record = records.get(execution_id, _RECORD_NOT_LOADED)
            if record is _RECORD_NOT_LOADED:
                projection.failures = 0
                projection.retry_at = time.monotonic() + settings.a2a_projection_interval
                continue
            try:
                await self._reconcile_one(execution_id, projection, record=record, owned=True)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._backoff_projection(execution_id, projection, error)

    async def _owned_projections(
        self,
        batch: list[tuple[str, _ProjectionCursor]],
    ) -> list[tuple[str, _ProjectionCursor]]:
        owned: list[tuple[str, _ProjectionCursor]] = []
        for execution_id, projection in batch:
            try:
                if await self._own_projection(execution_id, projection):
                    owned.append((execution_id, projection))
                else:
                    projection.retry_at = time.monotonic() + settings.a2a_projection_interval
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._backoff_projection(execution_id, projection, error)
        return owned

    async def _reconcile_one(  # noqa: C901, PLR0912 - explicit projection state table
        self,
        execution_id: str,
        projection: _ProjectionCursor,
        *,
        record: Any = _RECORD_NOT_LOADED,
        owned: bool = False,
    ) -> None:
        if not owned and not await self._own_projection(execution_id, projection):
            projection.retry_at = time.monotonic() + settings.a2a_projection_interval
            return
        deployment = self._durable_deployment()
        try:
            with trace_operation(
                SPAN_A2A_DURABLE_PROJECT,
                tags=build_trace_tags(
                    {
                        "hayhooks.pipeline.name": self.pipeline_name,
                        "hayhooks.durable.execution_id": execution_id,
                    }
                ),
            ) as span:
                if record is _RECORD_NOT_LOADED:
                    record = await deployment.get(execution_id)
                elif record is None:
                    raise KeyError(execution_id)
                span.set_tag("hayhooks.a2a.execution_status", record.status.value)
                updated_at = getattr(record, "updated_at", None)
                if updated_at is not None:
                    span.set_tag(
                        "hayhooks.a2a.projection_lag_ms",
                        max(0, int((time.time() - updated_at.timestamp()) * 1_000)),
                    )
                updater = projection.updater
                if record.progress and record.progress[-1].sequence > projection.last_sequence:
                    await updater.add_artifact(
                        [new_text_part("\n".join(event.message for event in record.progress))],
                        artifact_id=f"{execution_id}-{DURABLE_PROGRESS_ARTIFACT_NAME}",
                        name=DURABLE_PROGRESS_ARTIFACT_NAME,
                        append=False,
                    )
                    projection.last_sequence = record.progress[-1].sequence
                terminal_projection = True
                if record.status is ExecutionStatus.WAITING:
                    await updater.requires_input(
                        message=updater.new_agent_message(
                            [new_text_part("The durable Agent requires input to continue.")]
                        )
                    )
                elif record.status is ExecutionStatus.COMPLETED:
                    await updater.add_artifact(
                        [new_text_part(_result_text(record.result))],
                        artifact_id=f"{execution_id}-{DURABLE_RESULT_ARTIFACT_NAME}",
                        name=DURABLE_RESULT_ARTIFACT_NAME,
                        append=False,
                        last_chunk=True,
                    )
                    await updater.complete()
                elif record.status is ExecutionStatus.FAILED:
                    text = record.error.message if record.error else "Durable Agent execution failed"
                    await updater.failed(message=updater.new_agent_message([new_text_part(text)]))
                elif record.status is ExecutionStatus.CANCELED:
                    await updater.cancel(
                        message=updater.new_agent_message([new_text_part("The durable Agent execution was canceled.")])
                    )
                else:
                    terminal_projection = False
            projection.failures = 0
            projection.record_sequence = record.sequence
            if terminal_projection:
                await self._finish_projection(execution_id, projection)
            else:
                projection.retry_at = time.monotonic() + settings.a2a_projection_interval
        except asyncio.CancelledError:
            raise
        except _ProjectionConflictError as error:
            await self._refresh_projection(execution_id, projection)
            self._backoff_projection(execution_id, projection, error)
        except Exception as error:
            self._backoff_projection(execution_id, projection, error)

    def _backoff_projection(
        self,
        execution_id: str,
        projection: _ProjectionCursor,
        error: BaseException,
    ) -> None:
        projection.failures += 1
        exponent = min(projection.failures - 1, 10)
        delay = min(settings.a2a_projection_interval * (2**exponent), 5.0)
        projection.retry_at = time.monotonic() + delay
        log.opt(exception=error).warning(
            "A2A durable projection failed for '{}'; retrying in {:.2f}s: {}",
            execution_id,
            delay,
            error,
        )

    async def _refresh_projection(self, execution_id: str, projection: _ProjectionCursor) -> None:
        await self._release_projection(execution_id, projection)
        load = getattr(self.task_store, "get_for_execution", None)
        if not callable(load):
            return
        task = await load(execution_id)
        if task is None:
            await self._finish_projection(execution_id, projection)
            return
        projection.task = task
        projection.updater = TaskUpdater(
            cast(EventQueue, _RecoveryEventQueue(self.task_store, task)),
            task.id,
            task.context_id,
        )

    async def _own_projection(self, execution_id: str, projection: _ProjectionCursor) -> bool:
        acquire = getattr(self.task_store, "acquire_projection", None)
        renew = getattr(self.task_store, "renew_projection", None)
        if not callable(acquire) or not callable(renew):
            return True
        now = time.monotonic()
        if projection.lease_token is None:
            token = await acquire(execution_id, lease_ms=settings.a2a_projection_lease_ms)
            if token is None:
                return False
            projection.lease_token = token
            projection.lease_renew_at = now + settings.a2a_projection_lease_ms / 3_000
            version_method = getattr(self.task_store, "projection_version", None)
            if (
                projection.task is not None
                and isinstance(self.task_store, RecoverableTaskStore)
                and callable(version_method)
            ):
                version = await version_method(execution_id)
                projection.updater = TaskUpdater(
                    cast(
                        EventQueue,
                        _RecoveryEventQueue(
                            self.task_store,
                            projection.task,
                            lease_token=token,
                            expected_version=version,
                        ),
                    ),
                    projection.task.id,
                    projection.task.context_id,
                )
            return True
        if now < projection.lease_renew_at:
            return True
        if not await renew(
            execution_id,
            projection.lease_token,
            lease_ms=settings.a2a_projection_lease_ms,
        ):
            projection.lease_token = None
            return False
        projection.lease_renew_at = now + settings.a2a_projection_lease_ms / 3_000
        return True

    async def _finish_projection(self, execution_id: str, projection: _ProjectionCursor) -> None:
        self._projections.pop(execution_id, None)
        await self._release_projection(execution_id, projection)
        watcher = self._watchers.pop(execution_id, None)
        if watcher is not None and not watcher.done():
            watcher.set_result(None)

    async def _release_projection(self, execution_id: str, projection: _ProjectionCursor) -> None:
        release = getattr(self.task_store, "release_projection", None)
        if projection.lease_token is not None and callable(release):
            with suppress(Exception):
                await release(execution_id, projection.lease_token)
        projection.lease_token = None

    async def _recovery_tasks(self):
        batch = getattr(self.task_store, "recoverable_task_batch", None)
        if callable(batch):
            offset: int | None = 0
            while offset is not None:
                tasks, offset = await batch(offset, settings.a2a_projection_batch_size)
                for task in tasks:
                    yield task
            return
        for task in await self.task_store.recoverable_tasks():
            yield task

    async def _discover_recovery_batch(self) -> None:
        """Periodically compete for abandoned tasks without polling non-owned records."""
        if not isinstance(self.task_store, RecoverableTaskStore):
            self._next_recovery_scan_at = float("inf")
            return
        batch = getattr(self.task_store, "recoverable_task_batch", None)
        if not callable(batch):
            self._next_recovery_scan_at = time.monotonic() + settings.a2a_projection_lease_ms / 1_000
            return
        cursor = self._recovery_cursor or 0
        tasks, next_cursor = await batch(cursor, settings.a2a_projection_batch_size)
        deployment = self._durable_deployment()
        candidates = [task for task in tasks if task.id not in self._projections]
        records = await deployment.get_changed({task.id: -1 for task in candidates})
        for task in candidates:
            if records.get(task.id) is None:
                continue
            projection = _ProjectionCursor(
                updater=TaskUpdater(
                    cast(EventQueue, _RecoveryEventQueue(self.task_store, task)),
                    task.id,
                    task.context_id,
                ),
                task=task,
            )
            if await self._own_projection(task.id, projection):
                self._projections[task.id] = projection
        self._recovery_cursor = next_cursor if next_cursor is not None else 0
        self._next_recovery_scan_at = (
            time.monotonic() if next_cursor is not None else time.monotonic() + settings.a2a_projection_lease_ms / 1_000
        )

    def _durable_deployment(self) -> DurableDeployment:
        """Use the injected deployment; retain fallback compatibility for direct construction."""
        if self._deployment is None:
            self._deployment = durable_runtime.deployment(self.pipeline_name, self.wrapper)
        return self._deployment

def _result_text(result: Any) -> str:
    if isinstance(result, dict):
        last = result.get("last_message")
        if isinstance(last, dict):
            content = last.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                if text := "".join(text_parts):
                    return text
        return json_string(result)
    return str(result or "")


def json_string(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)
