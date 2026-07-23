"""A2A executors, including the projection of the shared durable core."""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from haystack.dataclasses import StreamingChunk

from hayhooks.a2a import RecoverableTaskStore
from hayhooks.durable_runtime import durable_runtime
from hayhooks.execution import ExecutionStatus
from hayhooks.server.a2a.imports import (
    AgentExecutor,
    EventQueue,
    RequestContext,
    Role,
    TaskUpdater,
    a2a_import,
    get_message_text,
    new_task_from_user_message,
    new_text_part,
)
from hayhooks.server.logger import log
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.tracing import SPAN_A2A_DURABLE_PROJECT, SPAN_A2A_RUN_AGENT, build_trace_tags, trace_operation
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings

RESPONSE_ARTIFACT_NAME = "response"
DURABLE_PROGRESS_ARTIFACT_NAME = "durable-progress"
DURABLE_RESULT_ARTIFACT_NAME = "durable-result"


class _ProjectionConflictError(RuntimeError):
    """A projection event was based on a stale version or lost lease."""


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


def _build_openai_messages(context: RequestContext) -> list[dict[str, str]]:
    """Map A2A history plus the current message to OpenAI-compatible messages."""
    messages: list[dict[str, str]] = []
    history = list(context.current_task.history) if context.current_task else []
    history_ids = {message.message_id for message in history}
    for message in history:
        text = get_message_text(message)
        if text:
            messages.append({"role": "assistant" if message.role == Role.ROLE_AGENT else "user", "content": text})
    if context.message is not None and context.message.message_id not in history_ids:
        text = get_message_text(context.message)
        if text:
            messages.append({"role": "user", "content": text})
    return messages


def _build_haystack_messages(context: RequestContext) -> list[Any]:
    from haystack.dataclasses import ChatMessage

    converted: list[Any] = []
    for message in _build_openai_messages(context):
        if message["role"] == "assistant":
            converted.append(ChatMessage.from_assistant(message["content"]))
        else:
            converted.append(ChatMessage.from_user(message["content"]))
    return converted


def _build_haystack_resume_messages(context: RequestContext) -> list[Any]:
    """Convert only the follow-up turn; recovered Agent state already contains history."""
    from haystack.dataclasses import ChatMessage

    if context.message is None:
        return []
    text = get_message_text(context.message)
    if not text:
        return []
    if context.message.role == Role.ROLE_AGENT:
        return [ChatMessage.from_assistant(text)]
    return [ChatMessage.from_user(text)]


def _stream_item_to_text(item: Any) -> str | None:
    if isinstance(item, StreamingChunk):
        return item.content or None
    if isinstance(item, str):
        return item or None
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace") or None
    return None


async def _iter_text_chunks(result: Any) -> AsyncGenerator[str, None]:
    if isinstance(result, str):
        yield result
    elif isinstance(result, AsyncIterator):
        async for item in result:
            if text := _stream_item_to_text(item):
                yield text
    elif isinstance(result, Iterator):
        async for item in iterate_in_threadpool(result):
            if text := _stream_item_to_text(item):
                yield text
    else:
        msg = f"run_chat_completion returned unsupported type '{type(result).__name__}'"
        raise ValueError(msg)


async def _stream_result_as_artifact(result: Any, updater: TaskUpdater) -> None:
    artifact_id = str(uuid.uuid4())
    first = True

    async def emit(text: str, *, last: bool) -> None:
        nonlocal first
        await updater.add_artifact(
            [new_text_part(text)],
            artifact_id=artifact_id,
            name=RESPONSE_ARTIFACT_NAME,
            append=not first,
            last_chunk=last,
        )
        first = False

    if isinstance(result, str):
        await emit(result, last=True)
        return
    async for text in _iter_text_chunks(result):
        await emit(text, last=False)
    await emit("", last=True)


class ChatCompletionAgentExecutor(AgentExecutor):
    """Run a deployed wrapper's existing chat-completion capability through A2A."""

    def __init__(self, pipeline_name: str, pipeline_wrapper: BasePipelineWrapper | None) -> None:
        self.pipeline_name = pipeline_name
        self.pipeline_wrapper = pipeline_wrapper

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self._execute_agent_task(context, event_queue)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is not None:
            await TaskUpdater(event_queue, context.current_task.id, context.current_task.context_id).cancel()

    async def _run_chat_completion(self, context: RequestContext) -> Any:
        wrapper = self.pipeline_wrapper or registry.get(self.pipeline_name)
        if wrapper is None:
            msg = f"Pipeline '{self.pipeline_name}' not found"
            raise ValueError(msg)
        messages = _build_openai_messages(context)
        if wrapper._is_run_chat_completion_async_implemented:
            return await wrapper.run_chat_completion_async(model=self.pipeline_name, messages=messages, body={})
        return await run_in_threadpool(
            wrapper.run_chat_completion, model=self.pipeline_name, messages=messages, body={}
        )

    async def _execute_agent_task(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is not None:
            task = context.current_task
        elif context.message is not None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        else:
            msg = "A2A request has neither a current task nor a message"
            raise ValueError(msg)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        with trace_operation(
            SPAN_A2A_RUN_AGENT,
            tags=build_trace_tags({"hayhooks.transport": "a2a", "hayhooks.pipeline.name": self.pipeline_name}),
        ):
            try:
                await _stream_result_as_artifact(await self._run_chat_completion(context), updater)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                message = f"Error running pipeline '{self.pipeline_name}' as A2A agent: {error}"
                if settings.show_tracebacks:
                    message += f"\n{traceback.format_exc()}"
                log.opt(exception=True).error(message)
                await updater.failed(message=updater.new_agent_message([new_text_part(message)]))
                return
        await updater.complete()


@dataclass
class _Projection:
    updater: TaskUpdater
    task: Any | None = None
    last_sequence: int = 0
    lease_token: str | None = None
    lease_renew_at: float = 0.0
    failures: int = 0
    retry_at: float = 0.0


class DurableAgentExecutor(AgentExecutor):
    """Project a built-in durable Haystack Agent execution onto its A2A task."""

    def __init__(self, wrapper: BasePipelineWrapper, pipeline_name: str, task_store: Any) -> None:
        self.wrapper = wrapper
        self.pipeline_name = pipeline_name
        self.task_store = task_store
        # Futures keep blocking/streaming SDK requests open; one bounded
        # reconciler owns all polling rather than one coroutine per task.
        self._watchers: dict[str, asyncio.Future[None]] = {}
        self._projections: dict[str, _Projection] = {}
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
        deployment = durable_runtime.deployment(self.pipeline_name, self.wrapper)
        async for task in self._recovery_tasks():
            try:
                await deployment.get(task.id)
            except (KeyError, RuntimeError):
                continue
            recovery_queue = cast(EventQueue, _RecoveryEventQueue(self.task_store, task))
            updater = TaskUpdater(recovery_queue, task.id, task.context_id)
            projection = _Projection(updater=updater, task=task)
            if await self._own_projection(task.id, projection):
                self._projections[task.id] = projection
        self._recovery_cursor = 0
        self._next_recovery_scan_at = time.monotonic() + settings.a2a_projection_lease_ms / 1_000
        self._ensure_reconciler()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = await self._task(context, event_queue)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        deployment = durable_runtime.deployment(self.pipeline_name, self.wrapper)
        record = None
        if context.current_task is not None:
            # An existing A2A task may predate durable execution or be restored
            # without its durable record; treat it as a new durable submission.
            with suppress(KeyError):
                record = await deployment.get(task.id)
        if record is not None and record.status is ExecutionStatus.WAITING:
            await deployment.resume(
                task.id,
                {"messages": [message.to_dict() for message in _build_haystack_resume_messages(context)]},
            )
            await updater.start_work()
        elif record is None:
            await deployment.submit_agent_messages(_build_haystack_messages(context), execution_id=task.id)
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
        deployment = durable_runtime.deployment(self.pipeline_name, self.wrapper)
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
            self._projections[execution_id] = _Projection(updater=updater, task=task)
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
            due = sorted(
                [
                    (execution_id, projection)
                    for execution_id, projection in list(self._projections.items())
                    if projection.retry_at <= now
                ],
                key=lambda item: item[1].retry_at,
            )
            batch = due[: settings.a2a_projection_batch_size]
            for execution_id, projection in batch:
                try:
                    await self._reconcile_one(execution_id, projection)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._backoff_projection(execution_id, projection, error)
            if len(due) > len(batch):
                await asyncio.sleep(0)
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=settings.a2a_projection_interval)

    async def _reconcile_one(  # noqa: C901, PLR0912 - explicit projection state table
        self,
        execution_id: str,
        projection: _Projection,
    ) -> None:
        if not await self._own_projection(execution_id, projection):
            projection.retry_at = time.monotonic() + settings.a2a_projection_interval
            return
        deployment = durable_runtime.deployment(self.pipeline_name, self.wrapper)
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
                record = await deployment.get(execution_id)
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
        projection: _Projection,
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

    async def _refresh_projection(self, execution_id: str, projection: _Projection) -> None:
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

    async def _own_projection(self, execution_id: str, projection: _Projection) -> bool:
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

    async def _finish_projection(self, execution_id: str, projection: _Projection) -> None:
        self._projections.pop(execution_id, None)
        await self._release_projection(execution_id, projection)
        watcher = self._watchers.pop(execution_id, None)
        if watcher is not None and not watcher.done():
            watcher.set_result(None)

    async def _release_projection(self, execution_id: str, projection: _Projection) -> None:
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
        deployment = durable_runtime.deployment(self.pipeline_name, self.wrapper)
        for task in tasks:
            if task.id in self._projections:
                continue
            try:
                await deployment.get(task.id)
            except (KeyError, RuntimeError):
                continue
            projection = _Projection(
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


def _is_native_a2a_wrapper(wrapper: BasePipelineWrapper) -> bool:
    from hayhooks.a2a import A2APipelineWrapper

    return isinstance(wrapper, A2APipelineWrapper) and (
        getattr(type(wrapper), "create_a2a_agent_executor", None) is not A2APipelineWrapper.create_a2a_agent_executor
    )


def validate_agent_executor(executor: Any, pipeline_name: str) -> None:
    if not isinstance(executor, AgentExecutor):
        msg = f"Pipeline '{pipeline_name}' create_a2a_agent_executor() returned {type(executor).__name__}"
        raise TypeError(msg)


def create_agent_executor(
    wrapper_or_pipeline_name: BasePipelineWrapper | str,
    pipeline_name: str | None = None,
    *,
    runtime: Any | None = None,  # noqa: ARG001
    task_store: Any | None = None,
) -> AgentExecutor:
    """Select native A2A first, then durable Agent, then chat compatibility."""
    a2a_import.check()
    if pipeline_name is None:
        if not isinstance(wrapper_or_pipeline_name, str):
            msg = "The one-argument create_agent_executor form requires a pipeline name"
            raise TypeError(msg)
        pipeline_name = wrapper_or_pipeline_name
        wrapper = registry.get(pipeline_name)
    else:
        if isinstance(wrapper_or_pipeline_name, str):
            msg = "The two-argument form requires a pipeline wrapper"
            raise TypeError(msg)
        wrapper = wrapper_or_pipeline_name
    if wrapper is not None and _is_native_a2a_wrapper(wrapper):
        executor = cast(Any, wrapper).create_a2a_agent_executor()
        validate_agent_executor(executor, pipeline_name)
        return executor
    if wrapper is not None and getattr(wrapper, "durable", False):
        if task_store is None:
            msg = "A durable A2A Agent requires an A2A task store"
            raise RuntimeError(msg)
        # Constructing the deployment here validates the Haystack v3 Agent and the
        # durable definition before the Agent Card is exposed.
        durable_runtime.deployment(pipeline_name, wrapper)
        return DurableAgentExecutor(wrapper, pipeline_name, task_store)
    return ChatCompletionAgentExecutor(pipeline_name, wrapper)


async def _run_chat_completion(pipeline_name: str, context: RequestContext) -> Any:
    return await ChatCompletionAgentExecutor(pipeline_name, registry.get(pipeline_name))._run_chat_completion(context)


async def _execute_agent_task(pipeline_name: str, context: RequestContext, event_queue: EventQueue) -> None:
    await ChatCompletionAgentExecutor(pipeline_name, registry.get(pipeline_name))._execute_agent_task(
        context, event_queue
    )


__all__ = ["ChatCompletionAgentExecutor", "DurableAgentExecutor", "create_agent_executor"]
