"""A2A adapter for managed durable-Agent executions."""

from __future__ import annotations

import asyncio
import builtins
import json
from typing import Any, cast

from hayhooks.a2a import RecoverableTaskStore, default_a2a_owner
from hayhooks.durable.models import ExecutionAdmissionError, ExecutionStatus, ExecutionStoreError
from hayhooks.durable.runtime import DefinitionRevisionConflictError, DurableDeployment, execution_id_for
from hayhooks.server.a2a.imports import (
    AgentExecutor,
    EventQueue,
    InvalidParamsError,
    RequestContext,
    TaskStore,
    TaskUpdater,
    new_task_from_user_message,
    new_text_part,
)
from hayhooks.server.a2a.messages import (
    build_haystack_messages,
    build_haystack_resume_messages,
    build_haystack_task_messages,
    task_is_terminal,
    task_matches_filters,
)
from hayhooks.server.logger import log
from hayhooks.settings import settings

DURABLE_PROGRESS_ARTIFACT_NAME = "durable-progress"
DURABLE_RESULT_ARTIFACT_NAME = "durable-result"


class _TaskProjectionQueue:
    """Apply A2A events to a transient task snapshot."""

    def __init__(self, task: Any) -> None:
        self.task = task

    async def enqueue_event(self, event: Any) -> None:
        from a2a.server.tasks.task_manager import append_artifact_to_task
        from a2a.types import TaskArtifactUpdateEvent, TaskStatusUpdateEvent

        if isinstance(event, TaskStatusUpdateEvent):
            if self.task.status.HasField("message"):
                self.task.history.append(self.task.status.message)
            if event.metadata:
                self.task.metadata.MergeFrom(event.metadata)
            self.task.status.CopyFrom(event.status)
        elif isinstance(event, TaskArtifactUpdateEvent):
            append_artifact_to_task(self.task, event)


class DurableTaskStore(TaskStore):
    """Read durable execution state through an ordinary A2A task store."""

    def __init__(self, task_store: TaskStore, deployment: DurableDeployment) -> None:
        self._task_store = task_store
        self._deployment = deployment
        self._read_through_task_ids: set[str] = set()

    async def save(self, task: Any, context: Any) -> None:
        await self._task_store.save(task, context)

    async def get(self, task_id: str, context: Any) -> Any | None:
        owner_id = self.owner_id_for_context(context)
        task = await self._task_store.get(task_id, context)
        record = None
        if task is None:
            try:
                record = await self._deployment.get(
                    execution_id_for(owner_id, task_id),
                    owner_id=owner_id,
                    enforce_owner=True,
                    allow_revision_mismatch=True,
                )
            except KeyError:
                return None
            context_id = record.validated_input.get("a2a_context_id")
            if not isinstance(context_id, str) or not context_id:
                return None
            from a2a.types import Task, TaskState

            task = Task(id=task_id, context_id=context_id)
            task.status.state = (
                TaskState.TASK_STATE_WORKING
                if record.status is ExecutionStatus.RUNNING
                else TaskState.TASK_STATE_SUBMITTED
            )
        return await self._project(task, owner_id, context, record=record)

    async def list(self, params: Any, context: Any) -> Any:
        from a2a.types import ListTasksResponse
        from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
        from a2a.utils.task import decode_page_token, encode_page_token

        tasks = await self._all_tasks(params, context)
        owner_id = self.owner_id_for_context(context)
        projected: builtins.list[Any | None] = []
        # ponytail: fixed fan-out keeps Redis pools bounded; make adaptive only
        # if task-list latency becomes a measured bottleneck.
        for offset in range(0, len(tasks), 32):
            projected.extend(
                await asyncio.gather(*(self._project(task, owner_id, context) for task in tasks[offset : offset + 32]))
            )
        timestamp_after = params.status_timestamp_after if params.HasField("status_timestamp_after") else None
        filtered = [
            task for task in projected if task is not None and task_matches_filters(task, params, timestamp_after)
        ]
        page_size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
        start = 0
        if params.page_token:
            task_id = decode_page_token(params.page_token)
            try:
                start = next(index for index, task in enumerate(filtered) if task.id == task_id) + 1
            except StopIteration as error:
                msg = f"Invalid page token: {params.page_token}"
                raise InvalidParamsError(msg) from error
        page = filtered[start : start + page_size]
        next_page_token = encode_page_token(page[-1].id) if start + len(page) < len(filtered) else None
        return ListTasksResponse(
            tasks=page,
            next_page_token=next_page_token,
            page_size=page_size,
            total_size=len(filtered),
        )

    async def delete(self, task_id: str, context: Any) -> None:
        await self._task_store.delete(task_id, context)

    def owner_id_for_context(self, context: Any) -> str:
        resolver = getattr(self._task_store, "owner_id_for_context", None)
        call_context = getattr(context, "call_context", context)
        return resolver(call_context) if callable(resolver) else default_a2a_owner(call_context)

    async def _all_tasks(self, params: Any, context: Any) -> builtins.list[Any]:
        """Read the owner task list before applying live durable-state filters."""
        # ponytail: scans one owner's A2A tasks for exact live-state filters;
        # add a durable-state index only if this becomes a measured hot path.
        request = type(params)()
        request.CopyFrom(params)
        request.ClearField("context_id")
        request.ClearField("status")
        request.ClearField("status_timestamp_after")
        request.ClearField("page_token")
        request.page_size = settings.a2a_list_scan_batch_size
        tasks: list[Any] = []
        while True:
            page = await self._task_store.list(request, context)
            tasks.extend(page.tasks)
            if not page.next_page_token:
                return tasks
            request.page_token = page.next_page_token

    async def _project(  # noqa: C901
        self, task: Any | None, owner_id: str, context: Any, *, record: Any | None = None
    ) -> Any | None:
        if task is None:
            return None
        projected = type(task)()
        projected.CopyFrom(task)
        copy_version = getattr(self._task_store, "copy_task_version", None)
        if callable(copy_version):
            copy_version(task, projected)
        settled = False
        try:
            if record is None:
                record = await self._deployment.get(
                    execution_id_for(owner_id, task.id),
                    owner_id=owner_id,
                    enforce_owner=True,
                    allow_revision_mismatch=True,
                )
        except KeyError:
            if task_is_terminal(task):
                return projected
            if task.HasField("status"):
                from a2a.types import TaskState

                if task.status.state == TaskState.TASK_STATE_SUBMITTED:
                    return projected
                updater = TaskUpdater(cast(EventQueue, _TaskProjectionQueue(projected)), task.id, task.context_id)
                await updater.failed(
                    message=updater.new_agent_message(
                        [new_text_part("The durable Agent execution record is missing (durable_execution_missing).")]
                    )
                )
                settled = True
        else:
            if _task_matches_record(task, record):
                return projected
            settled = await _project_record(
                record,
                TaskUpdater(cast(EventQueue, _TaskProjectionQueue(projected)), task.id, task.context_id),
            )
        if settled and task.id in self._read_through_task_ids:
            try:
                await self._task_store.save(projected, context)
            except InvalidParamsError:
                pass
            else:
                self._read_through_task_ids.discard(task.id)
        return projected


class DurableAgentExecutor(AgentExecutor):
    """Run and stream a managed durable Agent without a second lifecycle store."""

    def __init__(self, pipeline_name: str, task_store: TaskStore, deployment: DurableDeployment) -> None:
        self.pipeline_name = pipeline_name
        self._task_store = task_store
        self.task_store = DurableTaskStore(task_store, deployment)
        self.deployment = deployment
        self._closed = False

    def health(self) -> dict[str, Any]:
        return {"healthy": not self._closed}

    async def start(self) -> None:
        self._closed = False
        if isinstance(self._task_store, RecoverableTaskStore):
            await self._recover_tasks(cast(RecoverableTaskStore, self._task_store))

    async def close(self) -> None:
        self._closed = True

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Submit or resume a durable execution and project its state into A2A events."""
        task = _task(context)
        self.task_store._read_through_task_ids.discard(task.id)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        owner_id = self.task_store.owner_id_for_context(context)
        execution_id = execution_id_for(owner_id, task.id)
        record = None
        if context.current_task is not None:
            try:
                record = await self.deployment.get(
                    execution_id,
                    owner_id=owner_id,
                    enforce_owner=True,
                    allow_revision_mismatch=True,
                )
            except KeyError:
                from a2a.types import TaskState

                if task.status.state != TaskState.TASK_STATE_SUBMITTED:
                    await updater.failed(
                        message=updater.new_agent_message(
                            [
                                new_text_part(
                                    "The durable Agent execution record is missing (durable_execution_missing)."
                                )
                            ]
                        )
                    )
                    return
        if record is not None and record.status is ExecutionStatus.WAITING:
            action = "resume"
            try:
                resumed = await self.deployment.resume(
                    execution_id,
                    {"messages": [message.to_dict() for message in build_haystack_resume_messages(context)]},
                    owner_id=owner_id,
                    enforce_owner=True,
                )
            except DefinitionRevisionConflictError as error:
                raise InvalidParamsError(str(error)) from error
            if not resumed:
                msg = f"Task '{task.id}' is no longer accepting follow-up messages"
                raise InvalidParamsError(msg)
        elif record is None:
            action = "submit"
            await self._task_store.save(task, context.call_context)
            try:
                record = await self._submit(task, owner_id, build_haystack_messages(context))
            except ValueError as error:
                log.bind(
                    pipeline_name=self.pipeline_name,
                    task_id=task.id,
                    error_type=type(error).__name__,
                ).warning("Rejected durable A2A task submission")
                await updater.failed(
                    message=updater.new_agent_message(
                        [new_text_part("The durable Agent submission was rejected (durable_submission_rejected).")]
                    )
                )
                return
            execution_id = record.execution_id
        else:
            msg = f"Task '{task.id}' is already running and cannot accept another message"
            raise InvalidParamsError(msg)
        log.bind(
            pipeline_name=self.pipeline_name,
            task_id=task.id,
            execution_id=execution_id,
            action=action,
        ).debug("Accepted durable A2A task action")
        await updater.start_work()
        await self._wait_for_update(execution_id, owner_id, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Request cancellation and continue projecting until the task settles."""
        task = context.current_task
        if task is None:
            return
        self.task_store._read_through_task_ids.discard(task.id)
        owner_id = self.task_store.owner_id_for_context(context)
        execution_id = execution_id_for(owner_id, task.id)
        try:
            accepted = await self.deployment.request_cancel(
                execution_id,
                owner_id=owner_id,
                enforce_owner=True,
            )
        except KeyError:
            return
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        if accepted:
            log.bind(pipeline_name=self.pipeline_name, task_id=task.id, execution_id=execution_id).debug(
                "Accepted durable A2A task cancellation"
            )
            await updater.add_artifact(
                [new_text_part("Cancellation requested")],
                artifact_id=f"{task.id}-{DURABLE_PROGRESS_ARTIFACT_NAME}",
                name=DURABLE_PROGRESS_ARTIFACT_NAME,
                append=False,
            )
        await self._wait_for_update(execution_id, owner_id, updater, terminal_only=True)

    async def _wait_for_update(
        self, execution_id: str, owner_id: str, updater: TaskUpdater, *, terminal_only: bool = False
    ) -> None:
        last_sequence = -1
        while not self._closed:
            try:
                record = await self.deployment.get(
                    execution_id,
                    owner_id=owner_id,
                    enforce_owner=True,
                    allow_revision_mismatch=True,
                )
            except KeyError:
                await updater.failed(
                    message=updater.new_agent_message(
                        [new_text_part("The durable Agent execution record is missing (durable_execution_missing).")]
                    )
                )
                return
            except ExecutionStoreError:
                await asyncio.sleep(max(0.1, settings.durable_poll_interval))
                continue
            if record.sequence != last_sequence:
                last_sequence = record.sequence
                settled = await _project_record(record, updater)
                if settled and (not terminal_only or record.status is not ExecutionStatus.WAITING):
                    log.bind(
                        pipeline_name=self.pipeline_name,
                        task_id=updater.task_id,
                        execution_id=execution_id,
                        status=record.status.value,
                    ).debug("Projected durable A2A task state")
                    return
            await asyncio.sleep(max(0.1, settings.durable_poll_interval))

    async def _submit(self, task: Any, owner_id: str, messages: list[Any]) -> Any:
        payload = {
            "messages": [message.to_dict() for message in messages],
            "a2a_context_id": task.context_id,
        }
        while not self._closed:
            try:
                return (await self.deployment.submit(payload, execution_id=task.id, owner_id=owner_id))[1]
            except (ExecutionAdmissionError, ExecutionStoreError):
                await asyncio.sleep(max(0.1, settings.durable_poll_interval))
        raise asyncio.CancelledError

    async def _recover_tasks(self, task_store: RecoverableTaskStore) -> None:  # noqa: C901
        """Repair durable A2A tasks saved before this executor started."""
        cursor = 0
        recovered = 0
        while not self._closed:
            tasks, next_cursor = await task_store.recoverable_task_batch(cursor, settings.a2a_list_scan_batch_size)
            for task, owner_id, version in tasks:
                execution_id = execution_id_for(owner_id, task.id)
                try:
                    record = await self.deployment.get(
                        execution_id,
                        owner_id=owner_id,
                        enforce_owner=True,
                        allow_revision_mismatch=True,
                    )
                except KeyError:
                    from a2a.types import TaskState

                    if task.status.state != TaskState.TASK_STATE_SUBMITTED:
                        continue
                    try:
                        record = await self._submit(task, owner_id, build_haystack_task_messages(task))
                    except ValueError as error:
                        record = None
                        log.bind(
                            pipeline_name=self.pipeline_name,
                            task_id=task.id,
                            error_type=type(error).__name__,
                        ).warning("Rejected recovered durable A2A task submission")
                if record is not None and record.status in {ExecutionStatus.QUEUED, ExecutionStatus.RUNNING}:
                    self.task_store._read_through_task_ids.add(task.id)
                if record is not None and _task_matches_record(task, record):
                    continue
                projected = type(task)()
                projected.CopyFrom(task)
                updater = TaskUpdater(cast(EventQueue, _TaskProjectionQueue(projected)), task.id, task.context_id)
                if record is None:
                    await updater.failed(
                        message=updater.new_agent_message(
                            [new_text_part("The durable Agent submission was rejected (durable_submission_rejected).")]
                        )
                    )
                else:
                    await _project_record(record, updater)
                if not await task_store.save_projection(projected, owner_id, version):
                    continue
                task.CopyFrom(projected)
                recovered += 1
            if next_cursor is None:
                if recovered:
                    log.bind(pipeline_name=self.pipeline_name, recovered=recovered).debug(
                        "Recovered durable A2A task projections"
                    )
                return
            cursor = next_cursor


def _task(context: RequestContext) -> Any:
    if context.current_task is not None:
        return context.current_task
    if context.message is None:
        msg = "A2A request has neither a current task nor a message"
        raise ValueError(msg)
    return new_task_from_user_message(context.message)


async def _project_record(record: Any, updater: TaskUpdater) -> bool:
    """Project one durable record and report whether the active request can stop waiting."""
    if record.progress:
        await updater.add_artifact(
            [new_text_part("\n".join(event.message for event in record.progress))],
            artifact_id=f"{updater.task_id}-{DURABLE_PROGRESS_ARTIFACT_NAME}",
            name=DURABLE_PROGRESS_ARTIFACT_NAME,
            append=False,
        )
    if record.status is ExecutionStatus.WAITING:
        await updater.requires_input(
            message=updater.new_agent_message([new_text_part("The durable Agent requires input to continue.")])
        )
    elif record.status is ExecutionStatus.COMPLETED:
        await updater.add_artifact(
            [new_text_part(_result_text(record.result))],
            artifact_id=f"{updater.task_id}-{DURABLE_RESULT_ARTIFACT_NAME}",
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
        return False
    return True


def _task_matches_record(task: Any, record: Any) -> bool:
    from a2a.types import TaskState

    states = {
        ExecutionStatus.WAITING: TaskState.TASK_STATE_INPUT_REQUIRED,
        ExecutionStatus.COMPLETED: TaskState.TASK_STATE_COMPLETED,
        ExecutionStatus.FAILED: TaskState.TASK_STATE_FAILED,
        ExecutionStatus.CANCELED: TaskState.TASK_STATE_CANCELED,
    }
    return record.status in states and task.status.state == states[record.status]


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
        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result or "")
