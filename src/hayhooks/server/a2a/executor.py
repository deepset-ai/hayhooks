"""A2A chat executor and authoring-mode selection."""

from __future__ import annotations

import asyncio
import traceback
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any

from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from haystack.dataclasses import StreamingChunk

from hayhooks.durable.mode import DurableAuthoringMode, durable_authoring_mode
from hayhooks.durable.runtime import DurableRuntime
from hayhooks.server.a2a.durable_executor import DurableAgentExecutor
from hayhooks.server.a2a.imports import (
    AgentExecutor,
    EventQueue,
    RequestContext,
    TaskUpdater,
    new_task_from_user_message,
    new_text_part,
)
from hayhooks.server.a2a.messages import build_openai_messages
from hayhooks.server.logger import log
from hayhooks.server.tracing import SPAN_A2A_RUN_AGENT, build_trace_tags, trace_operation
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings

RESPONSE_ARTIFACT_NAME = "response"


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

    def __init__(self, pipeline_name: str, pipeline_wrapper: BasePipelineWrapper) -> None:
        self.pipeline_name = pipeline_name
        self.pipeline_wrapper = pipeline_wrapper

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Run one A2A task and stream its result through task artifacts."""
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
        task_log = log.bind(pipeline_name=self.pipeline_name, task_id=task.id)
        task_log.debug("Started A2A task")

        with trace_operation(
            SPAN_A2A_RUN_AGENT,
            tags=build_trace_tags({"hayhooks.transport": "a2a", "hayhooks.pipeline.name": self.pipeline_name}),
        ):
            try:
                messages = build_openai_messages(context)
                if self.pipeline_wrapper._is_run_chat_completion_async_implemented:
                    result = await self.pipeline_wrapper.run_chat_completion_async(
                        model=self.pipeline_name,
                        messages=messages,
                        body={},
                    )
                else:
                    result = await run_in_threadpool(
                        self.pipeline_wrapper.run_chat_completion,
                        model=self.pipeline_name,
                        messages=messages,
                        body={},
                    )
                await _stream_result_as_artifact(result, updater)
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
        task_log.debug("Completed A2A task")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current A2A task when the request references one."""
        if context.current_task is not None:
            await TaskUpdater(event_queue, context.current_task.id, context.current_task.context_id).cancel()
            log.bind(pipeline_name=self.pipeline_name, task_id=context.current_task.id).debug("Canceled A2A task")


def create_agent_executor(
    wrapper: BasePipelineWrapper,
    pipeline_name: str,
    *,
    task_store: Any | None = None,
    durable_runtime: DurableRuntime | None = None,
) -> AgentExecutor:
    """Select a managed durable Agent or chat-compatible executor."""
    if durable_authoring_mode(wrapper) is DurableAuthoringMode.MANAGED_AGENT:
        if task_store is None:
            msg = "A durable A2A Agent requires an A2A task store"
            raise RuntimeError(msg)
        if durable_runtime is None:
            msg = "A durable A2A Agent requires an application-owned DurableRuntime"
            raise RuntimeError(msg)
        # Constructing the deployment here validates the Haystack v3 Agent and the
        # durable definition before the Agent Card is exposed.
        deployment = durable_runtime.deployment(pipeline_name, wrapper)
        return DurableAgentExecutor(pipeline_name, task_store, deployment)
    return ChatCompletionAgentExecutor(pipeline_name, wrapper)


__all__ = [
    "ChatCompletionAgentExecutor",
    "DurableAgentExecutor",
    "create_agent_executor",
]
