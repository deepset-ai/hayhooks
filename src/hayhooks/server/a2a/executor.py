"""A2A chat executor and authoring-mode selection."""

from __future__ import annotations

import asyncio
import traceback
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any, cast

from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from haystack.dataclasses import StreamingChunk

from hayhooks.durable.mode import DurableAuthoringMode, durable_authoring_mode
from hayhooks.durable.runtime import durable_runtime
from hayhooks.server.a2a.durable_executor import (
    DurableAgentExecutor,
    _ProjectionConflictError,
    _ProjectionCursor,
    _RecoveryEventQueue,
)
from hayhooks.server.a2a.imports import (
    AgentExecutor,
    EventQueue,
    RequestContext,
    TaskUpdater,
    a2a_import,
    new_task_from_user_message,
    new_text_part,
)
from hayhooks.server.a2a.messages import build_openai_messages
from hayhooks.server.logger import log
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.tracing import SPAN_A2A_RUN_AGENT, build_trace_tags, trace_operation
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings

RESPONSE_ARTIFACT_NAME = "response"

# Private compatibility alias retained for ``hayhooks.server.utils.a2a_utils``.
_build_openai_messages = build_openai_messages
# Deprecated internal alias kept for third-party tests that constructed a
# projection cursor directly before it acquired a descriptive name.
_Projection = _ProjectionCursor


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
        messages = build_openai_messages(context)
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
    runtime: Any | None = None,  # noqa: ARG001 - retained for public compatibility
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
    if wrapper is not None and durable_authoring_mode(wrapper) is DurableAuthoringMode.MANAGED_AGENT:
        if task_store is None:
            msg = "A durable A2A Agent requires an A2A task store"
            raise RuntimeError(msg)
        # Constructing the deployment here validates the Haystack v3 Agent and the
        # durable definition before the Agent Card is exposed.
        deployment = durable_runtime.deployment(pipeline_name, wrapper)
        return DurableAgentExecutor(wrapper, pipeline_name, task_store, deployment=deployment)
    return ChatCompletionAgentExecutor(pipeline_name, wrapper)


async def _run_chat_completion(pipeline_name: str, context: RequestContext) -> Any:
    return await ChatCompletionAgentExecutor(pipeline_name, registry.get(pipeline_name))._run_chat_completion(context)


async def _execute_agent_task(pipeline_name: str, context: RequestContext, event_queue: EventQueue) -> None:
    await ChatCompletionAgentExecutor(pipeline_name, registry.get(pipeline_name))._execute_agent_task(
        context, event_queue
    )


__all__ = [
    "ChatCompletionAgentExecutor",
    "DurableAgentExecutor",
    "_Projection",
    "_ProjectionConflictError",
    "_RecoveryEventQueue",
    "create_agent_executor",
]
