import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi_openai_compat import (
    FileObject,
    create_chat_completion_router,
    create_files_router,
    create_models_router,
    create_responses_router,
)
from haystack.dataclasses import StreamingChunk

from hayhooks.server.logger import log
from hayhooks.server.pipelines import registry
from hayhooks.server.tracing import (
    SPAN_OPENAI_FILE_UPLOAD,
    SPAN_OPENAI_RUN,
    build_streaming_trace_tags,
    build_trace_tags,
    trace_async_stream,
    trace_operation,
    trace_sync_stream,
)
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


@dataclass(frozen=True)
class _OpenAIDispatch:
    """Describes how to dispatch an OpenAI-compatible call to a pipeline wrapper."""

    operation_kind: str
    sync_flag: str
    async_flag: str
    sync_method: str
    async_method: str
    not_implemented_detail: str


_CHAT_COMPLETION_DISPATCH = _OpenAIDispatch(
    operation_kind="run_chat_completion",
    sync_flag="_is_run_chat_completion_implemented",
    async_flag="_is_run_chat_completion_async_implemented",
    sync_method="run_chat_completion",
    async_method="run_chat_completion_async",
    not_implemented_detail="Chat endpoint not implemented for this model",
)

_RESPONSE_DISPATCH = _OpenAIDispatch(
    operation_kind="run_response",
    sync_flag="_is_run_response_implemented",
    async_flag="_is_run_response_async_implemented",
    sync_method="run_response",
    async_method="run_response_async",
    not_implemented_detail="Responses endpoint not implemented for this model",
)


def _list_models() -> list[str]:
    return registry.get_names()


def _chunk_to_text(chunk: Any) -> str:
    if hasattr(chunk, "content"):
        return chunk.content
    if isinstance(chunk, str):
        return chunk
    return ""


def _collect_sync_generator(gen: Generator) -> str:
    """Consume a sync generator and return the concatenated string content."""
    return "".join(_chunk_to_text(chunk) for chunk in gen)


async def _collect_async_generator(gen: AsyncGenerator) -> str:
    """Consume an async generator and return the concatenated string content."""
    return "".join([_chunk_to_text(chunk) async for chunk in gen])


def _resolve_pipeline_wrapper(model: str) -> BasePipelineWrapper:
    """Look up *model* in the registry, raising 404 if it isn't a pipeline wrapper."""
    pipeline_wrapper = registry.get(model)
    if not isinstance(pipeline_wrapper, BasePipelineWrapper):
        raise HTTPException(status_code=404, detail=f"Pipeline '{model}' not found or not a pipeline wrapper")
    return pipeline_wrapper


def _select_execution_mode(wrapper: BasePipelineWrapper, dispatch: _OpenAIDispatch) -> tuple[str, str]:
    """
    Return the ``(mode, method_name)`` the wrapper should be invoked with.

    ``mode`` is ``"async"`` or ``"sync"``. Raises ``HTTPException(501)`` when
    neither variant is implemented.
    """
    if getattr(wrapper, dispatch.async_flag):
        return "async", dispatch.async_method
    if getattr(wrapper, dispatch.sync_flag):
        return "sync", dispatch.sync_method
    raise HTTPException(status_code=501, detail=dispatch.not_implemented_detail)


async def _invoke_pipeline_method(
    wrapper: BasePipelineWrapper, *, mode: str, method_name: str, model: str, call_kwargs: dict[str, Any]
) -> Any:
    """Invoke the resolved pipeline method in either async or threadpool-sync mode."""
    method = getattr(wrapper, method_name)
    log.debug("Using {} ({}) for model: {}", method_name, mode, model)
    if mode == "async":
        return await method(model=model, **call_kwargs)
    return await run_in_threadpool(method, model=model, **call_kwargs)


def _wrap_string_as_streaming(text: str) -> Generator[StreamingChunk, None, None]:
    """Adapt a plain string result to the streaming-chunk generator shape."""
    yield StreamingChunk(content=text)


async def _normalize_result(result: Any, *, stream_requested: bool) -> str | Generator | AsyncGenerator:
    """Normalize the wrapper's return value to what the OpenAI router expects."""
    if not stream_requested:
        if isinstance(result, Generator):
            return _collect_sync_generator(result)
        if isinstance(result, AsyncGenerator):
            return await _collect_async_generator(result)
        return result

    if isinstance(result, str):
        return _wrap_string_as_streaming(result)
    return result


async def _run_pipeline_method(
    dispatch: _OpenAIDispatch,
    *,
    model: str,
    kwargs: dict[str, Any],
    body: dict[str, Any],
) -> str | Generator | AsyncGenerator:
    """Shared dispatch logic for chat completions and responses endpoints."""
    stream_requested = bool(body.get("stream", False))
    trace_tags = build_trace_tags(
        {
            "hayhooks.transport": "openai",
            "hayhooks.openai.operation": dispatch.operation_kind,
            "hayhooks.pipeline.name": model,
            "hayhooks.openai.stream_requested": stream_requested,
        }
    )
    if stream_requested:
        try:
            wrapper = _resolve_pipeline_wrapper(model)
            mode, method_name = _select_execution_mode(wrapper, dispatch)
            result = await _invoke_pipeline_method(
                wrapper, mode=mode, method_name=method_name, model=model, call_kwargs={**kwargs, "body": body}
            )
            normalized_result = await _normalize_result(result, stream_requested=stream_requested)
        except BaseException:
            with trace_operation(SPAN_OPENAI_RUN, tags=trace_tags):
                raise

        trace_tags = build_trace_tags(trace_tags, **{"hayhooks.openai.execution_mode": mode})
        streaming_trace_tags = build_streaming_trace_tags(trace_tags, stream_type="sse")
        if isinstance(normalized_result, AsyncGenerator):
            return trace_async_stream(normalized_result, SPAN_OPENAI_RUN, tags=streaming_trace_tags)
        if isinstance(normalized_result, Generator):
            return trace_sync_stream(normalized_result, SPAN_OPENAI_RUN, tags=streaming_trace_tags)
        return normalized_result

    with trace_operation(SPAN_OPENAI_RUN, tags=trace_tags) as span:
        wrapper = _resolve_pipeline_wrapper(model)
        mode, method_name = _select_execution_mode(wrapper, dispatch)
        span.set_tag("hayhooks.openai.execution_mode", mode)
        result = await _invoke_pipeline_method(
            wrapper, mode=mode, method_name=method_name, model=model, call_kwargs={**kwargs, "body": body}
        )
        return await _normalize_result(result, stream_requested=stream_requested)


async def _run_completion(
    model: str, messages: list[dict[str, Any]], body: dict[str, Any]
) -> str | Generator | AsyncGenerator:
    return await _run_pipeline_method(_CHAT_COMPLETION_DISPATCH, model=model, kwargs={"messages": messages}, body=body)


async def _run_response(
    model: str, input_items: list[dict[str, Any]], body: dict[str, Any]
) -> str | Generator | AsyncGenerator:
    return await _run_pipeline_method(_RESPONSE_DISPATCH, model=model, kwargs={"input_items": input_items}, body=body)


def _find_file_upload_wrapper() -> BasePipelineWrapper | None:
    """Find the first registered pipeline wrapper that implements ``run_file_upload``."""
    for name in registry.get_names():
        wrapper = registry.get(name)
        if isinstance(wrapper, BasePipelineWrapper) and wrapper._is_run_file_upload_implemented:
            return wrapper
    return None


async def _run_file_upload(filename: str | None, content_type: str | None, content: bytes, purpose: str) -> FileObject:
    with trace_operation(
        SPAN_OPENAI_FILE_UPLOAD,
        tags=build_trace_tags(
            {
                "hayhooks.transport": "openai",
                "hayhooks.openai.operation": "file_upload",
                "hayhooks.openai.file.content_type": content_type,
                "hayhooks.openai.file.purpose": purpose,
                "hayhooks.openai.file.size_bytes": len(content),
            }
        ),
    ):
        wrapper = _find_file_upload_wrapper()
        if wrapper is not None:
            result = await run_in_threadpool(wrapper.run_file_upload, filename, content_type, content, purpose)
            if isinstance(result, FileObject):
                return result
            if isinstance(result, dict):
                return FileObject.model_validate(result)
            log.error(
                "run_file_upload returned unsupported type {} from wrapper {!r}",
                type(result).__name__,
                wrapper,
            )
            raise HTTPException(status_code=500, detail="run_file_upload returned an unsupported type")

        file_id = f"file-{uuid4().hex[:24]}"
        log.warning(
            "No pipeline implements run_file_upload: file '{}' (id={}) not persisted. "
            "Override run_file_upload in your PipelineWrapper to store files.",
            filename,
            file_id,
        )
        return FileObject(
            id=file_id,
            object="file",
            bytes=len(content),
            created_at=int(time.time()),
            filename=filename or "",
            purpose=purpose,
        )


router = APIRouter()

router.include_router(
    create_models_router(
        list_models=_list_models,
        owned_by="hayhooks",
        tags=["openai"],
    )
)

router.include_router(
    create_chat_completion_router(
        list_models=_list_models,
        run_completion=_run_completion,
        owned_by="hayhooks",
        tags=["openai"],
        include_models_endpoints=False,
    )
)

router.include_router(
    create_responses_router(
        list_models=_list_models,
        run_response=_run_response,
        owned_by="hayhooks",
        tags=["openai"],
        include_models_endpoints=False,
    )
)

router.include_router(
    create_files_router(
        run_file_upload=_run_file_upload,
        tags=["openai"],
    )
)
