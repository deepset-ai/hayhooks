import time
from collections.abc import AsyncGenerator, Generator
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
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


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
    chunks = [_chunk_to_text(chunk) async for chunk in gen]
    return "".join(chunks)


async def _run_pipeline_method(  # noqa: PLR0913
    model: str,
    kwargs: dict,
    body: dict,
    *,
    sync_flag: str,
    async_flag: str,
    sync_method: str,
    async_method: str,
    not_implemented_detail: str,
) -> str | Generator | AsyncGenerator:
    """Shared dispatch logic for chat completions and responses endpoints."""
    pipeline_wrapper = registry.get(model)
    if not isinstance(pipeline_wrapper, BasePipelineWrapper):
        raise HTTPException(status_code=404, detail=f"Pipeline '{model}' not found or not a pipeline wrapper")

    sync_implemented = getattr(pipeline_wrapper, sync_flag)
    async_implemented = getattr(pipeline_wrapper, async_flag)

    if not sync_implemented and not async_implemented:
        raise HTTPException(status_code=501, detail=not_implemented_detail)

    if async_implemented:
        log.debug("Using {} for model: {}", async_method, model)
        result = await getattr(pipeline_wrapper, async_method)(model=model, **kwargs, body=body)
    else:
        log.debug("Using {} (sync) for model: {}", sync_method, model)
        result = await run_in_threadpool(getattr(pipeline_wrapper, sync_method), model=model, **kwargs, body=body)

    stream_requested = body.get("stream", False)
    if not stream_requested:
        if isinstance(result, Generator):
            return _collect_sync_generator(result)
        if isinstance(result, AsyncGenerator):
            return await _collect_async_generator(result)

    if stream_requested and isinstance(result, str):
        def _wrap_str_as_generator():
            yield StreamingChunk(content=result)

        return _wrap_str_as_generator()

    return result


async def _run_completion(model: str, messages: list[dict], body: dict) -> str | Generator | AsyncGenerator:
    return await _run_pipeline_method(
        model,
        {"messages": messages},
        body,
        sync_flag="_is_run_chat_completion_implemented",
        async_flag="_is_run_chat_completion_async_implemented",
        sync_method="run_chat_completion",
        async_method="run_chat_completion_async",
        not_implemented_detail="Chat endpoint not implemented for this model",
    )


async def _run_response(model: str, input_items: list[dict], body: dict) -> str | Generator | AsyncGenerator:
    return await _run_pipeline_method(
        model,
        {"input_items": input_items},
        body,
        sync_flag="_is_run_response_implemented",
        async_flag="_is_run_response_async_implemented",
        sync_method="run_response",
        async_method="run_response_async",
        not_implemented_detail="Responses endpoint not implemented for this model",
    )


def _find_file_upload_wrapper() -> BasePipelineWrapper | None:
    """Find the first registered pipeline wrapper that implements ``run_file_upload``."""
    for name in registry.get_names():
        wrapper = registry.get(name)
        if isinstance(wrapper, BasePipelineWrapper) and wrapper._is_run_file_upload_implemented:
            return wrapper
    return None


async def _run_file_upload(
    filename: str | None, content_type: str | None, content: bytes, purpose: str
) -> FileObject:
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
