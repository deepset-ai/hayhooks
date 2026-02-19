from collections.abc import AsyncGenerator, Generator

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi_openai_compat import (
    ChatCompletion,
    ChatRequest,
    Message,
    ModelObject,
    ModelsResponse,
    create_openai_router,
)

from hayhooks.server.logger import log
from hayhooks.server.pipelines import registry
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

# Re-export models for backward compatibility
__all__ = ["ChatCompletion", "ChatRequest", "Message", "ModelObject", "ModelsResponse", "router"]


def _list_models() -> list[str]:
    return registry.get_names()


def _collect_sync_generator(gen: Generator) -> str:
    """Consume a sync generator and return the concatenated string content."""
    chunks = []
    for chunk in gen:
        if hasattr(chunk, "content"):
            chunks.append(chunk.content)
        elif isinstance(chunk, str):
            chunks.append(chunk)
    return "".join(chunks)


async def _collect_async_generator(gen: AsyncGenerator) -> str:
    """Consume an async generator and return the concatenated string content."""
    chunks = []
    async for chunk in gen:
        if hasattr(chunk, "content"):
            chunks.append(chunk.content)
        elif isinstance(chunk, str):
            chunks.append(chunk)
    return "".join(chunks)


async def _run_completion(model: str, messages: list[dict], body: dict) -> str | Generator | AsyncGenerator:
    pipeline_wrapper = registry.get(model)
    if not isinstance(pipeline_wrapper, BasePipelineWrapper):
        raise HTTPException(status_code=404, detail=f"Pipeline '{model}' not found or not a pipeline wrapper")

    sync_implemented = pipeline_wrapper._is_run_chat_completion_implemented
    async_implemented = pipeline_wrapper._is_run_chat_completion_async_implemented

    if not sync_implemented and not async_implemented:
        raise HTTPException(status_code=501, detail="Chat endpoint not implemented for this model")

    if async_implemented:
        log.debug("Using run_chat_completion_async for model: {}", model)
        result = await pipeline_wrapper.run_chat_completion_async(
            model=model,
            messages=messages,
            body=body,
        )
    else:
        log.debug("Using run_chat_completion (sync) for model: {}", model)
        result = await run_in_threadpool(
            pipeline_wrapper.run_chat_completion,
            model=model,
            messages=messages,
            body=body,
        )

    # If the client didn't request streaming, collapse any generator into a single string
    stream_requested = body.get("stream", False)
    if not stream_requested:
        if isinstance(result, Generator):
            return _collect_sync_generator(result)
        if isinstance(result, AsyncGenerator):
            return await _collect_async_generator(result)

    return result


router = create_openai_router(
    list_models=_list_models,
    run_completion=_run_completion,
    owned_by="hayhooks",
    tags=["openai"],
)
