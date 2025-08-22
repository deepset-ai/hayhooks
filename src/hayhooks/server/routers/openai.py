import json
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Literal, Union

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from haystack.dataclasses import StreamingChunk
from pydantic import BaseModel, ConfigDict

from hayhooks.open_webui import OpenWebUIEvent
from hayhooks.server.logger import log
from hayhooks.server.pipelines import registry
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.deploy_utils import handle_pipeline_exceptions

router = APIRouter()


class ModelObject(BaseModel):
    id: str
    name: str
    object: Literal["model"]
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    data: list[ModelObject]
    object: Literal["list"]


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatRequest(OpenAIBaseModel):
    model: str
    messages: list[dict]
    stream: bool = False


class Message(OpenAIBaseModel):
    role: Literal["user", "assistant"]
    content: str


class Choice(OpenAIBaseModel):
    index: int
    delta: Union[Message, None] = None
    finish_reason: Union[Literal["stop"], None] = None
    logprobs: Union[None, dict] = None
    message: Union[Message, None] = None


class ChatCompletion(OpenAIBaseModel):
    id: str
    object: Union[Literal["chat.completion"], Literal["chat.completion.chunk"]]
    created: int
    model: str
    choices: list[Choice]


MODELS_PARAMS: dict = {
    "response_model": ModelsResponse,
    "tags": ["openai"],
}


def _event_to_sse_msg(
    data: dict,
) -> str:
    event_payload = {"event": data}
    return f"data: {json.dumps(event_payload)}\n\n"


def _create_sse_data_msg(
    resp_id: str, model_name: str, chunk_content: str = "", finish_reason: Union[Literal["stop"], None] = None
) -> str:
    response = ChatCompletion(
        id=resp_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model_name,
        choices=[Choice(index=0, delta=Message(role="assistant", content=chunk_content), finish_reason=finish_reason)],
    )
    return f"data: {response.model_dump_json()}\n\n"


def _create_sync_streaming_response(
    result: Generator[Union[StreamingChunk, OpenWebUIEvent, str], None, None], resp_id: str, model_name: str
) -> StreamingResponse:
    def stream_chunks() -> Generator[str, None, None]:
        for chunk in result:
            if isinstance(chunk, StreamingChunk):
                yield _create_sse_data_msg(resp_id=resp_id, model_name=model_name, chunk_content=chunk.content)
            elif isinstance(chunk, OpenWebUIEvent):
                yield _event_to_sse_msg(chunk.to_dict())
            elif isinstance(chunk, str):
                yield _create_sse_data_msg(resp_id=resp_id, model_name=model_name, chunk_content=chunk)

        # After consuming the generator, send a final event with finish_reason "stop"
        yield _create_sse_data_msg(resp_id=resp_id, model_name=model_name, finish_reason="stop")

    return StreamingResponse(stream_chunks(), media_type="text/event-stream")


def _create_async_streaming_response(
    result: AsyncGenerator[Union[StreamingChunk, OpenWebUIEvent, str], None], resp_id: str, model_name: str
) -> StreamingResponse:
    async def stream_chunks_async() -> AsyncGenerator[str, None]:
        async for chunk in result:
            if isinstance(chunk, StreamingChunk):
                yield _create_sse_data_msg(resp_id=resp_id, model_name=model_name, chunk_content=chunk.content)
            elif isinstance(chunk, OpenWebUIEvent):
                yield _event_to_sse_msg(chunk.to_dict())
            elif isinstance(chunk, str):
                yield _create_sse_data_msg(resp_id=resp_id, model_name=model_name, chunk_content=chunk)

        yield _create_sse_data_msg(resp_id=resp_id, model_name=model_name, finish_reason="stop")

    return StreamingResponse(stream_chunks_async(), media_type="text/event-stream")


def _chat_completion_response(result: str, resp_id: str, model_name: str) -> ChatCompletion:
    return ChatCompletion(
        id=resp_id,
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[Choice(index=0, message=Message(role="assistant", content=result), finish_reason="stop")],
    )


async def _execute_pipeline_method(
    pipeline_wrapper: BasePipelineWrapper, model: str, messages: list[dict], body: dict
) -> Union[str, Generator, AsyncGenerator]:
    # Check if either sync or async chat completion is implemented
    sync_implemented = pipeline_wrapper._is_run_chat_completion_implemented
    async_implemented = pipeline_wrapper._is_run_chat_completion_async_implemented

    if not sync_implemented and not async_implemented:
        raise HTTPException(status_code=501, detail="Chat endpoint not implemented for this model")

    # Determine which run_chat_completion method to use (prefer async if available)
    if async_implemented:
        log.debug(f"Using run_chat_completion_async for model: {model}")
        return await pipeline_wrapper.run_chat_completion_async(
            model=model,
            messages=messages,
            body=body,
        )
    else:
        log.debug(f"Using run_chat_completion (sync) for model: {model}")
        return await run_in_threadpool(
            pipeline_wrapper.run_chat_completion,
            model=model,
            messages=messages,
            body=body,
        )


@router.get("/v1/models", **MODELS_PARAMS, operation_id="openai_models")
@router.get("/models", **MODELS_PARAMS, operation_id="openai_models_alias")
async def get_models():
    """
    Implementation of OpenAI /models endpoint.

    Here we list all hayhooks pipelines (using `name` field).
    They will appear as selectable models in `open-webui` frontend.

    References:
    - https://github.com/ollama/ollama/blob/main/docs/openai.md
    - https://platform.openai.com/docs/api-reference/models/list
    """
    pipelines = registry.get_names()

    return ModelsResponse(
        data=[
            ModelObject(
                id=pipeline_name,
                name=pipeline_name,
                object="model",
                created=int(time.time()),
                owned_by="hayhooks",
            )
            for pipeline_name in pipelines
        ],
        object="list",
    )


CHAT_COMPLETION_PARAMS: dict = {
    "response_model": ChatCompletion,
    "tags": ["openai"],
}


@router.post("/chat/completions", **CHAT_COMPLETION_PARAMS, operation_id="openai_chat_completions")
@router.post("/v1/chat/completions", **CHAT_COMPLETION_PARAMS, operation_id="openai_chat_completions_alias")
@router.post("/{pipeline_name}/chat", **CHAT_COMPLETION_PARAMS, operation_id="chat_completions")
@handle_pipeline_exceptions()
async def chat_endpoint(chat_req: ChatRequest) -> Union[ChatCompletion, StreamingResponse]:
    # Get and validate pipeline wrapper
    pipeline_wrapper = registry.get(chat_req.model)
    if not isinstance(pipeline_wrapper, BasePipelineWrapper):
        raise HTTPException(status_code=404, detail=f"Pipeline '{chat_req.model}' not found or not a pipeline wrapper")

    # Execute the appropriate pipeline method (async preferred, fallback to sync)
    result = await _execute_pipeline_method(
        pipeline_wrapper=pipeline_wrapper,
        model=chat_req.model,
        messages=chat_req.messages,
        body=chat_req.model_dump(),
    )

    resp_id = f"{chat_req.model}-{uuid.uuid4()}"

    if isinstance(result, str):
        return _chat_completion_response(result, resp_id, chat_req.model)

    elif isinstance(result, Generator):
        return _create_sync_streaming_response(result, resp_id, chat_req.model)

    elif isinstance(result, AsyncGenerator):
        return _create_async_streaming_response(result, resp_id, chat_req.model)

    else:
        raise HTTPException(status_code=500, detail="Unsupported response type from pipeline")
