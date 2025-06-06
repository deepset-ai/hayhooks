import time
import uuid
from typing import Generator, List, Literal, Union, AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from hayhooks.server.pipelines import registry
from hayhooks.server.utils.deploy_utils import handle_pipeline_exceptions
from hayhooks.server.logger import log

router = APIRouter()


def _create_sse_data_msg(
    resp_id: str, model_name: str, chunk_content: str = "", finish_reason: Union[Literal["stop"], None] = None
) -> str:
    """Helper function to create a ChatCompletion chunk and format it as an SSE string."""
    response = ChatCompletion(
        id=resp_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model_name,
        choices=[Choice(index=0, delta=Message(role="assistant", content=chunk_content), finish_reason=finish_reason)],
    )
    return f"data: {response.model_dump_json()}\n\n"


class ModelObject(BaseModel):
    id: str
    name: str
    object: Literal["model"]
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    data: List[ModelObject]
    object: Literal["list"]


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatRequest(OpenAIBaseModel):
    model: str
    messages: List[dict]
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
    choices: List[Choice]


MODELS_PARAMS: dict = {
    "response_model": ModelsResponse,
    "tags": ["openai"],
}


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
    pipeline_wrapper = registry.get(chat_req.model)

    if not pipeline_wrapper:
        raise HTTPException(status_code=404, detail=f"Pipeline '{chat_req.model}' not found")

    # Check if either sync or async chat completion is implemented
    sync_implemented = pipeline_wrapper._is_run_chat_completion_implemented
    async_implemented = pipeline_wrapper._is_run_chat_completion_async_implemented

    if not sync_implemented and not async_implemented:
        raise HTTPException(status_code=501, detail="Chat endpoint not implemented for this model")

    # Determine which run_chat_completion method to use (prefer async if available)
    if async_implemented:
        log.debug(f"Using run_chat_completion_async for model: {chat_req.model}")
        result = await pipeline_wrapper.run_chat_completion_async(
            model=chat_req.model,
            messages=chat_req.messages,
            body=chat_req.model_dump(),
        )
    elif sync_implemented:
        log.debug(f"Using run_chat_completion (sync) for model: {chat_req.model}")
        result = await run_in_threadpool(
            pipeline_wrapper.run_chat_completion,
            model=chat_req.model,
            messages=chat_req.messages,
            body=chat_req.model_dump(),
        )

    resp_id = f"{chat_req.model}-{uuid.uuid4()}"

    if isinstance(result, str):
        # If the pipeline returns a string, we can directly return a ChatCompletion object

        resp = ChatCompletion(
            id=resp_id,
            object="chat.completion",
            created=int(time.time()),
            model=chat_req.model,
            choices=[Choice(index=0, message=Message(role="assistant", content=result), finish_reason="stop")],
        )

        log.debug(f"resp: {resp.model_dump_json()}")
        return resp

    elif isinstance(result, Generator):
        # If the pipeline returns a generator, we need to stream the chunks as SSE events
        def stream_chunks() -> Generator[str, None, None]:
            # Consume the input generator sending chunks as SSE events
            for chunk in result:
                yield _create_sse_data_msg(resp_id=resp_id, model_name=chat_req.model, chunk_content=chunk)

            # After consuming the generator, send a final event with finish_reason "stop"
            yield _create_sse_data_msg(resp_id=resp_id, model_name=chat_req.model, finish_reason="stop")

        return StreamingResponse(stream_chunks(), media_type="text/event-stream")

    elif isinstance(result, AsyncGenerator):
        # If the pipeline returns an async generator, we need to stream the chunks as SSE events
        async def stream_chunks_async() -> AsyncGenerator[str, None]:
            async for chunk in result:
                yield _create_sse_data_msg(resp_id=resp_id, model_name=chat_req.model, chunk_content=chunk)

            yield _create_sse_data_msg(resp_id=resp_id, model_name=chat_req.model, finish_reason="stop")

        return StreamingResponse(stream_chunks_async(), media_type="text/event-stream")

    else:
        raise HTTPException(status_code=500, detail="Unsupported response type from pipeline")
