import time
import uuid
from typing import Generator, List, Literal, Union
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from hayhooks.server.pipelines import registry
from hayhooks.server.utils.deploy_utils import handle_pipeline_exceptions
from hayhooks.server.logger import log

router = APIRouter()


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


@router.get("/v1/models", response_model=ModelsResponse)
@router.get("/models", response_model=ModelsResponse)
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


@router.post("/v1/chat/completions", response_model=ChatCompletion)
@router.post("/chat/completions", response_model=ChatCompletion)
@router.post("/{pipeline_name}/chat", response_model=ChatCompletion)
@handle_pipeline_exceptions()
async def chat_endpoint(chat_req: ChatRequest) -> Union[ChatCompletion, StreamingResponse]:
    pipeline_wrapper = registry.get(chat_req.model)

    if not pipeline_wrapper:
        raise HTTPException(status_code=404, detail=f"Pipeline '{chat_req.model}' not found")

    if not pipeline_wrapper._is_run_chat_completion_implemented:
        raise HTTPException(status_code=501, detail="Chat endpoint not implemented for this model")

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

        def stream_chunks() -> Generator:
            # Consume the input generator sending chunks as SSE events
            for chunk in result:
                resp = ChatCompletion(
                    id=resp_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=chat_req.model,
                    choices=[Choice(index=0, delta=Message(role="assistant", content=chunk), finish_reason=None)],
                )

                # This is the format for SSE
                # Ref: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
                yield f"data: {resp.model_dump_json()}\n\n"

            # After consuming the generator, send a final event with finish_reason "stop"
            final_resp = ChatCompletion(
                id=resp_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=chat_req.model,
                choices=[Choice(index=0, delta=Message(role="assistant", content=""), finish_reason="stop")],
            )
            yield f"data: {final_resp.model_dump_json()}\n\n"

        return StreamingResponse(stream_chunks(), media_type="text/event-stream")

    else:
        raise HTTPException(status_code=500, detail="Unsupported response type from pipeline")
