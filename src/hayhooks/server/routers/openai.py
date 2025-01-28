import time
import uuid
from typing import List, Literal, Union
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from hayhooks.server.pipelines import registry
from hayhooks.server.schema import ModelsResponse, ModelObject
from hayhooks.server.utils.deploy_utils import handle_pipeline_exceptions
from hayhooks.server.logger import log

router = APIRouter()


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
async def chat_endpoint(chat_req: ChatRequest) -> ChatCompletion:
    log.debug(f"registry: {registry.get_names()}")
    pipeline_wrapper = registry.get(chat_req.model)

    if not pipeline_wrapper:
        raise HTTPException(status_code=404, detail=f"Pipeline '{chat_req.model}' not found")

    if not pipeline_wrapper._is_run_chat_implemented:
        raise HTTPException(status_code=501, detail="Chat endpoint not implemented for this model")

    result = await run_in_threadpool(
        pipeline_wrapper.run_chat,
        model=chat_req.model,
        messages=chat_req.messages,
        body=chat_req.model_dump(),
    )

    resp = ChatCompletion(
        id=f"{chat_req.model}-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=chat_req.model,
        choices=[Choice(index=0, message=Message(role="assistant", content=result), finish_reason="stop")],
    )

    log.debug(f"resp: {resp.model_dump_json()}")
    return resp
