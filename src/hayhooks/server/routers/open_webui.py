import time
from fastapi import APIRouter
from hayhooks.server.pipelines import registry
from hayhooks.server.schema import ModelsResponse, ModelObject

router = APIRouter()


@router.get("/v1/models", response_model=ModelsResponse)
@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """
    List all hayhooks pipelines as models.

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
