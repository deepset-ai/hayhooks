from fastapi import APIRouter
from hayhooks.server.pipelines import registry

router = APIRouter()


@router.get("/v1/models")
@router.get("/models")
async def get_models():
    return {"models": registry.get_names()}
