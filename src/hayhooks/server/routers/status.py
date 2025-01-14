from fastapi import APIRouter, HTTPException
from hayhooks.server.pipelines import registry

router = APIRouter()


@router.get("/status", tags=["status"])
async def status_all():
    pipelines = registry.get_names()
    return {"status": "Up!", "pipelines": pipelines}


@router.get("/status/{pipeline_name}", tags=["status"])
async def status(pipeline_name: str):
    if pipeline_name not in registry.get_names():
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
    return {"status": "Up!", "pipeline": pipeline_name}
