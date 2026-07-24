from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hayhooks.durable.runtime import durable_runtime
from hayhooks.server.pipelines.registry import registry

router = APIRouter()


class StatusResponse(BaseModel):
    status: str = Field(description="The current status of the system, 'Up!' when operational")
    pipelines: list[str] = Field(description="List of all available pipeline names")
    durable: dict = Field(default_factory=dict, description="Durable worker readiness and health")

    model_config = {
        "json_schema_extra": {"description": "Response model for the system status and available pipelines"}
    }


class PipelineStatusResponse(BaseModel):
    status: str = Field(description="The current status of the pipeline, 'Up!' when operational")
    pipeline: str = Field(description="The name of the requested pipeline")

    model_config = {"json_schema_extra": {"description": "Response model for a specific pipeline status"}}


@router.get(
    "/status",
    tags=["status"],
    response_model=StatusResponse,
    operation_id="status_all",
    summary="Get status of all pipelines",
    description="Returns the system status and a list of all available pipelines.",
)
async def status_all() -> StatusResponse:
    pipelines = registry.get_names()
    durable_health = await durable_runtime.health()
    if not durable_health["healthy"]:
        raise HTTPException(status_code=503, detail={"status": "Degraded", "durable": durable_health})
    return StatusResponse(status="Up!", pipelines=pipelines, durable=durable_health)


@router.get(
    "/status/{pipeline_name}",
    tags=["status"],
    response_model=PipelineStatusResponse,
    operation_id="status_pipeline",
    summary="Get status of a specific pipeline",
    description="Returns the status of a specific pipeline. Returns 404 if the pipeline doesn't exist.",
)
async def status(pipeline_name: str) -> PipelineStatusResponse:
    if pipeline_name not in registry.get_names():
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
    deployment = durable_runtime.current_deployment(pipeline_name)
    if deployment is not None and not deployment.manager.health["healthy"]:
        raise HTTPException(status_code=503, detail=f"Pipeline '{pipeline_name}' has no live durable worker slots")
    return PipelineStatusResponse(status="Up!", pipeline=pipeline_name)
