from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class StatusResponse(BaseModel):
    status: str = Field(description="The current status of the system, 'Up!' when operational")
    pipelines: list[str] = Field(description="List of all available pipeline names")
    durable_mode: bool = Field(description="Whether the pipeline set is fixed at startup and deployment is disabled")
    durable: dict[str, object] = Field(description="Durable deployment health and bounded store counts")

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
async def status_all(request: Request) -> StatusResponse:
    state = request.app.state
    runtime = state.durable_runtime
    durable = await runtime.health() if runtime is not None else {"healthy": True, "deployments": {}}
    return StatusResponse(
        status="Up!" if durable["healthy"] else "Degraded",
        pipelines=state.pipeline_registry.get_names(),
        durable_mode=state.durable_mode,
        durable=durable,
    )


@router.get(
    "/status/{pipeline_name}",
    tags=["status"],
    response_model=PipelineStatusResponse,
    operation_id="status_pipeline",
    summary="Get status of a specific pipeline",
    description="Returns the status of a specific pipeline. Returns 404 if the pipeline doesn't exist.",
)
async def status(pipeline_name: str, request: Request) -> PipelineStatusResponse:
    if pipeline_name not in request.app.state.pipeline_registry.get_names():
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
    return PipelineStatusResponse(status="Up!", pipeline=pipeline_name)
