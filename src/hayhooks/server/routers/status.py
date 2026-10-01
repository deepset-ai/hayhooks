import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.datastructures import State

from hayhooks.server.utils.module_loader import is_durable_wrapper

router = APIRouter()

# Probes within this window share one durable health read, which is bounded by the timeout.
_DURABLE_HEALTH_TTL_SECONDS = 1.0
_DURABLE_HEALTH_TIMEOUT_SECONDS = 1.0


class StatusResponse(BaseModel):
    status: str = Field(description="The current status of the system, 'Up!' when operational")
    pipelines: list[str] = Field(description="List of all available pipeline names")
    durable_mode: bool = Field(description="Whether the pipeline set is fixed at startup and deployment is disabled")
    durable: dict[str, object] = Field(
        description="Durable deployment health and bounded store counts, at most one second old"
    )

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
    durable = await _durable_health(state)
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


async def _durable_health(state: State) -> dict[str, object]:
    if state.durable_runtime is None:
        return {"healthy": True, "deployments": {}}
    now = time.monotonic()
    if state.durable_health is None or now - state.durable_health[0] >= _DURABLE_HEALTH_TTL_SECONDS:
        state.durable_health = (now, asyncio.create_task(_read_durable_health(state)))
    # Shielded: a cancelled probe must not cancel the read that concurrent probes await.
    return await asyncio.shield(state.durable_health[1])


async def _read_durable_health(state: State) -> dict[str, object]:
    try:
        return await asyncio.wait_for(state.durable_runtime.health(), _DURABLE_HEALTH_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        registry = state.pipeline_registry
        durable = [name for name in registry.get_names() if is_durable_wrapper(registry.get(name))]
        return {
            "healthy": False,
            "deployments": {name: {"healthy": False, "operational_error": "TimeoutError"} for name in durable},
        }
