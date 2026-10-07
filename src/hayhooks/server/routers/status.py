import asyncio
import time
from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.datastructures import State

from hayhooks.server.pipelines.models import deployment_type

router = APIRouter()

# Probes within this window share one durable health read, which is bounded by the timeout.
_DURABLE_HEALTH_TTL_SECONDS = 1.0
_DURABLE_HEALTH_TIMEOUT_SECONDS = 1.0


class DeploymentSummary(BaseModel):
    name: str
    type: Literal["agent", "pipeline", "unknown"]
    execution_mode: Literal["durable", "request-bound"]
    state: Literal["Loaded", "Healthy", "Degraded"]
    description: str
    capabilities: list[str]
    revision: str | None = None
    health: dict[str, object] | None = None
    worker_scope: str = "responding process"
    store_scope: str | None = None
    execution_enumeration: bool = False


class StatusResponse(BaseModel):
    status: str = Field(description="The current status of the system, 'Up!' when operational")
    pipelines: list[str] = Field(description="List of all available pipeline names")
    durable_mode: bool = Field(description="Whether the pipeline set is fixed at startup and deployment is disabled")
    durable: dict[str, object] = Field(
        description="Durable deployment health and bounded store counts, at most one second old"
    )
    deployments: list[DeploymentSummary] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=dict)

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
        deployments=_deployment_summaries(state, durable),
        capabilities={"execution_enumeration": getattr(state, "monitoring_enabled", False)},
    )


def _deployment_summaries(state: State, durable: dict[str, object]) -> list[DeploymentSummary]:
    summaries = []
    health_by_name = cast(dict[str, dict[str, object]], durable.get("deployments", {}))
    registry = state.pipeline_registry
    for name in registry.get_names():
        wrapper = registry.get(name)
        if wrapper is None:
            continue
        metadata = registry.get_metadata(name) or {}
        is_durable = wrapper._is_run_durable_implemented or wrapper._is_run_durable_async_implemented
        health = health_by_name.get(name, {})
        flags = {
            "run_api": wrapper._is_run_api_implemented or wrapper._is_run_api_async_implemented,
            "chat_completion": wrapper._is_run_chat_completion_implemented
            or wrapper._is_run_chat_completion_async_implemented,
            "responses": wrapper._is_run_response_implemented or wrapper._is_run_response_async_implemented,
            "file_upload": wrapper._is_run_file_upload_implemented,
            "mcp_eligible": bool(metadata.get("request_model")) and not metadata.get("skip_mcp"),
            "a2a_eligible": (
                wrapper._is_run_chat_completion_implemented or wrapper._is_run_chat_completion_async_implemented
            )
            and not metadata.get("skip_a2a"),
        }
        summaries.append(
            DeploymentSummary(
                name=name,
                type=deployment_type(wrapper),
                execution_mode="durable" if is_durable else "request-bound",
                state=("Healthy" if health.get("healthy") else "Degraded") if is_durable else "Loaded",
                description=str(metadata.get("description", "")),
                capabilities=[key for key, enabled in flags.items() if enabled],
                revision=wrapper.durable_revision,
                health=health if is_durable else None,
                store_scope=(
                    "shared Redis store; revision counts use served revision"
                    if getattr(state, "durable_store", "memory") == "redis"
                    else "process-local Memory store; revision counts use served revision"
                )
                if is_durable
                else None,
                execution_enumeration=is_durable and getattr(state, "monitoring_enabled", False),
            )
        )
    return summaries


@router.get(
    "/status/{pipeline_name}",
    tags=["status"],
    response_model=PipelineStatusResponse,
    operation_id="status_pipeline",
    summary="Get status of a specific pipeline",
    description="Returns the status of a specific pipeline. Returns 404 if the pipeline doesn't exist.",
)
async def status(pipeline_name: str, request: Request) -> PipelineStatusResponse:
    if request.app.state.pipeline_registry.get(pipeline_name) is None:
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
        durable = state.pipeline_registry.durable_names()
        return {
            "healthy": False,
            "deployments": {name: {"healthy": False, "operational_error": "TimeoutError"} for name in durable},
        }
