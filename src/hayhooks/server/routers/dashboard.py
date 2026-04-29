from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from hayhooks.server.pipelines import registry
from hayhooks.server.utils.live_trace_buffer import clear_live_traces, get_recent_traces
from hayhooks.settings import settings

router = APIRouter()


class EntrypointsResponse(BaseModel):
    entrypoints: list[str] = Field(description="List of deployed Hayhooks pipeline names available as entry points")

    model_config = {"json_schema_extra": {"description": "Response model for dashboard entry points"}}


class TraceTag(BaseModel):
    key: str
    value: str


class TraceSpanNode(BaseModel):
    span_id: str
    name: str
    start_time_ms: int
    duration_ms: int
    tags: list[TraceTag] = Field(default_factory=list)
    children: list["TraceSpanNode"] = Field(default_factory=list)


class TraceSummary(BaseModel):
    trace_id: str
    start_time_ms: int
    duration_ms: int
    entrypoint: str | None = None
    tags: list[TraceTag] = Field(default_factory=list)
    span_count: int
    root_span: TraceSpanNode


class TracesResponse(BaseModel):
    traces: list[TraceSummary]


class ClearTracesResponse(BaseModel):
    ok: bool = True
    message: str


class DashboardUiConfigResponse(BaseModel):
    poll_ms: int
    list_cap: int
    fetch_limit: int
    fresh_ms: int
    slow_component_min_duration_ms: int


@router.get(
    "/dashboard/api/entrypoints",
    tags=["dashboard"],
    response_model=EntrypointsResponse,
    operation_id="dashboard_entrypoints",
    summary="List dashboard entry points",
    description="Returns deployed Hayhooks pipelines used as dashboard entry points.",
)
async def entrypoints() -> EntrypointsResponse:
    return EntrypointsResponse(entrypoints=sorted(registry.get_names()))


@router.get(
    "/dashboard/api/config",
    tags=["dashboard"],
    response_model=DashboardUiConfigResponse,
    operation_id="dashboard_config",
    summary="Get dashboard UI config",
    description="Returns dashboard UI polling and list configuration derived from Hayhooks settings.",
)
async def config() -> DashboardUiConfigResponse:
    resolved_fetch_limit = min(
        settings.dashboard_ui_fetch_limit,
        settings.dashboard_trace_max_limit,
        settings.dashboard_ui_list_cap,
    )
    return DashboardUiConfigResponse(
        poll_ms=settings.dashboard_ui_poll_ms,
        list_cap=settings.dashboard_ui_list_cap,
        fetch_limit=resolved_fetch_limit,
        fresh_ms=settings.dashboard_ui_fresh_ms,
        slow_component_min_duration_ms=settings.dashboard_ui_slow_component_min_duration_ms,
    )


@router.get(
    "/dashboard/api/traces",
    tags=["dashboard"],
    response_model=TracesResponse,
    operation_id="dashboard_traces",
    summary="List recent traces for dashboard",
    description="Returns normalized traces captured by the in-process live trace buffer.",
)
async def traces(
    limit: int | None = Query(default=None, gt=0),
    since_ms: int | None = Query(default=None, ge=0),
) -> TracesResponse:
    requested_limit = settings.dashboard_trace_default_limit if limit is None else limit
    resolved_limit = min(requested_limit, settings.dashboard_trace_max_limit)
    traces_data = get_recent_traces(since_ms=since_ms, limit=resolved_limit)
    traces_data.sort(key=lambda trace: trace["start_time_ms"], reverse=True)
    return TracesResponse(traces=traces_data)


@router.post(
    "/dashboard/api/traces/clear",
    tags=["dashboard"],
    response_model=ClearTracesResponse,
    operation_id="dashboard_clear_traces",
    summary="Clear dashboard traces",
    description="Clears the in-process dashboard trace buffer and returns clear status.",
)
async def clear_traces() -> ClearTracesResponse:
    clear_live_traces()
    return ClearTracesResponse(message="Cleared dashboard traces from local in-process buffer.")


TraceSpanNode.model_rebuild()
