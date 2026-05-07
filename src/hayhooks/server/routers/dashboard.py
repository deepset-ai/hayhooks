from fastapi import APIRouter, Query, Response
from pydantic import BaseModel, Field

from hayhooks.server.pipelines.registry import registry
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
    next_after_seq: int = Field(description="Cursor value to pass as after_seq in the next request")
    has_more: bool = Field(description="Whether more traces are available beyond this page")


class ClearTracesResponse(BaseModel):
    ok: bool = True
    message: str


class DashboardUiConfigResponse(BaseModel):
    poll_ms: int
    list_cap: int
    fetch_limit: int
    fresh_ms: int
    slow_component_min_duration_ms: int
    api_base: str = Field(description="Base URL for dashboard API endpoints")


@router.get(
    "/api/entrypoints",
    tags=["dashboard"],
    response_model=EntrypointsResponse,
    operation_id="dashboard_entrypoints",
    summary="List dashboard entry points",
    description="Returns deployed Hayhooks pipelines used as dashboard entry points.",
)
async def entrypoints() -> EntrypointsResponse:
    return EntrypointsResponse(entrypoints=sorted(registry.get_names()))


@router.get(
    "/api/config",
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
    api_base = f"{settings.dashboard_path.rstrip('/')}/api"
    return DashboardUiConfigResponse(
        poll_ms=settings.dashboard_ui_poll_ms,
        list_cap=settings.dashboard_ui_list_cap,
        fetch_limit=resolved_fetch_limit,
        fresh_ms=settings.dashboard_ui_fresh_ms,
        slow_component_min_duration_ms=settings.dashboard_ui_slow_component_min_duration_ms,
        api_base=api_base,
    )


@router.get(
    "/api/traces",
    tags=["dashboard"],
    response_model=TracesResponse,
    operation_id="dashboard_traces",
    summary="List recent traces for dashboard",
    description="Returns normalized traces captured by the in-process live trace buffer.",
)
async def traces(
    response: Response,
    limit: int | None = Query(default=None, gt=0),
    since_ms: int | None = Query(default=None, ge=0),
    after_seq: int | None = Query(default=None, ge=0),
) -> TracesResponse:
    requested_limit = settings.dashboard_trace_default_limit if limit is None else limit
    resolved_limit = min(requested_limit, settings.dashboard_trace_max_limit)
    traces_data = get_recent_traces(since_ms=since_ms, limit=resolved_limit + 1, after_seq=after_seq)
    has_more = len(traces_data) > resolved_limit
    if has_more:
        traces_data = traces_data[:resolved_limit]
    traces_data.sort(key=lambda trace: trace["start_time_ms"], reverse=True)
    traces_payload = [TraceSummary.model_validate(trace) for trace in traces_data]
    next_after_seq = max((trace.get("_cursor_seq", 0) for trace in traces_data), default=after_seq or 0)
    response.headers["X-Hayhooks-Trace-Cursor"] = str(next_after_seq)
    return TracesResponse(traces=traces_payload, next_after_seq=next_after_seq, has_more=has_more)


@router.post(
    "/api/traces/clear",
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
