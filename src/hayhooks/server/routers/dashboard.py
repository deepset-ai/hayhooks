import time

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from hayhooks.server.pipelines import registry
from hayhooks.server.utils.dashboard_traces import (
    TraceBackendError,
    fetch_jaeger_traces,
    fetch_signoz_traces,
    normalize_jaeger_trace,
    normalize_signoz_rows,
)
from hayhooks.server.utils.live_trace_buffer import clear_live_traces, get_recent_traces
from hayhooks.settings import DashboardTraceBackend, settings

router = APIRouter()


class EntrypointsResponse(BaseModel):
    entrypoints: list[str] = Field(
        description="List of deployed Hayhooks pipeline names available as entry points"
    )

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
    "/dashboard/api/traces",
    tags=["dashboard"],
    response_model=TracesResponse,
    operation_id="dashboard_traces",
    summary="List recent traces for dashboard",
    description="Returns normalized traces from the configured backend, with local buffer fallback when unavailable.",
)
async def traces(
    limit: int | None = Query(default=None, gt=0),
    since_ms: int | None = Query(default=None, ge=0),
) -> TracesResponse:
    requested_limit = settings.dashboard_trace_default_limit if limit is None else limit
    resolved_limit = min(requested_limit, settings.dashboard_trace_max_limit)

    end_time_us = int(time.time() * 1_000_000)
    if since_ms is None:
        start_time_us = max(0, end_time_us - (settings.dashboard_trace_lookback_seconds * 1_000_000))
    else:
        start_time_us = since_ms * 1000

    try:
        if settings.dashboard_trace_backend == DashboardTraceBackend.LOCAL:
            traces_data = get_recent_traces(since_ms=since_ms, limit=resolved_limit)
        elif settings.dashboard_trace_backend == DashboardTraceBackend.SIGNOZ:
            signoz_rows = fetch_signoz_traces(
                backend_url=settings.dashboard_trace_backend_url,
                service_name=settings.dashboard_trace_service_name,
                start_time_ms=int(start_time_us / 1000),
                end_time_ms=int(end_time_us / 1000),
                limit=resolved_limit,
                timeout_seconds=settings.dashboard_trace_request_timeout_seconds,
                api_key=settings.dashboard_trace_signoz_api_key,
            )
            traces_data = normalize_signoz_rows(signoz_rows)
        else:
            raw_traces = fetch_jaeger_traces(
                backend_url=settings.dashboard_trace_backend_url,
                service_name=settings.dashboard_trace_service_name,
                start_time_us=start_time_us,
                end_time_us=end_time_us,
                limit=resolved_limit,
                timeout_seconds=settings.dashboard_trace_request_timeout_seconds,
            )
            traces_data = [normalize_jaeger_trace(trace) for trace in raw_traces]
    except TraceBackendError as exc:
        fallback_traces = get_recent_traces(since_ms=since_ms, limit=resolved_limit)
        if fallback_traces:
            fallback_traces.sort(key=lambda trace: trace["start_time_ms"], reverse=True)
            return TracesResponse(traces=fallback_traces)
        detail = str(exc)
        if settings.dashboard_trace_backend == DashboardTraceBackend.JAEGER:
            detail = (
                f"{detail}. If you're using SigNoz instead of Jaeger, set "
                "HAYHOOKS_DASHBOARD_TRACE_BACKEND=signoz and "
                "HAYHOOKS_DASHBOARD_TRACE_BACKEND_URL=http://localhost:8080"
                " (plus HAYHOOKS_DASHBOARD_TRACE_SIGNOZ_API_KEY when required)."
            )
        raise HTTPException(status_code=502, detail=detail) from exc

    if not traces_data:
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
    if settings.dashboard_trace_backend == DashboardTraceBackend.LOCAL:
        message = "Cleared dashboard traces from local in-process buffer."
    else:
        message = (
            "Cleared local fallback trace buffer. External backend traces are unaffected "
            "(Jaeger/SigNoz history is not deleted)."
        )
    return ClearTracesResponse(message=message)


TraceSpanNode.model_rebuild()
