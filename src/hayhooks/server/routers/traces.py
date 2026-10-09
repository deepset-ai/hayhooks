from secrets import compare_digest

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from hayhooks.server.utils.live_trace_buffer import get_recent_traces
from hayhooks.settings import settings


def require_tracing_access(request: Request) -> None:
    """
    Process traces may include every user's operations; require operator access.

    Hosts may override this dependency with their verified authentication function,
    or have trusted authentication middleware set request.state.tracing_operator.
    """
    if settings.tracing_token and settings.tracing_token.get_secret_value():
        expected = "Bearer " + settings.tracing_token.get_secret_value()
        if not compare_digest(request.headers.get("Authorization", "").encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Invalid tracing credentials")
        return
    if getattr(request.state, "tracing_operator", False) is not True:
        raise HTTPException(status_code=403, detail="Authenticated tracing/operator dependency is required")


router = APIRouter(dependencies=[Depends(require_tracing_access)])


class TraceTag(BaseModel):
    key: str
    value: str


class TraceSpanNode(BaseModel):
    span_id: str
    name: str
    start_time_ms: int
    duration_ms: int
    running: bool = Field(default=False, description="True while the span is in progress (not yet finished).")
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


class TraceConfigResponse(BaseModel):
    buffer_capacity: int
    fetch_limit: int
    slow_component_ms: int
    scope: str = "Responding process; recent in-memory traces, not durable execution history"


class TracesResponse(BaseModel):
    traces: list[TraceSummary]
    has_more: bool = Field(description="Whether the snapshot omits older traces")
    config: TraceConfigResponse


@router.get(
    "",
    tags=["tracing"],
    response_model=TracesResponse,
    operation_id="tracing_traces",
    summary="List recent process traces",
    description="Returns a bounded snapshot of in-process traces and inspection settings.",
)
async def traces(limit: int | None = Query(default=None, gt=0)) -> TracesResponse:
    fetch_limit = min(100, settings.tracing_max_limit, settings.tracing_buffer_capacity)
    resolved_limit = min(settings.tracing_default_limit if limit is None else limit, fetch_limit)
    traces_data = get_recent_traces(limit=resolved_limit + 1)
    has_more = len(traces_data) > resolved_limit
    traces_data = sorted(traces_data[:resolved_limit], key=lambda trace: trace["start_time_ms"], reverse=True)
    return TracesResponse(
        traces=[TraceSummary.model_validate(trace) for trace in traces_data],
        has_more=has_more,
        config=TraceConfigResponse(
            buffer_capacity=settings.tracing_buffer_capacity,
            fetch_limit=fetch_limit,
            slow_component_ms=settings.tracing_slow_component_ms,
        ),
    )


TraceSpanNode.model_rebuild()
