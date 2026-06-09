import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.live_trace_buffer import clear_live_traces, get_recent_traces
from hayhooks.server.utils.live_trace_stream import get_trace_stream_broadcaster
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
    stream_enabled: bool = Field(
        default=True,
        description="Whether the real-time SSE trace stream is available; clients fall back to polling when false.",
    )


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
        stream_enabled=settings.dashboard_stream_enabled,
    )


def read_trace_deltas(
    *,
    after_seq: int | None,
    limit: int,
    since_ms: int | None = None,
) -> TracesResponse:
    """
    Read normalized traces newer than ``after_seq`` and build the response.

    Shared by the polling endpoint (`/api/traces`) and the SSE stream so both
    use identical cursor/pagination semantics. ``next_after_seq`` advances the
    caller's cursor; ``has_more`` signals a backlog larger than ``limit``.
    """
    resolved_limit = min(limit, settings.dashboard_trace_max_limit)
    traces_data = get_recent_traces(since_ms=since_ms, limit=resolved_limit + 1, after_seq=after_seq)
    has_more = len(traces_data) > resolved_limit
    if has_more:
        traces_data = traces_data[:resolved_limit]
    traces_data.sort(key=lambda trace: trace["start_time_ms"], reverse=True)
    traces_payload = [TraceSummary.model_validate(trace) for trace in traces_data]
    next_after_seq = max((trace.get("_cursor_seq", 0) for trace in traces_data), default=after_seq or 0)
    return TracesResponse(traces=traces_payload, next_after_seq=next_after_seq, has_more=has_more)


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
    result = read_trace_deltas(after_seq=after_seq, limit=requested_limit, since_ms=since_ms)
    response.headers["X-Hayhooks-Trace-Cursor"] = str(result.next_after_seq)
    return result


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@router.get(
    "/api/traces/stream",
    tags=["dashboard"],
    operation_id="dashboard_traces_stream",
    summary="Stream recent traces in real time (SSE)",
    description=(
        "Server-Sent Events stream of trace updates. Emits a `snapshot` event on connect, then `trace` "
        "delta events as spans start/finish, with periodic keepalive comments. Payloads match `/api/traces`."
    ),
)
async def traces_stream(request: Request) -> StreamingResponse:
    raw_after_seq = request.query_params.get("after_seq")
    initial_after_seq: int | None = None
    if raw_after_seq is not None:
        with suppress(ValueError):
            parsed = int(raw_after_seq)
            initial_after_seq = parsed if parsed >= 0 else None

    broadcaster = get_trace_stream_broadcaster()
    queue = broadcaster.subscribe()
    heartbeat_seconds = settings.dashboard_stream_heartbeat_ms / 1000

    async def event_stream() -> AsyncIterator[str]:
        try:
            # On connect: replay from the client's cursor if provided (reconnect),
            # otherwise send the full current buffer. mergeTraces dedups, so a
            # repeated snapshot on reconnect is harmless.
            snapshot = read_trace_deltas(after_seq=initial_after_seq, limit=settings.dashboard_ui_list_cap)
            cursor = snapshot.next_after_seq
            yield _sse_event("snapshot", snapshot.model_dump(mode="json"))

            while True:
                if await request.is_disconnected():
                    break
                try:
                    await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                # Drain everything new since our cursor (handles bursts/backlog).
                while True:
                    delta = read_trace_deltas(after_seq=cursor, limit=settings.dashboard_trace_default_limit)
                    cursor = delta.next_after_seq
                    if delta.traces:
                        yield _sse_event("trace", delta.model_dump(mode="json"))
                    if not delta.has_more:
                        break
        finally:
            broadcaster.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


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
