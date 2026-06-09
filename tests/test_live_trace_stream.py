import asyncio
import json

from hayhooks.server.routers.dashboard import config, traces_stream
from hayhooks.server.utils.live_trace_buffer import (
    clear_live_traces,
    get_recent_traces,
    record_live_span_finish,
    record_live_span_start,
)
from hayhooks.server.utils.live_trace_stream import _TraceStreamBroadcaster, get_trace_stream_broadcaster
from hayhooks.settings import settings


def setup_function():
    clear_live_traces()


class _FakeRequest:
    """
    Minimal stand-in for a Starlette Request to drive the SSE handler directly.

    httpx's ASGITransport buffers the whole response before returning, so it
    cannot consume an open-ended stream — we exercise the handler's
    StreamingResponse.body_iterator instead.
    """

    def __init__(self, after_seq: int | None = None) -> None:
        self.query_params: dict[str, str] = {} if after_seq is None else {"after_seq": str(after_seq)}

    async def is_disconnected(self) -> bool:
        return False


def _frame_data(frame: str) -> str:
    """Extract the JSON payload from an SSE event frame's `data:` line(s)."""
    return "".join(line[len("data:") :].lstrip() for line in frame.splitlines() if line.startswith("data:"))


async def _next_sse_event(body_iterator, timeout: float = 3.0) -> str:
    """Pull the next named SSE event frame, skipping keepalive comments."""

    async def run() -> str:
        async for frame in body_iterator:
            if frame.startswith(":"):  # keepalive comment
                continue
            return frame
        msg = "stream ended before an event was received"
        raise AssertionError(msg)

    return await asyncio.wait_for(run(), timeout=timeout)


# --- Broadcaster unit tests -------------------------------------------------


def test_notify_without_loop_is_noop():
    broadcaster = _TraceStreamBroadcaster()
    # No loop set and no subscribers: must not raise.
    broadcaster.notify()


async def test_subscribe_unsubscribe_tracks_count():
    broadcaster = _TraceStreamBroadcaster()
    broadcaster.set_loop(asyncio.get_running_loop())
    assert broadcaster.subscriber_count() == 0
    queue = broadcaster.subscribe()
    assert broadcaster.subscriber_count() == 1
    broadcaster.unsubscribe(queue)
    assert broadcaster.subscriber_count() == 0


async def test_notify_wakes_subscriber_and_coalesces():
    broadcaster = _TraceStreamBroadcaster()
    broadcaster.set_loop(asyncio.get_running_loop())
    queue = broadcaster.subscribe()
    try:
        broadcaster.notify()
        broadcaster.notify()
        broadcaster.notify()
        await asyncio.sleep(0)  # let call_soon_threadsafe callbacks run
        # Coalescing: maxsize=1 means three rapid notifies collapse to one signal.
        assert queue.qsize() == 1
    finally:
        broadcaster.unsubscribe(queue)


async def test_record_span_wakes_subscriber_via_buffer_hook():
    broadcaster = get_trace_stream_broadcaster()
    broadcaster.set_loop(asyncio.get_running_loop())
    queue = broadcaster.subscribe()
    try:
        record_live_span_start(
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            operation_name="hayhooks.pipeline.run",
            start_time_ms=1000,
        )
        await asyncio.sleep(0)
        assert not queue.empty()
    finally:
        broadcaster.unsubscribe(queue)
        broadcaster.clear_loop()


# --- SSE endpoint tests (drive the StreamingResponse directly) --------------


async def test_stream_emits_snapshot_then_trace_delta():
    broadcaster = get_trace_stream_broadcaster()
    broadcaster.set_loop(asyncio.get_running_loop())
    try:
        response = await traces_stream(_FakeRequest())
        body = response.body_iterator
        try:
            snapshot = await _next_sse_event(body)
            assert snapshot.startswith("event: snapshot")
            assert json.loads(_frame_data(snapshot))["traces"] == []  # buffer cleared in setup

            record_live_span_start(
                trace_id="t1",
                span_id="s1",
                parent_span_id=None,
                operation_name="hayhooks.pipeline.run",
                start_time_ms=1000,
                tags={"hayhooks.pipeline.name": "demo"},
            )
            record_live_span_finish(trace_id="t1", span_id="s1", duration_ms=5)

            trace_frame = await _next_sse_event(body)
            assert trace_frame.startswith("event: trace")
            payload = json.loads(_frame_data(trace_frame))
            assert "t1" in [t["trace_id"] for t in payload["traces"]]
        finally:
            await body.aclose()
    finally:
        broadcaster.clear_loop()


async def test_stream_unsubscribes_on_close():
    broadcaster = get_trace_stream_broadcaster()
    broadcaster.set_loop(asyncio.get_running_loop())
    baseline = broadcaster.subscriber_count()
    try:
        response = await traces_stream(_FakeRequest())
        body = response.body_iterator
        await _next_sse_event(body)  # snapshot — starts the generator; subscription active
        assert broadcaster.subscriber_count() == baseline + 1
        await body.aclose()  # generator's finally runs unsubscribe
        assert broadcaster.subscriber_count() == baseline
    finally:
        broadcaster.clear_loop()


def test_span_running_flag_reflects_finish_state():
    record_live_span_start(
        trace_id="t1",
        span_id="root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=1000,
    )
    record_live_span_start(
        trace_id="t1",
        span_id="child",
        parent_span_id="root",
        operation_name="haystack.component.run",
        start_time_ms=1002,
    )
    # child still running, root still running
    trace = get_recent_traces(since_ms=None, limit=10)[0]
    assert trace["root_span"]["running"] is True
    assert trace["root_span"]["children"][0]["running"] is True

    record_live_span_finish(trace_id="t1", span_id="child", duration_ms=3)
    trace = get_recent_traces(since_ms=None, limit=10)[0]
    assert trace["root_span"]["running"] is True  # root still open
    assert trace["root_span"]["children"][0]["running"] is False  # child finished

    record_live_span_finish(trace_id="t1", span_id="root", duration_ms=9)
    trace = get_recent_traces(since_ms=None, limit=10)[0]
    assert trace["root_span"]["running"] is False
    assert trace["root_span"]["children"][0]["running"] is False


async def test_config_exposes_stream_enabled(monkeypatch):
    monkeypatch.setattr(settings, "dashboard_stream_enabled", True, raising=False)
    result = await config()
    assert result.stream_enabled is True
