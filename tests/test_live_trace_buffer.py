from hayhooks.server.utils.live_trace_buffer import clear_live_traces, get_recent_traces, record_live_span_finish, record_live_span_start


def setup_function():
    clear_live_traces()


def test_live_trace_buffer_builds_parent_child_tree():
    record_live_span_start(
        trace_id="trace-1",
        span_id="root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=1000,
        tags={"hayhooks.pipeline.name": "demo"},
    )
    record_live_span_start(
        trace_id="trace-1",
        span_id="child",
        parent_span_id="root",
        operation_name="hayhooks.openai.run",
        start_time_ms=1003,
    )
    record_live_span_finish(trace_id="trace-1", span_id="child", duration_ms=2)
    record_live_span_finish(trace_id="trace-1", span_id="root", duration_ms=9)

    traces = get_recent_traces(since_ms=None, limit=10)

    assert len(traces) == 1
    trace = traces[0]
    assert trace["trace_id"] == "trace-1"
    assert trace["entrypoint"] == "demo"
    assert trace["tags"][0] == {"key": "hayhooks.pipeline.name", "value": "demo"}
    assert trace["span_count"] == 2
    assert trace["root_span"]["span_id"] == "root"
    assert trace["root_span"]["children"][0]["span_id"] == "child"


def test_live_trace_buffer_respects_since_filter():
    record_live_span_start(
        trace_id="old",
        span_id="old-root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.deploy",
        start_time_ms=1_000,
    )
    record_live_span_finish(trace_id="old", span_id="old-root", duration_ms=2)

    record_live_span_start(
        trace_id="new",
        span_id="new-root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=10_000,
    )
    record_live_span_finish(trace_id="new", span_id="new-root", duration_ms=3)

    traces = get_recent_traces(since_ms=5_000, limit=10)

    assert len(traces) == 1
    assert traces[0]["trace_id"] == "new"
