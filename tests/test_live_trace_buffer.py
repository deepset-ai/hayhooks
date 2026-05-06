from hayhooks.server.utils.live_trace_buffer import (
    clear_live_traces,
    get_recent_traces,
    get_trace_cursor_head,
    record_live_span_finish,
    record_live_span_start,
)
from hayhooks.settings import settings


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


def test_live_trace_buffer_preserves_long_tag_values():
    long_value = "x" * 400
    record_live_span_start(
        trace_id="trace-long-tag",
        span_id="root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=2_000,
        tags={
            "hayhooks.pipeline.name": "demo",
            "hayhooks.component.input_types": long_value,
        },
    )
    record_live_span_finish(trace_id="trace-long-tag", span_id="root", duration_ms=5)

    traces = get_recent_traces(since_ms=None, limit=10)

    assert len(traces) == 1
    trace = traces[0]
    trace_tag = next(tag for tag in trace["tags"] if tag["key"] == "hayhooks.component.input_types")
    span_tag = next(tag for tag in trace["root_span"]["tags"] if tag["key"] == "hayhooks.component.input_types")
    assert trace_tag["value"] == long_value
    assert span_tag["value"] == long_value


def test_live_trace_buffer_respects_configured_capacity(monkeypatch):
    monkeypatch.setattr(settings, "dashboard_trace_buffer_capacity", 2)

    for index in range(3):
        trace_id = f"trace-{index}"
        span_id = f"{trace_id}-root"
        start_time_ms = 1_000 + index
        record_live_span_start(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name="hayhooks.pipeline.run",
            start_time_ms=start_time_ms,
        )

    traces = get_recent_traces(since_ms=None, limit=10)

    assert [trace["trace_id"] for trace in traces] == ["trace-2", "trace-1"]


def test_live_trace_buffer_supports_incremental_after_seq_cursor():
    same_start_time = 12_345
    for index in range(3):
        trace_id = f"trace-{index}"
        record_live_span_start(
            trace_id=trace_id,
            span_id=f"{trace_id}-root",
            parent_span_id=None,
            operation_name="hayhooks.pipeline.run",
            start_time_ms=same_start_time,
        )

    first_batch = get_recent_traces(since_ms=None, limit=2)
    assert len(first_batch) == 2
    first_cursor = max(trace["_cursor_seq"] for trace in first_batch)

    second_batch = get_recent_traces(since_ms=None, limit=10, after_seq=first_cursor)
    assert [trace["trace_id"] for trace in second_batch] == ["trace-2"]


def test_live_trace_buffer_after_seq_returns_trace_updates():
    record_live_span_start(
        trace_id="trace-update",
        span_id="root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=10_000,
    )
    initial_batch = get_recent_traces(since_ms=None, limit=10)
    initial_cursor = max(trace["_cursor_seq"] for trace in initial_batch)

    record_live_span_finish(trace_id="trace-update", span_id="root", duration_ms=5)

    updated_batch = get_recent_traces(since_ms=None, limit=10, after_seq=initial_cursor)
    assert len(updated_batch) == 1
    assert updated_batch[0]["trace_id"] == "trace-update"
    assert updated_batch[0]["duration_ms"] == 5


def test_live_trace_buffer_cursor_head_resets_on_clear():
    record_live_span_start(
        trace_id="trace-reset",
        span_id="root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=10_000,
    )
    assert get_trace_cursor_head() > 0

    clear_live_traces()
    assert get_trace_cursor_head() == 0


def test_live_trace_buffer_spans_capped_per_trace(monkeypatch):
    monkeypatch.setattr(
        "hayhooks.server.utils.live_trace_buffer._MAX_SPANS_PER_TRACE",
        3,
        raising=False,
    )

    trace_id = "trace-span-cap"
    for i in range(5):
        record_live_span_start(
            trace_id=trace_id,
            span_id=f"span-{i}",
            parent_span_id=None if i == 0 else "span-0",
            operation_name="hayhooks.pipeline.run",
            start_time_ms=1_000 + i,
        )

    traces = get_recent_traces(since_ms=None, limit=10)
    assert len(traces) == 1
    assert traces[0]["span_count"] <= 3


def test_live_trace_buffer_tag_values_truncated(monkeypatch):
    monkeypatch.setattr(
        "hayhooks.server.utils.live_trace_buffer._MAX_TAG_VALUE_LENGTH",
        10,
        raising=False,
    )

    long_value = "a" * 200
    record_live_span_start(
        trace_id="trace-trunc",
        span_id="root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=2_000,
        tags={"hayhooks.pipeline.name": "demo", "long_key": long_value},
    )
    record_live_span_finish(trace_id="trace-trunc", span_id="root", duration_ms=5)

    traces = get_recent_traces(since_ms=None, limit=10)
    assert len(traces) == 1
    trace = traces[0]

    tag_values = [tag["value"] for tag in trace["tags"]]
    for value in tag_values:
        assert len(value) <= 10
