from hayhooks.server.utils.live_trace_buffer import (
    clear_live_traces,
    get_recent_traces,
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

    traces = get_recent_traces(limit=10)

    assert len(traces) == 1
    trace = traces[0]
    assert trace["trace_id"] == "trace-1"
    assert trace["entrypoint"] == "demo"
    assert trace["tags"][0] == {"key": "hayhooks.pipeline.name", "value": "demo"}
    assert trace["span_count"] == 2
    assert trace["root_span"]["span_id"] == "root"
    assert trace["root_span"]["children"][0]["span_id"] == "child"


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

    traces = get_recent_traces(limit=10)

    assert len(traces) == 1
    trace = traces[0]
    trace_tag = next(tag for tag in trace["tags"] if tag["key"] == "hayhooks.component.input_types")
    span_tag = next(tag for tag in trace["root_span"]["tags"] if tag["key"] == "hayhooks.component.input_types")
    assert trace_tag["value"] == long_value
    assert span_tag["value"] == long_value


def test_live_trace_buffer_respects_configured_capacity(monkeypatch):
    monkeypatch.setattr(settings, "tracing_buffer_capacity", 2)

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

    traces = get_recent_traces(limit=10)

    assert [trace["trace_id"] for trace in traces] == ["trace-2", "trace-1"]


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

    traces = get_recent_traces(limit=10)
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

    traces = get_recent_traces(limit=10)
    assert len(traces) == 1
    trace = traces[0]

    tag_values = [tag["value"] for tag in trace["tags"]]
    for value in tag_values:
        assert len(value) <= 10


def test_live_trace_buffer_structured_tag_values_truncated(monkeypatch):
    monkeypatch.setattr(
        "hayhooks.server.utils.live_trace_buffer._MAX_TAG_VALUE_LENGTH",
        20,
        raising=False,
    )

    record_live_span_start(
        trace_id="trace-structured-trunc",
        span_id="root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=2_000,
        tags={
            "hayhooks.pipeline.name": "demo",
            "structured": {"payload": "x" * 200},
        },
    )
    record_live_span_finish(trace_id="trace-structured-trunc", span_id="root", duration_ms=5)

    traces = get_recent_traces(limit=10)
    assert len(traces) == 1
    trace = traces[0]

    structured_trace_tag = next(tag for tag in trace["tags"] if tag["key"] == "structured")
    structured_span_tag = next(tag for tag in trace["root_span"]["tags"] if tag["key"] == "structured")
    assert len(structured_trace_tag["value"]) <= 20
    assert len(structured_span_tag["value"]) <= 20


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
    trace = get_recent_traces(limit=10)[0]
    assert trace["root_span"]["running"] is True
    assert trace["root_span"]["children"][0]["running"] is True

    record_live_span_finish(trace_id="t1", span_id="child", duration_ms=3)
    trace = get_recent_traces(limit=10)[0]
    assert trace["root_span"]["running"] is True  # root still open
    assert trace["root_span"]["children"][0]["running"] is False  # child finished

    record_live_span_finish(trace_id="t1", span_id="root", duration_ms=9)
    trace = get_recent_traces(limit=10)[0]
    assert trace["root_span"]["running"] is False
    assert trace["root_span"]["children"][0]["running"] is False


def test_snapshot_returns_updated_old_traces_and_clears():
    for trace_id, start in [("old", 1000), ("new", 2000)]:
        record_live_span_start(
            trace_id=trace_id,
            span_id="root",
            parent_span_id=None,
            operation_name="hayhooks.pipeline.run",
            start_time_ms=start,
        )
    assert get_recent_traces(limit=1)[0]["trace_id"] == "new"
    record_live_span_finish(trace_id="old", span_id="root", duration_ms=5)
    snapshot = get_recent_traces(limit=1)
    assert snapshot[0]["trace_id"] == "old"
    assert snapshot[0]["duration_ms"] == 5
    clear_live_traces()
    assert get_recent_traces(limit=10) == []
