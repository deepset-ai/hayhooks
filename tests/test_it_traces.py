import pytest

from hayhooks.server.routers.traces import require_tracing_access
from hayhooks.server.utils.live_trace_buffer import clear_live_traces, record_live_span_finish, record_live_span_start
from hayhooks.settings import settings
from hayhooks.tui.client import MAX_TRACE_DEPTH, DashboardClient


@pytest.fixture(autouse=True)
def tracing_access(client):
    # Some CLI tests explicitly disable tracing; support either shared-app configuration.
    from hayhooks.server.routers.traces import router

    if not any(getattr(route, "path", None) == "/traces" for route in client.app.routes):
        client.app.include_router(router, prefix="/traces")
    client.app.dependency_overrides[require_tracing_access] = lambda: None
    yield
    client.app.dependency_overrides.pop(require_tracing_access, None)


def _sample_trace(trace_id: str, start_time_ms: int) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "start_time_ms": start_time_ms,
        "duration_ms": 12,
        "entrypoint": "demo-pipeline",
        "tags": [],
        "span_count": 2,
        "root_span": {
            "span_id": "root",
            "name": "hayhooks.pipeline.run",
            "start_time_ms": start_time_ms,
            "duration_ms": 12,
            "running": False,
            "tags": [],
            "children": [
                {
                    "span_id": "child",
                    "name": "hayhooks.openai.run",
                    "start_time_ms": start_time_ms + 4,
                    "duration_ms": 3,
                    "running": False,
                    "tags": [],
                    "children": [],
                }
            ],
        },
    }


def test_tracing_snapshot_includes_buffer_settings(client, monkeypatch):
    monkeypatch.setattr(settings, "tracing_buffer_capacity", 40)
    monkeypatch.setattr(settings, "tracing_max_limit", 60)
    monkeypatch.setattr(settings, "tracing_slow_component_ms", 1800)
    body = client.get("/traces").json()["config"]
    assert body["buffer_capacity"] == 40
    assert body["fetch_limit"] == 40
    assert body["slow_component_ms"] == 1800
    assert "Responding process" in body["scope"]


def test_tracing_traces_returns_local_buffer_traces(client, monkeypatch):
    clear_live_traces()
    trace = _sample_trace("local-trace-1", 1_700_000_000_000)
    monkeypatch.setattr(
        "hayhooks.server.routers.traces.get_recent_traces",
        lambda **_kwargs: [trace],
        raising=False,
    )

    response = client.get("/traces?limit=5")

    assert response.status_code == 200
    body = response.json()
    assert body["traces"] == [trace]
    assert "next_after_seq" not in body
    assert "has_more" in body


def test_tracing_traces_reads_live_buffer_without_mocking(client):
    clear_live_traces()
    trace_id = "integration-trace-1"
    start_time_ms = 1_700_000_000_000

    record_live_span_start(
        trace_id=trace_id,
        span_id="root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=start_time_ms,
        tags={"hayhooks.pipeline.name": "demo-pipeline"},
    )
    record_live_span_start(
        trace_id=trace_id,
        span_id="child",
        parent_span_id="root",
        operation_name="hayhooks.openai.run",
        start_time_ms=start_time_ms + 4,
    )
    record_live_span_finish(trace_id=trace_id, span_id="child", duration_ms=3)
    record_live_span_finish(trace_id=trace_id, span_id="root", duration_ms=12)

    response = client.get("/traces?limit=5")

    assert response.status_code == 200
    traces = response.json()["traces"]
    assert len(traces) == 1
    trace = traces[0]
    assert trace["trace_id"] == trace_id
    assert trace["entrypoint"] == "demo-pipeline"
    assert trace["root_span"]["name"] == "hayhooks.pipeline.run"
    assert trace["root_span"]["children"][0]["name"] == "hayhooks.openai.run"


def test_tracing_traces_sorts_by_start_time_desc(client, monkeypatch):
    clear_live_traces()
    older_trace = _sample_trace("trace-older", 1_700_000_000_000)
    newer_trace = _sample_trace("trace-newer", 1_700_000_000_100)
    monkeypatch.setattr(
        "hayhooks.server.routers.traces.get_recent_traces",
        lambda **_kwargs: [older_trace, newer_trace],
        raising=False,
    )

    response = client.get("/traces?limit=5")

    assert response.status_code == 200
    traces = response.json()["traces"]
    assert traces[0]["trace_id"] == "trace-newer"
    assert traces[1]["trace_id"] == "trace-older"


def test_tracing_traces_respects_default_and_max_limit(client, monkeypatch):
    clear_live_traces()
    monkeypatch.setattr(settings, "tracing_default_limit", 11)
    monkeypatch.setattr(settings, "tracing_max_limit", 20)

    observed_limits: list[int] = []

    def _fake_get_recent_traces(*, limit):
        observed_limits.append(limit)
        return []

    monkeypatch.setattr(
        "hayhooks.server.routers.traces.get_recent_traces",
        _fake_get_recent_traces,
        raising=False,
    )

    first_response = client.get("/traces")
    second_response = client.get("/traces?limit=999")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert observed_limits == [12, 21]


def test_tracing_traces_has_more_when_beyond_limit(client, monkeypatch):
    clear_live_traces()

    traces = [_sample_trace(f"trace-{i}", 1_700_000_000_000 + i) for i in range(10)]

    limit_arg: list[int] = []

    def _fake_get_recent_traces(*, limit):
        limit_arg.append(limit)
        return traces[:limit]

    monkeypatch.setattr(
        "hayhooks.server.routers.traces.get_recent_traces",
        _fake_get_recent_traces,
        raising=False,
    )

    response = client.get("/traces?limit=4")

    assert response.status_code == 200
    body = response.json()
    assert len(body["traces"]) == 4
    assert body["has_more"] is True
    assert limit_arg[-1] == 5


def test_tracing_traces_has_more_false_when_no_more(client, monkeypatch):
    clear_live_traces()

    traces = [_sample_trace(f"trace-{i}", 1_700_000_000_000 + i) for i in range(2)]

    def _fake_get_recent_traces(*, limit):
        return traces[:limit]

    monkeypatch.setattr(
        "hayhooks.server.routers.traces.get_recent_traces",
        _fake_get_recent_traces,
        raising=False,
    )

    response = client.get("/traces?limit=5")

    assert response.status_code == 200
    body = response.json()
    assert len(body["traces"]) == 2
    assert body["has_more"] is False


@pytest.mark.parametrize("limit", ["0", "-1", "invalid"])
def test_trace_snapshot_rejects_invalid_limits(client, limit):
    assert client.get("/traces", params={"limit": limit}).status_code == 422


def test_trace_snapshot_is_bounded_even_when_settings_allow_more(client, monkeypatch):
    monkeypatch.setattr(settings, "tracing_max_limit", 500)
    limits = []
    monkeypatch.setattr("hayhooks.server.routers.traces.get_recent_traces", lambda *, limit: limits.append(limit) or [])
    body = client.get("/traces?limit=999").json()
    assert limits == [101]
    assert body["config"]["fetch_limit"] == 100


@pytest.mark.parametrize("depth", [64, 65, 499])
def test_deep_trace_snapshot_remains_usable_by_dashboard(client, monkeypatch, depth):
    clear_live_traces()
    for index in range(depth + 1):
        record_live_span_start(
            trace_id="deep",
            span_id=str(index),
            parent_span_id=str(index - 1) if index else None,
            operation_name="nested",
            start_time_ms=1000 + index,
        )
    record_live_span_start(
        trace_id="healthy",
        span_id="root",
        parent_span_id=None,
        operation_name="normal",
        start_time_ms=2000,
    )
    dashboard = DashboardClient("http://test")
    monkeypatch.setattr(dashboard, "get", lambda path, **params: client.get(f"/{path}", params=params).json())
    try:
        snapshot = dashboard.traces()
    finally:
        dashboard.close()
    assert {trace["trace_id"] for trace in snapshot["traces"]} == {"deep", "healthy"}
    trace = next(trace for trace in snapshot["traces"] if trace["trace_id"] == "deep")
    assert trace["span_count"] == depth + 1
    node, shown = trace["root_span"], 1
    while node["children"]:
        node = node["children"][0]
        shown += 1
    assert shown == min(depth + 1, MAX_TRACE_DEPTH + 1)
