from pathlib import Path

from hayhooks.server.pipelines import registry
from hayhooks.server.utils.live_trace_buffer import (
    clear_live_traces,
    record_live_span_finish,
    record_live_span_start,
)
from hayhooks.settings import settings


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
            "tags": [],
            "children": [
                {
                    "span_id": "child",
                    "name": "hayhooks.openai.run",
                    "start_time_ms": start_time_ms + 4,
                    "duration_ms": 3,
                    "tags": [],
                    "children": [],
                }
            ],
        },
    }


def test_dashboard_entrypoints_empty(client):
    clear_live_traces()
    registry.clear()

    response = client.get("/dashboard/api/entrypoints")

    assert response.status_code == 200
    assert response.json() == {"entrypoints": []}


def test_dashboard_entrypoints_lists_deployed_pipelines(client, deploy_yaml_pipeline):
    clear_live_traces()
    registry.clear()
    pipeline_file = Path(__file__).parent / "test_files/yaml" / "inputs_outputs_pipeline.yml"
    pipeline_name = "dashboard_trace_pipeline"
    deploy_response = deploy_yaml_pipeline(client, pipeline_name, pipeline_file.read_text())
    assert deploy_response.status_code == 200

    response = client.get("/dashboard/api/entrypoints")

    assert response.status_code == 200
    body = response.json()
    assert "entrypoints" in body
    assert pipeline_name in body["entrypoints"]


def test_dashboard_config_returns_ui_settings(client, monkeypatch):
    monkeypatch.setattr(settings, "dashboard_ui_poll_ms", 1250)
    monkeypatch.setattr(settings, "dashboard_ui_list_cap", 40)
    monkeypatch.setattr(settings, "dashboard_ui_fetch_limit", 80)
    monkeypatch.setattr(settings, "dashboard_ui_fresh_ms", 7000)
    monkeypatch.setattr(settings, "dashboard_trace_max_limit", 60)

    response = client.get("/dashboard/api/config")

    assert response.status_code == 200
    assert response.json() == {
        "poll_ms": 1250,
        "list_cap": 40,
        "fetch_limit": 40,
        "fresh_ms": 7000,
    }


def test_dashboard_traces_returns_local_buffer_traces(client, monkeypatch):
    clear_live_traces()
    trace = _sample_trace("local-trace-1", 1_700_000_000_000)
    monkeypatch.setattr(
        "hayhooks.server.routers.dashboard.get_recent_traces",
        lambda **_kwargs: [trace],
        raising=False,
    )

    response = client.get("/dashboard/api/traces?limit=5")

    assert response.status_code == 200
    assert response.json() == {"traces": [trace]}


def test_dashboard_traces_reads_live_buffer_without_mocking(client):
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

    response = client.get("/dashboard/api/traces?limit=5")

    assert response.status_code == 200
    traces = response.json()["traces"]
    assert len(traces) == 1
    trace = traces[0]
    assert trace["trace_id"] == trace_id
    assert trace["entrypoint"] == "demo-pipeline"
    assert trace["root_span"]["name"] == "hayhooks.pipeline.run"
    assert trace["root_span"]["children"][0]["name"] == "hayhooks.openai.run"


def test_dashboard_traces_sorts_by_start_time_desc(client, monkeypatch):
    clear_live_traces()
    older_trace = _sample_trace("trace-older", 1_700_000_000_000)
    newer_trace = _sample_trace("trace-newer", 1_700_000_000_100)
    monkeypatch.setattr(
        "hayhooks.server.routers.dashboard.get_recent_traces",
        lambda **_kwargs: [older_trace, newer_trace],
        raising=False,
    )

    response = client.get("/dashboard/api/traces?limit=5")

    assert response.status_code == 200
    traces = response.json()["traces"]
    assert traces[0]["trace_id"] == "trace-newer"
    assert traces[1]["trace_id"] == "trace-older"


def test_dashboard_traces_respects_default_and_max_limit(client, monkeypatch):
    clear_live_traces()
    monkeypatch.setattr(settings, "dashboard_trace_default_limit", 11)
    monkeypatch.setattr(settings, "dashboard_trace_max_limit", 20)

    observed_limits: list[int] = []

    def _fake_get_recent_traces(*, since_ms, limit):
        assert since_ms is None
        observed_limits.append(limit)
        return []

    monkeypatch.setattr(
        "hayhooks.server.routers.dashboard.get_recent_traces",
        _fake_get_recent_traces,
        raising=False,
    )

    first_response = client.get("/dashboard/api/traces")
    second_response = client.get("/dashboard/api/traces?limit=999")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert observed_limits == [11, 20]


def test_dashboard_clear_traces_clears_local_buffer(client, monkeypatch):
    clear_live_traces()

    clear_calls: list[bool] = []

    def _fake_clear_live_traces() -> None:
        clear_calls.append(True)

    monkeypatch.setattr(
        "hayhooks.server.routers.dashboard.clear_live_traces",
        _fake_clear_live_traces,
        raising=False,
    )

    response = client.post("/dashboard/api/traces/clear")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "message": "Cleared dashboard traces from local in-process buffer.",
    }
    assert clear_calls == [True]


def test_dashboard_clear_traces_clears_live_buffer_without_mocking(client):
    clear_live_traces()
    trace_id = "integration-trace-2"
    start_time_ms = 1_700_000_000_100
    record_live_span_start(
        trace_id=trace_id,
        span_id="root",
        parent_span_id=None,
        operation_name="hayhooks.pipeline.run",
        start_time_ms=start_time_ms,
        tags={"hayhooks.pipeline.name": "demo-pipeline"},
    )
    record_live_span_finish(trace_id=trace_id, span_id="root", duration_ms=8)

    before_clear = client.get("/dashboard/api/traces")
    assert before_clear.status_code == 200
    assert len(before_clear.json()["traces"]) == 1

    clear_response = client.post("/dashboard/api/traces/clear")
    assert clear_response.status_code == 200

    after_clear = client.get("/dashboard/api/traces")
    assert after_clear.status_code == 200
    assert after_clear.json()["traces"] == []
