from pathlib import Path

from hayhooks.server.pipelines import registry
from hayhooks.server.utils.dashboard_traces import TraceBackendError
from hayhooks.server.utils.live_trace_buffer import clear_live_traces
from hayhooks.settings import DashboardTraceBackend, settings


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


def test_dashboard_traces_returns_normalized_tree(client, monkeypatch):
    clear_live_traces()
    monkeypatch.setattr(settings, "dashboard_trace_backend", DashboardTraceBackend.JAEGER)
    def _fake_fetch_jaeger_traces(**_kwargs):
        return [
            {
                "traceID": "trace-1",
                "spans": [
                    {
                        "spanID": "span-root",
                        "operationName": "hayhooks.pipeline.run",
                        "startTime": 1_700_000_000_000_000,
                        "duration": 6_000,
                        "references": [],
                        "tags": [{"key": "hayhooks.pipeline.name", "type": "string", "value": "demo-pipeline"}],
                    },
                    {
                        "spanID": "span-child",
                        "operationName": "hayhooks.openai.run",
                        "startTime": 1_700_000_000_001_000,
                        "duration": 2_000,
                        "references": [{"refType": "CHILD_OF", "spanID": "span-root"}],
                        "tags": [],
                    },
                ],
            }
        ]

    monkeypatch.setattr("hayhooks.server.routers.dashboard.fetch_jaeger_traces", _fake_fetch_jaeger_traces, raising=False)

    response = client.get("/dashboard/api/traces?limit=5")

    assert response.status_code == 200
    body = response.json()
    assert "traces" in body
    assert len(body["traces"]) == 1
    trace = body["traces"][0]
    assert trace["trace_id"] == "trace-1"
    assert trace["entrypoint"] == "demo-pipeline"
    assert trace["tags"][0] == {"key": "hayhooks.pipeline.name", "value": "demo-pipeline"}
    assert trace["span_count"] == 2
    assert trace["root_span"]["span_id"] == "span-root"
    assert trace["root_span"]["children"][0]["span_id"] == "span-child"


def test_dashboard_traces_returns_backend_error(client, monkeypatch):
    clear_live_traces()
    monkeypatch.setattr(settings, "dashboard_trace_backend", DashboardTraceBackend.JAEGER)

    def _fake_fetch_jaeger_traces(**_kwargs):
        raise TraceBackendError("jaeger is not reachable")

    monkeypatch.setattr("hayhooks.server.routers.dashboard.fetch_jaeger_traces", _fake_fetch_jaeger_traces, raising=False)

    response = client.get("/dashboard/api/traces")

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "jaeger is not reachable" in detail
    assert "HAYHOOKS_DASHBOARD_TRACE_BACKEND=signoz" in detail


def test_dashboard_traces_falls_back_to_local_buffer_when_backend_unreachable(client, monkeypatch):
    clear_live_traces()
    monkeypatch.setattr(settings, "dashboard_trace_backend", DashboardTraceBackend.JAEGER)
    def _fake_fetch_jaeger_traces(**_kwargs):
        raise TraceBackendError("jaeger is not reachable")

    fallback_trace = {
        "trace_id": "local-trace-1",
        "start_time_ms": 1_700_000_000_000,
        "duration_ms": 12,
        "entrypoint": "demo-pipeline",
        "tags": [],
        "span_count": 2,
        "root_span": {
            "span_id": "root",
            "name": "hayhooks.pipeline.run",
            "start_time_ms": 1_700_000_000_000,
            "duration_ms": 12,
            "tags": [],
            "children": [
                {
                    "span_id": "child",
                    "name": "hayhooks.openai.run",
                    "start_time_ms": 1_700_000_000_004,
                    "duration_ms": 3,
                    "tags": [],
                    "children": [],
                }
            ],
        },
    }

    monkeypatch.setattr("hayhooks.server.routers.dashboard.fetch_jaeger_traces", _fake_fetch_jaeger_traces, raising=False)
    monkeypatch.setattr(
        "hayhooks.server.routers.dashboard.get_recent_traces",
        lambda **_kwargs: [fallback_trace],
        raising=False,
    )

    response = client.get("/dashboard/api/traces")

    assert response.status_code == 200
    assert response.json() == {"traces": [fallback_trace]}


def test_dashboard_traces_supports_signoz_backend(client, monkeypatch):
    clear_live_traces()
    monkeypatch.setattr(settings, "dashboard_trace_backend", DashboardTraceBackend.SIGNOZ)

    signoz_rows = [
        {
            "data": {
                "traceID": "trace-signoz",
                "spanID": "span-root",
                "parentSpanID": "",
                "name": "hayhooks.pipeline.run",
                "timestamp": "2026-04-21T10:00:00Z",
                "durationNano": 6_000_000,
                "hayhooks.pipeline.name": "demo-pipeline",
            }
        },
        {
            "data": {
                "traceID": "trace-signoz",
                "spanID": "span-child",
                "parentSpanID": "span-root",
                "name": "hayhooks.openai.run",
                "timestamp": "2026-04-21T10:00:00.001Z",
                "durationNano": 2_000_000,
            }
        },
    ]
    monkeypatch.setattr(
        "hayhooks.server.routers.dashboard.fetch_signoz_traces",
        lambda **_kwargs: signoz_rows,
        raising=False,
    )

    response = client.get("/dashboard/api/traces?limit=5")

    assert response.status_code == 200
    body = response.json()
    assert len(body["traces"]) == 1
    trace = body["traces"][0]
    assert trace["trace_id"] == "trace-signoz"
    assert trace["entrypoint"] == "demo-pipeline"
    assert trace["tags"][0] == {"key": "hayhooks.pipeline.name", "value": "demo-pipeline"}
    assert trace["root_span"]["span_id"] == "span-root"
    assert trace["root_span"]["children"][0]["span_id"] == "span-child"


def test_dashboard_traces_supports_local_backend(client, monkeypatch):
    clear_live_traces()
    monkeypatch.setattr(settings, "dashboard_trace_backend", DashboardTraceBackend.LOCAL)

    fallback_trace = {
        "trace_id": "local-trace-1",
        "start_time_ms": 1_700_000_000_000,
        "duration_ms": 12,
        "entrypoint": "demo-pipeline",
        "tags": [],
        "span_count": 2,
        "root_span": {
            "span_id": "root",
            "name": "hayhooks.pipeline.run",
            "start_time_ms": 1_700_000_000_000,
            "duration_ms": 12,
            "tags": [],
            "children": [
                {
                    "span_id": "child",
                    "name": "hayhooks.openai.run",
                    "start_time_ms": 1_700_000_000_004,
                    "duration_ms": 3,
                    "tags": [],
                    "children": [],
                }
            ],
        },
    }
    monkeypatch.setattr(
        "hayhooks.server.routers.dashboard.get_recent_traces",
        lambda **_kwargs: [fallback_trace],
        raising=False,
    )

    response = client.get("/dashboard/api/traces?limit=5")

    assert response.status_code == 200
    assert response.json() == {"traces": [fallback_trace]}


def test_dashboard_clear_traces_clears_local_buffer(client, monkeypatch):
    clear_live_traces()
    monkeypatch.setattr(settings, "dashboard_trace_backend", DashboardTraceBackend.LOCAL)

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


def test_dashboard_clear_traces_keeps_external_backend_history(client, monkeypatch):
    clear_live_traces()
    monkeypatch.setattr(settings, "dashboard_trace_backend", DashboardTraceBackend.SIGNOZ)
    monkeypatch.setattr(
        "hayhooks.server.routers.dashboard.clear_live_traces",
        lambda: None,
        raising=False,
    )

    response = client.post("/dashboard/api/traces/clear")

    assert response.status_code == 200
    message = response.json()["message"]
    assert "External backend traces are unaffected" in message
