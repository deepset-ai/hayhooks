from unittest.mock import MagicMock

import pytest
from requests import RequestException

from hayhooks.server.utils.dashboard_traces import (
    TraceBackendError,
    fetch_jaeger_traces,
    fetch_signoz_traces,
    normalize_jaeger_trace,
    normalize_signoz_rows,
)


def test_normalize_jaeger_trace_builds_tree():
    trace = {
        "traceID": "trace-42",
        "spans": [
            {
                "spanID": "root",
                "operationName": "hayhooks.pipeline.run",
                "startTime": 1_700_000_100_000_000,
                "duration": 8_000,
                "references": [],
                "tags": [{"key": "hayhooks.pipeline.name", "value": "demo"}],
            },
            {
                "spanID": "child",
                "operationName": "hayhooks.openai.run",
                "startTime": 1_700_000_100_001_000,
                "duration": 2_000,
                "references": [{"refType": "CHILD_OF", "spanID": "root"}],
                "tags": [],
            },
        ],
    }

    normalized = normalize_jaeger_trace(trace)

    assert normalized["trace_id"] == "trace-42"
    assert normalized["entrypoint"] == "demo"
    assert normalized["tags"][0] == {"key": "hayhooks.pipeline.name", "value": "demo"}
    assert normalized["span_count"] == 2
    assert normalized["duration_ms"] == 8
    assert normalized["root_span"]["span_id"] == "root"
    assert normalized["root_span"]["children"][0]["span_id"] == "child"


def test_fetch_jaeger_traces_raises_backend_error(monkeypatch):
    mock_get = MagicMock(side_effect=RequestException("jaeger down"))
    monkeypatch.setattr("hayhooks.server.utils.dashboard_traces.requests.get", mock_get)

    with pytest.raises(TraceBackendError, match="jaeger down"):
        fetch_jaeger_traces(
            backend_url="http://localhost:16686",
            service_name="hayhooks",
            start_time_us=1_700_000_000_000_000,
            end_time_us=1_700_000_001_000_000,
            limit=10,
            timeout_seconds=2.0,
        )


def test_normalize_signoz_rows_builds_tree():
    rows = [
        {
            "data": {
                "traceID": "trace-signoz",
                "spanID": "root",
                "parentSpanID": "",
                "name": "hayhooks.pipeline.run",
                "timestamp": "2026-04-21T10:00:00Z",
                "durationNano": 8_000_000,
                "hayhooks.pipeline.name": "demo",
            }
        },
        {
            "data": {
                "traceID": "trace-signoz",
                "spanID": "child",
                "parentSpanID": "root",
                "name": "hayhooks.openai.run",
                "timestamp": "2026-04-21T10:00:00.002Z",
                "durationNano": 2_000_000,
            }
        },
    ]

    traces = normalize_signoz_rows(rows)

    assert len(traces) == 1
    assert traces[0]["trace_id"] == "trace-signoz"
    assert traces[0]["entrypoint"] == "demo"
    assert traces[0]["tags"][0] == {"key": "hayhooks.pipeline.name", "value": "demo"}
    assert traces[0]["span_count"] == 2
    assert traces[0]["root_span"]["span_id"] == "root"
    assert traces[0]["root_span"]["children"][0]["span_id"] == "child"


def test_fetch_signoz_traces_requires_api_key(monkeypatch):
    response = MagicMock()
    response.status_code = 401
    response.text = '{"status":"error","error":{"message":"unauthenticated"}}'
    mock_post = MagicMock(return_value=response)
    monkeypatch.setattr("hayhooks.server.utils.dashboard_traces.requests.post", mock_post)

    with pytest.raises(TraceBackendError, match="API key required"):
        fetch_signoz_traces(
            backend_url="http://localhost:8080",
            service_name="hayhooks",
            start_time_ms=1_700_000_000_000,
            end_time_ms=1_700_000_001_000,
            limit=10,
            timeout_seconds=2.0,
            api_key="",
        )
