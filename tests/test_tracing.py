from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.applications import Starlette

from hayhooks.server.app import create_app
from hayhooks.server.logger import normalize_trace_correlation_data
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.tracing import (
    SPAN_OPENAI_RUN,
    SPAN_PIPELINE_DEPLOY,
    SPAN_PIPELINE_DEPLOY_COMMIT,
    SPAN_PIPELINE_DEPLOY_PREPARE,
    SPAN_PIPELINE_RUN,
    SPAN_PIPELINE_STARTUP_DEPLOY,
    SPAN_PIPELINE_UNDEPLOY,
    _OTLP_HTTP_PROTOBUF,
    _build_otlp_span_exporter,
    _normalize_otlp_protocol,
    configure_tracing,
    instrument_fastapi_app,
    instrument_starlette_app,
)
from hayhooks.server.utils.deploy_utils import deploy_pipeline_yaml, undeploy_pipeline
from hayhooks.settings import StartupDeployStrategy, settings

SAMPLE_YAML = (Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml").read_text()

CHAT_PIPELINE_DIR = Path(__file__).parent / "test_files/files/chat_with_website"
CHAT_PIPELINE_FILES = {
    "pipeline_wrapper.py": (CHAT_PIPELINE_DIR / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (CHAT_PIPELINE_DIR / "chat_with_website.yml").read_text(),
}

CHAT_STREAMING_PIPELINE_DIR = Path(__file__).parent / "test_files/files/chat_with_website_streaming"
CHAT_STREAMING_PIPELINE_FILES = {
    "pipeline_wrapper.py": (CHAT_STREAMING_PIPELINE_DIR / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (CHAT_STREAMING_PIPELINE_DIR / "chat_with_website.yml").read_text(),
}

RUN_API_STREAMING_PIPELINE_DIR = Path(__file__).parent / "test_files/files/run_api_streaming"
RUN_API_STREAMING_PIPELINE_FILES = {
    "pipeline_wrapper.py": (RUN_API_STREAMING_PIPELINE_DIR / "pipeline_wrapper.py").read_text(),
}

BROKEN_STREAMING_PIPELINE_FILES = {
    "pipeline_wrapper.py": """
from collections.abc import Generator

from haystack.dataclasses import StreamingChunk

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    def run_api(self, query: str) -> Generator[StreamingChunk, None, None]:
        def stream() -> Generator[StreamingChunk, None, None]:
            yield StreamingChunk(content=f"{query} ")
            raise RuntimeError("run_api stream exploded")

        return stream()

    def run_chat_completion(
        self, model: str, messages: list[dict], body: dict
    ) -> Generator[StreamingChunk, None, None]:
        def stream() -> Generator[StreamingChunk, None, None]:
            yield StreamingChunk(content="first ")
            raise RuntimeError("chat stream exploded")

        return stream()
""",
}


def test_normalize_trace_correlation_data_formats_identifiers():
    normalized = normalize_trace_correlation_data({"trace_id": 10, "span_id": 11, "custom": "value"})
    assert normalized == {
        "trace_id": "0000000000000000000000000000000a",
        "span_id": "000000000000000b",
        "custom": "value",
    }


def test_instrument_fastapi_app_is_idempotent(monkeypatch):
    class FakeFastAPIInstrumentor:
        calls = 0

        @classmethod
        def instrument_app(cls, app):  # noqa: ARG003
            cls.calls += 1

    monkeypatch.setattr("hayhooks.server.tracing._load_fastapi_instrumentor", lambda: FakeFastAPIInstrumentor)

    app = FastAPI()
    assert instrument_fastapi_app(app) is True
    assert instrument_fastapi_app(app) is True
    assert FakeFastAPIInstrumentor.calls == 1


def test_instrument_fastapi_app_passes_excluded_spans(monkeypatch):
    recorded: dict[str, object] = {}

    class FakeFastAPIInstrumentor:
        @classmethod
        def instrument_app(cls, app, **kwargs):  # noqa: ARG003
            recorded["kwargs"] = kwargs

    monkeypatch.setattr("hayhooks.server.tracing._load_fastapi_instrumentor", lambda: FakeFastAPIInstrumentor)
    monkeypatch.setattr(settings, "tracing_excluded_spans", ["send"])

    app = FastAPI()
    assert instrument_fastapi_app(app) is True
    assert recorded["kwargs"] == {"exclude_spans": ["send"]}


def test_instrument_starlette_app_noop_when_dependency_missing(monkeypatch):
    monkeypatch.setattr("hayhooks.server.tracing._load_starlette_instrumentor", lambda: None)
    app = Starlette()
    assert instrument_starlette_app(app) is False


def test_instrument_starlette_app_passes_excluded_spans(monkeypatch):
    recorded: dict[str, object] = {}

    class FakeStarletteInstrumentor:
        @classmethod
        def instrument_app(cls, app, **kwargs):  # noqa: ARG003
            recorded["kwargs"] = kwargs

    monkeypatch.setattr("hayhooks.server.tracing._load_starlette_instrumentor", lambda: FakeStarletteInstrumentor)
    monkeypatch.setattr(settings, "tracing_excluded_spans", ["send", "receive"])

    app = Starlette()
    assert instrument_starlette_app(app) is True
    assert recorded["kwargs"] == {"exclude_spans": ["send", "receive"]}


def test_normalize_otlp_protocol():
    assert _normalize_otlp_protocol(None) == "http/protobuf"
    assert _normalize_otlp_protocol("http") == "http/protobuf"
    assert _normalize_otlp_protocol("http/protobuf") == "http/protobuf"
    assert _normalize_otlp_protocol("grpc") == "grpc"
    assert _normalize_otlp_protocol("unsupported") is None


def test_build_otlp_http_exporter_uses_env_endpoint_resolution(monkeypatch):
    class _FakeLazyImport:
        @staticmethod
        def check() -> None:
            return None

    class _FakeExporter:
        kwargs: dict = {}

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("hayhooks.server.tracing.otlp_http_exporter_import", _FakeLazyImport())
    monkeypatch.setattr("hayhooks.server.tracing.OTLPHTTPSpanExporter", _FakeExporter, raising=False)

    exporter = _build_otlp_span_exporter(_OTLP_HTTP_PROTOBUF)

    assert isinstance(exporter, _FakeExporter)
    assert exporter.kwargs == {}


def test_configure_tracing_noop_when_already_enabled(monkeypatch):
    calls = {"auto": 0, "bootstrap": 0}
    monkeypatch.setattr("hayhooks.server.tracing.is_tracing_enabled", lambda: True)

    def fake_auto_enable():
        calls["auto"] += 1

    def fake_bootstrap():
        calls["bootstrap"] += 1
        return True

    monkeypatch.setattr("hayhooks.server.tracing.auto_enable_tracing", fake_auto_enable)
    monkeypatch.setattr("hayhooks.server.tracing._configure_otel_tracer_from_env", fake_bootstrap)

    assert configure_tracing() is True
    assert calls == {"auto": 0, "bootstrap": 0}


def test_configure_tracing_uses_haystack_auto_enable(monkeypatch):
    state = {"enabled": False}
    calls = {"bootstrap": 0}

    def fake_is_tracing_enabled():
        return state["enabled"]

    def fake_auto_enable():
        state["enabled"] = True

    def fake_bootstrap():
        calls["bootstrap"] += 1
        return True

    monkeypatch.setattr("hayhooks.server.tracing.is_tracing_enabled", fake_is_tracing_enabled)
    monkeypatch.setattr("hayhooks.server.tracing.auto_enable_tracing", fake_auto_enable)
    monkeypatch.setattr("hayhooks.server.tracing._configure_otel_tracer_from_env", fake_bootstrap)

    assert configure_tracing() is True
    assert calls["bootstrap"] == 0


def test_configure_tracing_falls_back_to_otlp_bootstrap(monkeypatch):
    calls = {"auto": 0, "bootstrap": 0}

    monkeypatch.setattr("hayhooks.server.tracing.is_tracing_enabled", lambda: False)

    def fake_auto_enable():
        calls["auto"] += 1

    def fake_bootstrap():
        calls["bootstrap"] += 1
        return True

    monkeypatch.setattr("hayhooks.server.tracing.auto_enable_tracing", fake_auto_enable)
    monkeypatch.setattr("hayhooks.server.tracing._configure_otel_tracer_from_env", fake_bootstrap)

    assert configure_tracing() is True
    assert calls == {"auto": 1, "bootstrap": 1}


def test_deploy_and_undeploy_emit_lifecycle_spans(recording_tracer):
    registry.clear()

    deploy_pipeline_yaml(
        pipeline_name="trace_lifecycle_pipeline",
        source_code=SAMPLE_YAML,
        options={"save_file": False},
    )
    undeploy_pipeline("trace_lifecycle_pipeline")

    span_names = [span.operation_name for span in recording_tracer.spans]

    assert SPAN_PIPELINE_DEPLOY in span_names
    assert SPAN_PIPELINE_DEPLOY_PREPARE in span_names
    assert SPAN_PIPELINE_DEPLOY_COMMIT in span_names
    assert SPAN_PIPELINE_UNDEPLOY in span_names


def test_run_endpoint_emits_pipeline_run_span(client, deploy_yaml_pipeline, recording_tracer):
    registry.clear()
    pipeline_name = "trace_run_pipeline"

    deploy_response = deploy_yaml_pipeline(client, pipeline_name, SAMPLE_YAML)
    assert deploy_response.status_code == 200

    run_response = client.post(f"/{pipeline_name}/run", json={"value": 3})
    assert run_response.status_code == 200

    run_spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_PIPELINE_RUN]
    assert run_spans
    assert any(span.tags.get("hayhooks.pipeline.name") == pipeline_name for span in run_spans)

    client.post(f"/undeploy/{pipeline_name}")


def test_openai_chat_completion_emits_openai_run_span(client, deploy_files, recording_tracer):
    registry.clear()
    pipeline_name = "trace_openai_pipeline"

    deploy_response = deploy_files(client, pipeline_name, CHAT_PIPELINE_FILES)
    assert deploy_response.status_code == 200

    response = client.post(
        "/chat/completions",
        json={
            "model": pipeline_name,
            "messages": [{"role": "user", "content": "What is Redis?"}],
            "stream": False,
        },
    )
    assert response.status_code == 200

    openai_spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_OPENAI_RUN]
    assert openai_spans
    assert any(
        span.tags.get("hayhooks.pipeline.name") == pipeline_name
        and span.tags.get("hayhooks.openai.operation") == "run_chat_completion"
        for span in openai_spans
    )

    client.post(f"/undeploy/{pipeline_name}")


def test_run_endpoint_streaming_span_tags_reflect_actual_streaming(
    client, deploy_yaml_pipeline, deploy_files, recording_tracer
):
    registry.clear()

    non_streaming_pipeline = "trace_run_non_streaming"
    streaming_pipeline = "trace_run_streaming"

    assert deploy_yaml_pipeline(client, non_streaming_pipeline, SAMPLE_YAML).status_code == 200
    assert deploy_files(client, streaming_pipeline, RUN_API_STREAMING_PIPELINE_FILES).status_code == 200

    non_streaming_response = client.post(f"/{non_streaming_pipeline}/run", json={"value": 3})
    assert non_streaming_response.status_code == 200

    with client.stream("POST", f"/{streaming_pipeline}/run", json={"query": "stream me"}) as stream_response:
        assert stream_response.status_code == 200
        list(stream_response.iter_text())

    run_spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_PIPELINE_RUN]
    assert len(run_spans) >= 2

    non_streaming_span = next(span for span in run_spans if span.tags.get("hayhooks.pipeline.name") == non_streaming_pipeline)
    streaming_span = next(span for span in run_spans if span.tags.get("hayhooks.pipeline.name") == streaming_pipeline)

    assert "hayhooks.response.streaming" not in non_streaming_span.tags
    assert "hayhooks.response.stream_type" not in non_streaming_span.tags
    assert streaming_span.tags["hayhooks.response.streaming"] is True
    assert streaming_span.tags["hayhooks.response.stream_type"] == "plain"


def test_openai_streaming_span_tags_reflect_actual_streaming(client, deploy_files, recording_tracer):
    registry.clear()

    non_streaming_pipeline = "trace_openai_non_streaming"
    streaming_pipeline = "trace_openai_streaming"

    assert deploy_files(client, non_streaming_pipeline, CHAT_PIPELINE_FILES).status_code == 200
    assert deploy_files(client, streaming_pipeline, CHAT_STREAMING_PIPELINE_FILES).status_code == 200

    non_streaming_response = client.post(
        "/chat/completions",
        json={
            "model": non_streaming_pipeline,
            "messages": [{"role": "user", "content": "What is Redis?"}],
            "stream": False,
        },
    )
    assert non_streaming_response.status_code == 200

    with client.stream(
        "POST",
        "/chat/completions",
        json={
            "model": streaming_pipeline,
            "messages": [{"role": "user", "content": "What is Redis?"}],
            "stream": True,
        },
    ) as stream_response:
        assert stream_response.status_code == 200
        list(stream_response.iter_lines())

    openai_spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_OPENAI_RUN]
    assert len(openai_spans) >= 2

    non_streaming_span = next(
        span for span in openai_spans if span.tags.get("hayhooks.pipeline.name") == non_streaming_pipeline
    )
    streaming_span = next(span for span in openai_spans if span.tags.get("hayhooks.pipeline.name") == streaming_pipeline)

    assert "hayhooks.response.streaming" not in non_streaming_span.tags
    assert "hayhooks.response.stream_type" not in non_streaming_span.tags
    assert streaming_span.tags["hayhooks.response.streaming"] is True
    assert streaming_span.tags["hayhooks.response.stream_type"] == "sse"


def test_run_endpoint_streaming_error_marks_span_failure(client, deploy_files, recording_tracer):
    registry.clear()
    pipeline_name = "trace_run_streaming_failure"

    deploy_response = deploy_files(client, pipeline_name, BROKEN_STREAMING_PIPELINE_FILES)
    assert deploy_response.status_code == 200

    with pytest.raises(RuntimeError, match="run_api stream exploded"):
        with client.stream("POST", f"/{pipeline_name}/run", json={"query": "stream"}) as stream_response:
            assert stream_response.status_code == 200
            list(stream_response.iter_text())

    run_spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_PIPELINE_RUN]
    assert run_spans
    assert run_spans[-1].tags["hayhooks.success"] is False
    assert run_spans[-1].tags["hayhooks.error.type"] == "RuntimeError"


def test_openai_streaming_error_marks_span_failure(client, deploy_files, recording_tracer):
    registry.clear()
    pipeline_name = "trace_openai_streaming_failure"

    deploy_response = deploy_files(client, pipeline_name, BROKEN_STREAMING_PIPELINE_FILES)
    assert deploy_response.status_code == 200

    with pytest.raises(RuntimeError, match="chat stream exploded"):
        with client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": pipeline_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        ) as stream_response:
            assert stream_response.status_code == 200
            list(stream_response.iter_lines())

    openai_spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_OPENAI_RUN]
    assert openai_spans
    assert openai_spans[-1].tags["hayhooks.success"] is False
    assert openai_spans[-1].tags["hayhooks.error.type"] == "RuntimeError"


def test_openai_http_4xx_marks_span_failure(client, recording_tracer):
    response = client.post(
        "/chat/completions",
        json={
            "model": "non_existent_pipeline",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 404

    openai_spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_OPENAI_RUN]
    assert openai_spans
    assert openai_spans[-1].tags["hayhooks.success"] is False
    assert openai_spans[-1].tags["hayhooks.http.status_code"] == 404


def test_parallel_startup_prepare_spans_keep_startup_parent(monkeypatch, recording_tracer, tmp_path):
    registry.clear()

    (tmp_path / "startup_one.yml").write_text(SAMPLE_YAML)
    (tmp_path / "startup_two.yml").write_text(SAMPLE_YAML)

    monkeypatch.setattr(settings, "pipelines_dir", str(tmp_path))
    monkeypatch.setattr(settings, "startup_deploy_strategy", StartupDeployStrategy.PARALLEL)
    monkeypatch.setattr(settings, "startup_deploy_workers", 2)

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/status")
        assert response.status_code == 200

    startup_spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_PIPELINE_STARTUP_DEPLOY]
    prepare_spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_PIPELINE_DEPLOY_PREPARE]

    assert len(startup_spans) == 1
    assert len(prepare_spans) == 2
    assert all(span.parent_span_id == startup_spans[0].span_id for span in prepare_spans)
