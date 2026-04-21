import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture
from fastapi import FastAPI
from fastapi.testclient import TestClient
from haystack.tracing import Span, Tracer, disable_tracing, enable_tracing

from hayhooks.server.app import create_app
from hayhooks.server.logger import log
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.mcp_utils import create_mcp_server, create_starlette_app
from hayhooks.settings import settings


class _RecordedSpan(Span):
    def __init__(
        self, operation_name: str, tags: dict[str, Any], trace_id: int, span_id: int, parent_span_id: int | None
    ):
        self.operation_name = operation_name
        self.tags = tags
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "span_id": self.span_id}


class _RecordingTracer(Tracer):
    def __init__(self) -> None:
        self.spans: list[_RecordedSpan] = []
        self._active_spans: ContextVar[tuple[_RecordedSpan, ...]] = ContextVar("_recording_tracer_active_spans", default=())
        self._next_id = 1

    @contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: Span | None = None,
    ) -> Iterator[Span]:
        active_spans = self._active_spans.get()
        parent = parent_span if isinstance(parent_span, _RecordedSpan) else (active_spans[-1] if active_spans else None)
        trace_id = parent.trace_id if isinstance(parent, _RecordedSpan) else self._allocate_id()
        span_id = self._allocate_id()
        span = _RecordedSpan(
            operation_name=operation_name,
            tags=dict(tags or {}),
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent.span_id if isinstance(parent, _RecordedSpan) else None,
        )
        self.spans.append(span)
        token = self._active_spans.set((*active_spans, span))
        try:
            yield span
        finally:
            self._active_spans.reset(token)

    def current_span(self) -> Span | None:
        active_spans = self._active_spans.get()
        if active_spans:
            return active_spans[-1]
        return None

    def _allocate_id(self) -> int:
        current = self._next_id
        self._next_id += 1
        return current


@pytest.fixture
def recording_tracer() -> Iterator[_RecordingTracer]:
    tracer = _RecordingTracer()
    enable_tracing(tracer)
    try:
        yield tracer
    finally:
        disable_tracing()


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """
    Override caplog fixture to work with loguru.

    See: https://loguru.readthedocs.io/en/stable/resources/migration.html
    """
    handler_id = log.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    log.remove(handler_id)


def pytest_configure(config):
    config.addinivalue_line("markers", "mcp: mark tests that require the MCP package")


@pytest.fixture(scope="session", autouse=True)
def test_settings():
    settings.pipelines_dir = str(Path(__file__).parent / "pipelines")
    return settings


@pytest.fixture(scope="session", autouse=True)
def test_app():
    return create_app()


@pytest.fixture
def test_mcp_server():
    return create_starlette_app(create_mcp_server(), debug=True, json_response=True)


@pytest.fixture
def test_mcp_client(test_mcp_server):
    return TestClient(test_mcp_server)


@pytest.fixture
def client(test_app: FastAPI):
    return TestClient(test_app)


@pytest.fixture(scope="module", autouse=True)
def cleanup_pipelines(test_settings):
    """
    This fixture is used to cleanup the pipelines directory
    and the registry after each test module.
    """
    yield
    registry.clear()
    if Path(test_settings.pipelines_dir).exists():
        shutil.rmtree(test_settings.pipelines_dir)


@pytest.fixture
def deploy_yaml_pipeline():
    def _deploy_yaml_pipeline(client: TestClient, pipeline_name: str, pipeline_source_code: str):
        deploy_response = client.post("/deploy-yaml", json={"name": pipeline_name, "source_code": pipeline_source_code})
        return deploy_response

    return _deploy_yaml_pipeline


@pytest.fixture
def undeploy_pipeline():
    def _undeploy_pipeline(client: TestClient, pipeline_name: str):
        undeploy_response = client.post(f"/undeploy/{pipeline_name}")
        return undeploy_response

    return _undeploy_pipeline


@pytest.fixture
def draw_pipeline():
    def _draw_pipeline(client: TestClient, pipeline_name: str):
        draw_response = client.get(f"/draw/{pipeline_name}")
        return draw_response

    return _draw_pipeline


@pytest.fixture
def status_pipeline():
    def _status_pipeline(client: TestClient, pipeline_name: str):
        status_response = client.get(f"/status/{pipeline_name}")
        return status_response

    return _status_pipeline


@pytest.fixture
def chat_completion():
    def _chat_completion(client: TestClient, pipeline_name: str, messages: list):
        chat_response = client.post("/chat/completions", json={"messages": messages, "model": pipeline_name})
        return chat_response

    return _chat_completion


@pytest.fixture
def deploy_files():
    def _deploy_files(
        client: TestClient, pipeline_name: str, pipeline_files: dict, overwrite: bool = False, save_files: bool = True
    ):
        deploy_response = client.post(
            "/deploy_files",
            json={"name": pipeline_name, "files": pipeline_files, "overwrite": overwrite, "save_files": save_files},
        )
        return deploy_response

    return _deploy_files
