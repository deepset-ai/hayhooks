import importlib.metadata
import time

import pytest
from fastapi.testclient import TestClient

from hayhooks.durable_runtime import durable_runtime
from hayhooks.server.app import create_app
from hayhooks.server.pipelines.registry import registry
from hayhooks.settings import settings

pytestmark = pytest.mark.skipif(
    not importlib.metadata.version("haystack-ai").startswith("3."), reason="durable execution requires Haystack 3"
)


def _durable_source(*, field: str, increment: int, revision: str) -> str:
    return f"""
from haystack import Pipeline
from pydantic import BaseModel
from hayhooks import BasePipelineWrapper, DurableContext, DurableOptions

class Request(BaseModel):
    {field}: int

class PipelineWrapper(BasePipelineWrapper):
    durable_options = DurableOptions(revision="{revision}")

    def setup(self):
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: Request) -> dict:
        return {{"value": request.{field} + {increment}}}
"""


def _api_source(*, increment: int = 1, fail_start: bool = False) -> str:
    start = 'raise RuntimeError("startup failed")' if fail_start else "self.started = True"
    return f"""
from haystack import Pipeline
from hayhooks import BasePipelineWrapper

class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = Pipeline()
        self.started = False
        self.closed = False

    async def start(self):
        {start}

    async def close(self):
        self.closed = True

    def run_api(self, value: int) -> int:
        return value + {increment}
"""


def _waiting_source(*, revision: str) -> str:
    return f"""
from haystack import Pipeline
from pydantic import BaseModel
from hayhooks import BasePipelineWrapper, DurableContext, DurableOptions

class Request(BaseModel):
    value: int

class PipelineWrapper(BasePipelineWrapper):
    durable_options = DurableOptions(revision="{revision}")

    def setup(self):
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: Request) -> dict:
        if context.resume_input is None:
            await context.suspend({{"kind": "input", "message": "waiting"}})
        return {{"value": request.value}}
"""


def _blocking_source() -> str:
    return """
import threading
from haystack import Pipeline
from pydantic import BaseModel
from hayhooks import BasePipelineWrapper, DurableContext, DurableOptions

class Request(BaseModel):
    value: int

class PipelineWrapper(BasePipelineWrapper):
    durable_options = DurableOptions(revision="blocking-v1")

    def setup(self):
        self.pipeline = Pipeline()
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False

    async def close(self):
        self.closed = True

    def run_durable(self, context: DurableContext, request: Request) -> dict:
        self.started.set()
        assert self.release.wait(timeout=5)
        return {"value": request.value}
"""


def _deploy(client: TestClient, source: str, *, overwrite: bool = False):
    return client.post(
        "/deploy_files",
        json={
            "name": "job",
            "files": {"pipeline_wrapper.py": source},
            "save_files": False,
            "overwrite": overwrite,
        },
    )


def _wait_for_completion(client: TestClient, response) -> dict:
    body = response.json()
    for _ in range(200):
        result = client.get(body["links"]["self"])
        if result.json()["status"] == "completed":
            return result.json()
        time.sleep(0.01)
    pytest.fail("durable execution did not complete")


@pytest.fixture(autouse=True)
def _isolated_runtime(monkeypatch, tmp_path):
    registry.clear()
    monkeypatch.setattr(settings, "pipelines_dir", str(tmp_path))
    monkeypatch.setattr(settings, "durable_store", "memory")
    yield
    registry.clear()


def test_undeploy_removes_entire_durable_route_family() -> None:
    app = create_app()
    with TestClient(app) as client:
        assert _deploy(client, _durable_source(field="value", increment=1, revision="v1")).status_code == 200
        submitted = client.post("/job/run-durable", json={"value": 1})
        execution_id = submitted.json()["execution_id"]
        deployment = durable_runtime.current_deployment("job")
        assert deployment is not None

        assert client.post("/undeploy/job").status_code == 200
        assert client.post("/job/run-durable", json={"value": 2}).status_code == 404
        assert client.get(f"/job/executions/{execution_id}").status_code == 404
        assert client.post(f"/job/executions/{execution_id}/cancel").status_code == 404
        assert client.post(f"/job/executions/{execution_id}/resume").status_code == 404
        persisted = deployment.store._records[execution_id]
        assert persisted.terminal


def test_undeploy_terminalizes_waiting_execution() -> None:
    app = create_app()
    with TestClient(app) as client:
        assert _deploy(client, _waiting_source(revision="v1")).status_code == 200
        submitted = client.post("/job/run-durable", json={"value": 1})
        execution_id = submitted.json()["execution_id"]
        deployment = durable_runtime.current_deployment("job")
        assert deployment is not None
        for _ in range(100):
            if client.get(submitted.json()["links"]["self"]).json()["status"] == "waiting":
                break
            time.sleep(0.01)
        else:
            pytest.fail("execution did not enter waiting before undeploy")

        assert client.post("/undeploy/job").status_code == 200
        persisted = deployment.store._records[execution_id]

    assert persisted.status.value == "failed"
    assert persisted.error is not None
    assert persisted.error.code == "definition_revision_conflict"


def test_durable_overwrite_routes_bind_new_model_runner_and_revision() -> None:
    app = create_app()
    with TestClient(app) as client:
        assert _deploy(client, _durable_source(field="old_value", increment=1, revision="v1")).status_code == 200
        first = client.post("/job/run-durable", json={"old_value": 2})
        assert _wait_for_completion(client, first)["result"] == {"value": 3}

        assert (
            _deploy(
                client,
                _durable_source(field="new_value", increment=20, revision="v2"),
                overwrite=True,
            ).status_code
            == 200
        )
        assert client.post("/job/run-durable", json={"old_value": 2}).status_code == 422
        second = client.post("/job/run-durable", json={"new_value": 2})
        assert second.status_code == 202
        assert _wait_for_completion(client, second)["result"] == {"value": 22}


def test_overwrite_terminalizes_waiting_execution_from_old_revision() -> None:
    app = create_app()
    with TestClient(app) as client:
        assert _deploy(client, _waiting_source(revision="v1")).status_code == 200
        submitted = client.post("/job/run-durable", json={"value": 2})
        url = submitted.json()["links"]["self"]
        for _ in range(100):
            waiting = client.get(url)
            if waiting.json()["status"] == "waiting":
                break
            time.sleep(0.01)
        else:
            pytest.fail("old-revision execution did not enter waiting")

        assert (
            _deploy(
                client,
                _durable_source(field="value", increment=20, revision="v2"),
                overwrite=True,
            ).status_code
            == 200
        )
        retired = client.get(url)

    assert retired.status_code == 200
    assert retired.json()["status"] == "failed"
    assert retired.json()["error"]["code"] == "definition_revision_conflict"


def test_overwrite_defers_old_wrapper_close_until_thread_backed_work_drains(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_shutdown_grace_period", 0.001)
    app = create_app()
    with TestClient(app) as client:
        assert _deploy(client, _blocking_source()).status_code == 200
        old_wrapper = registry.get("job")
        assert old_wrapper is not None
        submitted = client.post("/job/run-durable", json={"value": 2})
        assert old_wrapper.started.wait(timeout=1)

        assert (
            _deploy(
                client,
                _durable_source(field="value", increment=20, revision="v2"),
                overwrite=True,
            ).status_code
            == 200
        )
        assert not old_wrapper.closed
        old_wrapper.release.set()
        for _ in range(100):
            if old_wrapper.closed:
                break
            time.sleep(0.01)
        else:
            pytest.fail("replaced wrapper did not close after detached work drained")

        completed = client.get(submitted.json()["links"]["self"])
        assert completed.json()["status"] == "completed"


def test_undeploy_defers_old_wrapper_close_until_thread_backed_work_drains(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_shutdown_grace_period", 0.001)
    app = create_app()
    with TestClient(app) as client:
        assert _deploy(client, _blocking_source()).status_code == 200
        old_wrapper = registry.get("job")
        assert old_wrapper is not None
        client.post("/job/run-durable", json={"value": 2})
        assert old_wrapper.started.wait(timeout=1)

        assert client.post("/undeploy/job").status_code == 200
        assert not old_wrapper.closed
        old_wrapper.release.set()
        for _ in range(100):
            if old_wrapper.closed:
                break
            time.sleep(0.01)
        else:
            pytest.fail("undeployed wrapper did not close after detached work drained")


def test_durable_to_non_durable_overwrite_removes_control_routes() -> None:
    app = create_app()
    with TestClient(app) as client:
        assert _deploy(client, _durable_source(field="value", increment=1, revision="v1")).status_code == 200
        execution_id = client.post("/job/run-durable", json={"value": 1}).json()["execution_id"]

        assert _deploy(client, _api_source(increment=5), overwrite=True).status_code == 200
        assert client.post("/job/run-durable", json={"value": 2}).status_code == 404
        assert client.get(f"/job/executions/{execution_id}").status_code == 404
        assert client.post("/job/run", json={"value": 2}).json() == {"result": 7}


def test_failed_overwrite_restores_old_registry_routes_and_lifecycle() -> None:
    app = create_app()
    with TestClient(app) as client:
        assert _deploy(client, _api_source(increment=1)).status_code == 200
        old_wrapper = registry.get("job")
        assert old_wrapper is not None

        failed = _deploy(client, _api_source(increment=100, fail_start=True), overwrite=True)

        assert failed.status_code == 500
        assert registry.get("job") is old_wrapper
        assert not old_wrapper.closed
        assert client.post("/job/run", json={"value": 2}).json() == {"result": 3}


class _FailingStore:
    async def initialize(self):
        msg = "redis unavailable"
        raise ConnectionError(msg)

    async def close(self):
        return None


class _FailingProvider:
    def create_execution_store(self, _deployment_name):
        return _FailingStore()

    async def close(self):
        return None


def test_store_initialization_failure_never_publishes_candidate(monkeypatch) -> None:
    monkeypatch.setattr(durable_runtime, "provider", _FailingProvider())
    app = create_app()
    with TestClient(app) as client:
        failed = _deploy(client, _durable_source(field="value", increment=1, revision="v1"))

        assert failed.status_code == 500
        assert registry.get("job") is None
        assert client.post("/job/run-durable", json={"value": 1}).status_code == 404
