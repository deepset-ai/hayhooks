import importlib.metadata

import pytest
from fastapi.testclient import TestClient

from hayhooks.durable.runtime import DurableRuntime
from hayhooks.server.app import create_app
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.deploy_utils import deploy_pipeline_files
from hayhooks.settings import settings
from tests.durable_helpers import wait_for_status

pytestmark = pytest.mark.skipif(
    not importlib.metadata.version("haystack-ai").startswith("3."), reason="durable execution requires Haystack 3"
)


def _durable_source(*, field: str, increment: int, revision: str, result_field: str = "value") -> str:
    return f"""
from haystack import Pipeline
from pydantic import BaseModel
from hayhooks import BasePipelineWrapper, DurableContext

class Request(BaseModel):
    {field}: int

class Result(BaseModel):
    {result_field}: int

class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "{revision}"

    def setup(self):
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
        return Result({result_field}=request.{field} + {increment})
"""


def _api_source(*, increment: int = 1) -> str:
    return f"""
from haystack import Pipeline
from hayhooks import BasePipelineWrapper

class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = Pipeline()

    def run_api(self, value: int) -> int:
        return value + {increment}
"""


def _waiting_source(*, revision: str) -> str:
    return f"""
from haystack import Pipeline
from pydantic import BaseModel
from hayhooks import BasePipelineWrapper, DurableContext

class Request(BaseModel):
    value: int

class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "{revision}"

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
from hayhooks import BasePipelineWrapper, DurableContext

class Request(BaseModel):
    value: int

class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "blocking"

    def setup(self):
        self.pipeline = Pipeline()
        self.started = threading.Event()
        self.release = threading.Event()

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


def _deploy_ok(client: TestClient, source: str, *, overwrite: bool = False) -> None:
    assert _deploy(client, source, overwrite=overwrite).status_code == 200


def _wait_for_completion(client: TestClient, response) -> dict:
    return wait_for_status(client, response.json()["links"]["self"], "completed")


@pytest.fixture(autouse=True)
def _isolated_runtime(monkeypatch, tmp_path):
    registry.clear()
    monkeypatch.setattr(settings, "pipelines_dir", str(tmp_path))
    monkeypatch.setattr(settings, "durable_store", "memory")
    monkeypatch.setattr(settings, "durable_poll_interval", 0.05)
    yield
    registry.clear()


def test_undeploy_removes_entire_durable_route_family() -> None:
    app = create_app()
    with TestClient(app) as client:
        _deploy_ok(client, _durable_source(field="value", increment=1, revision="first"))
        assert client.post("/undeploy/job").status_code == 200
        assert client.post("/job/run-durable", json={"value": 2}).status_code == 404
        assert client.get("/job/executions/missing").status_code == 404
        assert client.post("/job/executions/missing/cancel").status_code == 404
        assert client.post("/job/executions/missing/resume").status_code == 404
        assert app.state.durable_runtime.current_deployment("job") is None


def test_undeploy_refuses_to_strand_waiting_execution() -> None:
    app = create_app()
    with TestClient(app) as client:
        _deploy_ok(client, _waiting_source(revision="first"))
        submitted = client.post("/job/run-durable", json={"value": 1})
        wait_for_status(client, submitted.json()["links"]["self"], "waiting")

        blocked = client.post("/undeploy/job")
        assert blocked.status_code == 409
        assert "completed or canceled" in blocked.json()["detail"]
        assert client.get(submitted.json()["links"]["self"]).json()["status"] == "waiting"
        assert client.post(f"{submitted.json()['links']['self']}/cancel").status_code == 202
        assert client.post("/undeploy/job").status_code == 200
        assert client.get(submitted.json()["links"]["self"]).status_code == 404


def test_durable_overwrite_routes_bind_new_model_runner_and_revision() -> None:
    app = create_app()
    with TestClient(app) as client:
        _deploy_ok(client, _durable_source(field="old_value", increment=1, revision="first", result_field="old_result"))
        first = client.post("/job/run-durable", json={"old_value": 2})
        assert _wait_for_completion(client, first)["result"] == {"old_result": 3}

        _deploy_ok(
            client,
            _durable_source(field="new_value", increment=20, revision="second", result_field="new_result"),
            overwrite=True,
        )
        assert client.get(first.json()["links"]["self"]).json()["result"] == {"old_result": 3}
        assert client.post("/job/run-durable", json={"old_value": 2}).status_code == 422
        second = client.post("/job/run-durable", json={"new_value": 2})
        assert second.status_code == 202
        assert _wait_for_completion(client, second)["result"] == {"new_result": 22}


@pytest.mark.parametrize("operation", ["overwrite", "undeploy"])
def test_failed_durable_preflight_restarts_existing_deployment(monkeypatch, operation: str) -> None:
    app = create_app()
    with TestClient(app) as client:
        source = _durable_source(field="value", increment=1, revision="first")
        _deploy_ok(client, source)
        deployment = app.state.durable_runtime.current_deployment("job")
        assert deployment is not None

        async def fail_counts():
            msg = "redis unavailable"
            raise ConnectionError(msg)

        monkeypatch.setattr(deployment.store, "operational_counts", fail_counts)
        if operation == "overwrite":
            assert _deploy(client, source, overwrite=True).status_code == 500
        else:
            with pytest.raises(ConnectionError, match="redis unavailable"):
                client.post("/undeploy/job")
        assert deployment.manager.accepting
        assert client.post("/job/run-durable", json={"value": 1}).status_code == 202


def test_overwrite_refuses_to_prepare_while_old_execution_is_waiting(tmp_path) -> None:
    app = create_app()
    with TestClient(app) as client:
        _deploy_ok(client, _waiting_source(revision="first"))
        submitted = client.post("/job/run-durable", json={"value": 2})
        url = submitted.json()["links"]["self"]
        wait_for_status(client, url, "waiting")

        preparation_marker = tmp_path / "replacement-prepared"
        replacement_source = _durable_source(field="value", increment=20, revision="second").replace(
            "from haystack import Pipeline",
            f"from haystack import Pipeline\nfrom pathlib import Path\nPath({str(preparation_marker)!r}).touch()",
        )
        replacement = _deploy(
            client,
            replacement_source,
            overwrite=True,
        )
        assert replacement.status_code == 409
        assert "complete or cancel" in replacement.json()["detail"]
        assert not preparation_marker.exists()
        assert client.get(url).json()["status"] == "waiting"
        assert client.post(f"{url}/cancel").status_code == 202
        _deploy_ok(client, _durable_source(field="value", increment=20, revision="second"), overwrite=True)


def test_undeploy_refuses_to_strand_thread_backed_work(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_shutdown_grace_period", 0.001)
    app = create_app()
    with TestClient(app) as client:
        _deploy_ok(client, _blocking_source())
        old_wrapper = app.state.pipeline_registry.get("job")
        assert old_wrapper is not None
        submitted = client.post("/job/run-durable", json={"value": 2})
        assert old_wrapper.started.wait(timeout=1)

        assert client.post("/undeploy/job").status_code == 409
        old_wrapper.release.set()
        _wait_for_completion(client, submitted)
        assert client.post("/undeploy/job").status_code == 200


def test_fastapi_apps_own_pipeline_publication_and_allow_the_same_name() -> None:
    app_a = create_app()
    app_b = create_app()

    with TestClient(app_a) as client_a, TestClient(app_b) as client_b:
        _deploy_ok(client_a, _durable_source(field="value", increment=1, revision="app-a"))

        assert client_a.get("/status").json()["pipelines"] == ["job"]
        assert client_b.get("/status").json()["pipelines"] == []
        assert client_b.get("/status/job").status_code == 404
        assert client_b.post("/job/run-durable", json={"value": 1}).status_code == 404

        _deploy_ok(client_b, _durable_source(field="value", increment=10, revision="app-b"))
        submitted_a = client_a.post("/job/run-durable", json={"value": 1})
        submitted_b = client_b.post("/job/run-durable", json={"value": 1})
        assert _wait_for_completion(client_a, submitted_a)["result"] == {"value": 2}
        assert _wait_for_completion(client_b, submitted_b)["result"] == {"value": 11}


def test_deploy_rejects_a_runtime_owned_by_another_app() -> None:
    app = create_app()
    other_runtime = DurableRuntime(app_settings=settings)

    with pytest.raises(ValueError, match="does not belong"):
        deploy_pipeline_files(
            "job",
            {"pipeline_wrapper.py": _api_source()},
            app=app,
            save_files=False,
            durable_runtime=other_runtime,
        )

    assert app.state.pipeline_registry.get("job") is None
    assert all(getattr(route, "path", None) != "/job/run" for route in app.routes)


def test_durable_to_non_durable_overwrite_removes_control_routes() -> None:
    app = create_app()
    with TestClient(app) as client:
        _deploy_ok(client, _durable_source(field="value", increment=1, revision="first"))
        submitted = client.post("/job/run-durable", json={"value": 1})
        execution_id = submitted.json()["execution_id"]
        _wait_for_completion(client, submitted)

        _deploy_ok(client, _api_source(increment=5), overwrite=True)
        assert client.post("/job/run-durable", json={"value": 2}).status_code == 404
        assert client.get(f"/job/executions/{execution_id}").status_code == 404
        assert client.post("/job/run", json={"value": 2}).json() == {"result": 7}


def test_failed_commit_does_not_irreversibly_retire_old_durable_work(monkeypatch) -> None:
    app = create_app()
    with TestClient(app) as client:
        source = _durable_source(field="value", increment=1, revision="first")
        _deploy_ok(client, source)
        submitted = client.post("/job/run-durable", json={"value": 1})
        url = submitted.json()["links"]["self"]
        _wait_for_completion(client, submitted)

        def fail_commit(*_args, **_kwargs):
            msg = "commit fault"
            raise RuntimeError(msg)

        monkeypatch.setattr("hayhooks.server.utils.deploy_utils.commit_prepared_pipeline", fail_commit)
        failed = _deploy(client, source, overwrite=True)

        assert failed.status_code == 500
        restored = client.get(url)
        assert restored.status_code == 200
        assert restored.json()["status"] == "completed"


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
    app = create_app(durable_runtime=DurableRuntime(_FailingProvider(), app_settings=settings))
    with TestClient(app) as client:
        failed = _deploy(client, _durable_source(field="value", increment=1, revision="first"))

        assert failed.status_code == 500
        assert registry.get("job") is None
        assert client.post("/job/run-durable", json={"value": 1}).status_code == 404
