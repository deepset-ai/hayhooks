"""Hayhooks integration for the portable durable runtime."""

from __future__ import annotations

import shutil
from importlib.metadata import version
from pathlib import Path
from time import sleep

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from haystack import Pipeline
from pydantic import BaseModel

from hayhooks.durable.context import DurableContext
from hayhooks.server.app import create_app
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils import deploy_utils
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.deploy_utils import commit_prepared_pipeline, deploy_pipeline_files
from hayhooks.server.utils.models import PreparedPipeline
from hayhooks.server.utils.module_loader import create_pipeline_wrapper_instance
from hayhooks.settings import settings

_HAYSTACK_V3 = tuple(map(int, version("haystack-ai").split(".")[:2])) >= (3, 1)


class DurableRequest(BaseModel):
    value: int


class DurableWrapper(BasePipelineWrapper):
    durable_revision = "test-v1"

    def setup(self) -> None:
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: DurableRequest) -> dict[str, int]:
        await context.report_progress("started")
        return {"value": request.value}


class DurableWrapperV2(DurableWrapper):
    durable_revision = "test-v2"

    async def run_durable_async(self, context: DurableContext, request: DurableRequest) -> dict[str, str | int]:
        await context.report_progress("started")
        return {"revision": self.durable_revision, "value": request.value}


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def durable_client(monkeypatch):
    monkeypatch.setattr(settings, "durable_store", "memory")
    pipeline_dir = Path(settings.pipelines_dir) / "durable-job"
    shutil.rmtree(pipeline_dir, ignore_errors=True)
    with TestClient(create_app()) as client:
        yield client
    shutil.rmtree(pipeline_dir, ignore_errors=True)


def wrapper_for(wrapper_type: type[BasePipelineWrapper]) -> BasePipelineWrapper:
    module = type("Module", (), {"PipelineWrapper": wrapper_type})
    return create_pipeline_wrapper_instance(module)


def wait_for_status(client: TestClient, execution_id: str, status: str) -> dict[str, object]:
    for _ in range(100):
        result = client.get(f"/durable-job/executions/{execution_id}")
        if result.json()["status"] == status:
            return result.json()
        sleep(0.005)
    pytest.fail(f"execution {execution_id} did not reach {status}")


def durable_source(revision: str, *, waiting: bool = False) -> str:
    runner = (
        'del request\n        await context.suspend({"reason": "test"})\n        return {}'
        if waiting
        else 'return {"value": request.value}'
    )
    return f"""\
from haystack import Pipeline
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext


class Request(BaseModel):
    value: int


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "{revision}"

    def setup(self) -> None:
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: Request) -> dict:
        {runner}
"""


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_hayhooks_mounts_and_runs_a_durable_wrapper(durable_client):
    commit_prepared_pipeline(PreparedPipeline("durable-job", wrapper_for(DurableWrapper)), app=durable_client.app)
    submitted = durable_client.post("/durable-job/run-durable", json={"value": 7})
    assert submitted.status_code == 202
    execution_id = submitted.json()["execution_id"]
    assert wait_for_status(durable_client, execution_id, "completed")["result"] == {"value": 7}
    assert durable_client.get("/status").json()["durable"]["deployments"]["durable-job"]["healthy"]
    assert durable_client.post("/undeploy/durable-job").status_code == 200
    assert not [
        route
        for route in durable_client.app.routes
        if getattr(getattr(route, "include_context", None), "prefix", None) == "/durable-job"
        or getattr(route, "path", "").startswith("/durable-job/")
    ]
    assert durable_client.get(f"/durable-job/executions/{execution_id}").status_code == 404


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_durable_overwrite_replaces_an_idle_deployment(durable_client):
    commit_prepared_pipeline(PreparedPipeline("durable-job", wrapper_for(DurableWrapper)), app=durable_client.app)
    execution_id = durable_client.post("/durable-job/run-durable", json={"value": 7}).json()["execution_id"]
    wait_for_status(durable_client, execution_id, "completed")

    commit_prepared_pipeline(
        PreparedPipeline("durable-job", wrapper_for(DurableWrapperV2)), app=durable_client.app, overwrite=True
    )

    execution_id = durable_client.post("/durable-job/run-durable", json={"value": 9}).json()["execution_id"]
    assert wait_for_status(durable_client, execution_id, "completed")["result"] == {"revision": "test-v2", "value": 9}
    assert durable_client.app.state.durable_runtime._deployments["durable-job"].revision == "test-v2"


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_durable_overwrite_rejects_live_work_and_preserves_the_old_deployment(durable_client):
    old_source = durable_source("test-v1", waiting=True)
    candidate_source = durable_source("test-v2", waiting=True)
    deploy_pipeline_files("durable-job", {"pipeline_wrapper.py": old_source}, app=durable_client.app, save_files=True)
    old_wrapper = registry.get("durable-job")
    execution_id = durable_client.post("/durable-job/run-durable", json={"value": 7}).json()["execution_id"]
    wait_for_status(durable_client, execution_id, "waiting")

    with pytest.raises(HTTPException, match="durable executions are still active") as error:
        deploy_pipeline_files(
            "durable-job",
            {"pipeline_wrapper.py": candidate_source},
            app=durable_client.app,
            save_files=True,
            overwrite=True,
        )

    assert error.value.status_code == 409
    assert registry.get("durable-job") is old_wrapper
    assert durable_client.app.state.durable_runtime._deployments["durable-job"].revision == "test-v1"
    assert wait_for_status(durable_client, execution_id, "waiting")["status"] == "waiting"
    assert (Path(settings.pipelines_dir) / "durable-job" / "pipeline_wrapper.py").read_text() == old_source


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_durable_publication_failure_restores_the_old_deployment(durable_client, monkeypatch):
    old_source = durable_source("test-v1")
    candidate_source = durable_source("test-v2")
    deploy_pipeline_files("durable-job", {"pipeline_wrapper.py": old_source}, app=durable_client.app, save_files=True)
    old_wrapper = registry.get("durable-job")
    original = deploy_utils.add_pipeline_api_route

    def fail_candidate(app, pipeline_name, pipeline_wrapper, **kwargs):
        if pipeline_wrapper.durable_revision == "test-v2":
            message = "route publication failed"
            raise RuntimeError(message)
        return original(app, pipeline_name, pipeline_wrapper, **kwargs)

    monkeypatch.setattr(deploy_utils, "add_pipeline_api_route", fail_candidate)

    with pytest.raises(RuntimeError, match="route publication failed"):
        deploy_pipeline_files(
            "durable-job",
            {"pipeline_wrapper.py": candidate_source},
            app=durable_client.app,
            save_files=True,
            overwrite=True,
        )

    assert registry.get("durable-job") is old_wrapper
    assert durable_client.app.state.durable_runtime._deployments["durable-job"].revision == "test-v1"
    assert (Path(settings.pipelines_dir) / "durable-job" / "pipeline_wrapper.py").read_text() == old_source
    execution_id = durable_client.post("/durable-job/run-durable", json={"value": 7}).json()["execution_id"]
    assert wait_for_status(durable_client, execution_id, "completed")["result"] == {"value": 7}
