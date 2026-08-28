"""Hayhooks integration for the portable durable runtime."""

from __future__ import annotations

import asyncio
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from haystack import Pipeline
from pydantic import BaseModel

from hayhooks.durable.context import DurableContext
from hayhooks.durable.engine import Claim, Suspend, initial_control
from hayhooks.durable.models import encode_json
from hayhooks.durable.runtime import DurableDeployment
from hayhooks.durable.store import ExecutionStoreError, MemoryExecutionStore
from hayhooks.server.app import create_app
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils import deploy_utils
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.deploy_utils import commit_prepared_pipeline, deploy_pipeline_files, undeploy_pipeline
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

    async def run_durable_async(self, context: DurableContext, payload: DurableRequest) -> dict[str, int]:
        await context.report_progress("started")
        return {"value": payload.value}


class DurableWrapperV2(DurableWrapper):
    durable_revision = "test-v2"

    async def run_durable_async(self, context: DurableContext, request: DurableRequest) -> dict[str, str | int]:
        await context.report_progress("started")
        return {"revision": self.durable_revision, "value": request.value}


@pytest.fixture
def durable_client(monkeypatch):
    monkeypatch.setattr(settings, "durable_store", "memory")
    pipeline_dir = Path(settings.pipelines_dir) / "durable-job"
    registry.clear()
    shutil.rmtree(pipeline_dir, ignore_errors=True)
    try:
        with TestClient(create_app()) as client:
            yield client
    finally:
        registry.clear()
        shutil.rmtree(pipeline_dir, ignore_errors=True)


@pytest.fixture
def reject_candidate_routes(monkeypatch):
    original = deploy_utils.add_pipeline_api_route

    def fail_candidate(app, pipeline_name, pipeline_wrapper, **kwargs):
        if pipeline_wrapper.durable_revision == "test-v2":
            message = "route publication failed"
            raise RuntimeError(message)
        return original(app, pipeline_name, pipeline_wrapper, **kwargs)

    monkeypatch.setattr(deploy_utils, "add_pipeline_api_route", fail_candidate)


def wrapper_for(wrapper_type: type[BasePipelineWrapper]) -> BasePipelineWrapper:
    module = type("Module", (), {"PipelineWrapper": wrapper_type})
    return create_pipeline_wrapper_instance(module)


def durable_source(revision: str) -> str:
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
        return {{"value": request.value}}
"""


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_hayhooks_mounts_and_runs_a_durable_wrapper(durable_client, wait_for_execution):
    commit_prepared_pipeline(PreparedPipeline("durable-job", wrapper_for(DurableWrapper)), app=durable_client.app)
    headers = {"Idempotency-Key": "retry-after-response-loss"}
    submitted = durable_client.post("/durable-job/run-durable", json={"value": 7}, headers=headers)
    assert submitted.status_code == 202
    execution_id = submitted.json()["execution_id"]
    path = f"/durable-job/executions/{execution_id}"
    assert wait_for_execution(durable_client, path, "completed")["result"] == {"value": 7}
    replay = durable_client.post("/durable-job/run-durable", json={"value": 7}, headers=headers)
    assert replay.status_code == 200 and replay.headers["idempotent-replay"] == "true"
    assert replay.json()["execution_id"] == execution_id
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
def test_deploy_response_points_to_durable_endpoint(durable_client):
    response = durable_client.post(
        "/deploy_files",
        json={
            "name": "durable-job",
            "files": {"pipeline_wrapper.py": durable_source("test-v1")},
            "save_files": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["endpoint"] == "/durable-job/run-durable"
    assert durable_client.post(response.json()["endpoint"], json={"value": 7}).status_code == 202


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_durable_overwrite_replaces_an_idle_deployment(durable_client, wait_for_execution):
    commit_prepared_pipeline(PreparedPipeline("durable-job", wrapper_for(DurableWrapper)), app=durable_client.app)
    execution_id = durable_client.post("/durable-job/run-durable", json={"value": 7}).json()["execution_id"]
    wait_for_execution(durable_client, f"/durable-job/executions/{execution_id}", "completed")

    commit_prepared_pipeline(
        PreparedPipeline("durable-job", wrapper_for(DurableWrapperV2)), app=durable_client.app, overwrite=True
    )

    execution_id = durable_client.post("/durable-job/run-durable", json={"value": 9}).json()["execution_id"]
    assert wait_for_execution(durable_client, f"/durable-job/executions/{execution_id}", "completed")["result"] == {
        "revision": "test-v2",
        "value": 9,
    }
    assert durable_client.app.state.durable_runtime._deployments["durable-job"].revision == "test-v2"


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
@pytest.mark.parametrize("operation", ["overwrite", "undeploy"])
@pytest.mark.parametrize("live_status", ["queued", "running", "waiting"])
def test_dynamic_changes_reject_live_work_and_preserve_the_old_deployment(
    durable_client, operation: str, live_status: str
):
    old_source = durable_source("test-v1")
    deploy_pipeline_files("durable-job", {"pipeline_wrapper.py": old_source}, app=durable_client.app, save_files=True)
    old_wrapper = registry.get("durable-job")
    deployment = durable_client.app.state.durable_runtime._deployments["durable-job"]
    loop = durable_client.app.state.durable_loop
    asyncio.run_coroutine_threadsafe(deployment.quiesce(), loop).result()
    run_id = f"{operation}-{live_status}"
    control = initial_control(
        run_id=run_id,
        idempotency_digest=run_id,
        idempotency_binding_digest="binding",
        deployment="durable-job",
        definition_revision="test-v1",
        owner_id=None,
        kind=deployment.kind.value,
        now_ms=0,
    )
    asyncio.run_coroutine_threadsafe(
        deployment.store.submit(control, encode_json({"value": 7}, max_bytes=1_024)), loop
    ).result()
    if live_status != "queued":
        claimed = asyncio.run_coroutine_threadsafe(
            deployment.store.transition(run_id, Claim("matrix", 0, 30_000, 3, "test-v1", b"{}")),
            loop,
        ).result()
        if live_status == "waiting":
            asyncio.run_coroutine_threadsafe(
                deployment.store.transition(run_id, Suspend(claimed.next_control.fence, "matrix", 0, b"{}", b"{}")),
                loop,
            ).result()

    with pytest.raises(HTTPException, match="durable executions are still active") as error:
        if operation == "overwrite":
            deploy_pipeline_files(
                "durable-job",
                {"pipeline_wrapper.py": durable_source("test-v2")},
                app=durable_client.app,
                save_files=True,
                overwrite=True,
            )
        else:
            undeploy_pipeline("durable-job", app=durable_client.app)

    assert error.value.status_code == 409
    assert registry.get("durable-job") is old_wrapper
    assert durable_client.app.state.durable_runtime._deployments["durable-job"].revision == "test-v1"
    assert (Path(settings.pipelines_dir) / "durable-job" / "pipeline_wrapper.py").read_text() == old_source


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
@pytest.mark.parametrize("operation", ["overwrite", "undeploy"])
def test_dynamic_change_store_failure_restarts_the_old_deployment(durable_client, monkeypatch, operation: str):
    old_wrapper = wrapper_for(DurableWrapper)
    commit_prepared_pipeline(PreparedPipeline("durable-job", old_wrapper), app=durable_client.app)
    deployment = durable_client.app.state.durable_runtime._deployments["durable-job"]
    monkeypatch.setattr(
        deployment.store,
        "operational_counts",
        AsyncMock(side_effect=ExecutionStoreError("store unavailable")),
    )

    with pytest.raises(ExecutionStoreError, match="store unavailable"):
        if operation == "overwrite":
            commit_prepared_pipeline(
                PreparedPipeline("durable-job", wrapper_for(DurableWrapperV2)),
                app=durable_client.app,
                overwrite=True,
            )
        else:
            undeploy_pipeline("durable-job", app=durable_client.app)

    assert deployment.accepting
    assert registry.get("durable-job") is old_wrapper


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_startup_deploys_durable_work_and_app_runtimes_are_isolated(tmp_path, monkeypatch, wait_for_execution):
    monkeypatch.setattr(settings, "durable_store", "memory")
    monkeypatch.setattr(settings, "pipelines_dir", str(tmp_path))
    pipeline_dir = tmp_path / "durable-job"
    pipeline_dir.mkdir()
    (pipeline_dir / "pipeline_wrapper.py").write_text(durable_source("test-v1"))
    registry.clear()
    app = create_app()
    other_app = create_app()
    try:
        with TestClient(app) as client, TestClient(other_app):
            assert app.state.durable_runtime is not other_app.state.durable_runtime
            assert set(app.state.durable_runtime._deployments) == {"durable-job"}
            assert not other_app.state.durable_runtime._deployments
            submitted = client.post("/durable-job/run-durable", json={"value": 7}).json()
            assert wait_for_execution(client, submitted["links"]["self"], "completed")["result"] == {"value": 7}
    finally:
        registry.clear()


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_durable_to_nondurable_overwrite_removes_durable_routes(durable_client):
    class OrdinaryWrapper(BasePipelineWrapper):
        def setup(self) -> None:
            self.pipeline = Pipeline()

        def run_api(self, value: int) -> dict[str, int]:
            return {"value": value}

    commit_prepared_pipeline(PreparedPipeline("durable-job", wrapper_for(DurableWrapper)), app=durable_client.app)
    commit_prepared_pipeline(
        PreparedPipeline("durable-job", wrapper_for(OrdinaryWrapper)), app=durable_client.app, overwrite=True
    )

    assert "durable-job" not in durable_client.app.state.durable_runtime._deployments
    assert "durable_deployment" not in registry.get_metadata("durable-job")
    assert durable_client.post("/durable-job/run-durable", json={"value": 1}).status_code == 404
    assert durable_client.post("/durable-job/run", json={"value": 1}).json() == {"result": {"value": 1}}


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_store_initialization_failure_publishes_nothing(durable_client, monkeypatch):
    monkeypatch.setattr(
        MemoryExecutionStore,
        "initialize",
        AsyncMock(side_effect=ExecutionStoreError("store unavailable")),
    )

    with pytest.raises(ExecutionStoreError, match="store unavailable"):
        commit_prepared_pipeline(PreparedPipeline("durable-job", wrapper_for(DurableWrapper)), app=durable_client.app)

    assert registry.get("durable-job") is None
    assert "durable-job" not in durable_client.app.state.durable_runtime._deployments
    assert not [route for route in durable_client.app.routes if getattr(route, "path", "").startswith("/durable-job/")]


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
@pytest.mark.parametrize("save_files", [True, False])
def test_durable_publication_failure_restores_the_old_deployment(
    durable_client, reject_candidate_routes, wait_for_execution, save_files: bool
):
    old_source = durable_source("test-v1")
    candidate_source = durable_source("test-v2")
    deploy_pipeline_files("durable-job", {"pipeline_wrapper.py": old_source}, app=durable_client.app, save_files=True)
    old_wrapper = registry.get("durable-job")
    module_names = ("durable-job", "durable-job.pipeline_wrapper")
    old_modules = {name: sys.modules[name] for name in module_names}
    with pytest.raises(RuntimeError, match="route publication failed"):
        deploy_pipeline_files(
            "durable-job",
            {"pipeline_wrapper.py": candidate_source},
            app=durable_client.app,
            save_files=save_files,
            overwrite=True,
        )

    assert registry.get("durable-job") is old_wrapper
    assert {name: sys.modules[name] for name in module_names} == old_modules
    assert durable_client.app.state.durable_runtime._deployments["durable-job"].revision == "test-v1"
    assert (Path(settings.pipelines_dir) / "durable-job" / "pipeline_wrapper.py").read_text() == old_source
    execution_id = durable_client.post("/durable-job/run-durable", json={"value": 7}).json()["execution_id"]
    assert wait_for_execution(durable_client, f"/durable-job/executions/{execution_id}", "completed")["result"] == {
        "value": 7
    }


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_candidate_close_failure_does_not_block_publication_rollback(
    durable_client, monkeypatch, reject_candidate_routes, wait_for_execution
):
    old_wrapper = wrapper_for(DurableWrapper)
    commit_prepared_pipeline(PreparedPipeline("durable-job", old_wrapper), app=durable_client.app)
    original_close = DurableDeployment.close

    async def close_then_fail(deployment):
        await original_close(deployment)
        if deployment.revision == "test-v2":
            message = "candidate close failed"
            raise RuntimeError(message)

    monkeypatch.setattr(DurableDeployment, "close", close_then_fail)

    with pytest.raises(RuntimeError, match="route publication failed"):
        commit_prepared_pipeline(
            PreparedPipeline("durable-job", wrapper_for(DurableWrapperV2)),
            app=durable_client.app,
            overwrite=True,
        )

    assert registry.get("durable-job") is old_wrapper
    execution_id = durable_client.post("/durable-job/run-durable", json={"value": 7}).json()["execution_id"]
    assert wait_for_execution(durable_client, f"/durable-job/executions/{execution_id}", "completed")["result"] == {
        "value": 7
    }


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_file_cleanup_failure_does_not_block_publication_rollback(
    durable_client, monkeypatch, reject_candidate_routes, wait_for_execution
):
    old_source = durable_source("test-v1")
    deploy_pipeline_files("durable-job", {"pipeline_wrapper.py": old_source}, app=durable_client.app, save_files=True)
    old_wrapper = registry.get("durable-job")
    original_remove = deploy_utils.remove_pipeline_files

    def remove_then_fail(pipeline_name, pipelines_dir):
        original_remove(pipeline_name, pipelines_dir)
        message = "file cleanup failed"
        raise OSError(message)

    monkeypatch.setattr(deploy_utils, "remove_pipeline_files", remove_then_fail)

    with pytest.raises(RuntimeError, match="route publication failed"):
        commit_prepared_pipeline(
            PreparedPipeline("durable-job", wrapper_for(DurableWrapperV2)),
            app=durable_client.app,
            overwrite=True,
            source_files={"pipeline_wrapper.py": durable_source("test-v2")},
        )

    assert registry.get("durable-job") is old_wrapper
    assert (Path(settings.pipelines_dir) / "durable-job" / "pipeline_wrapper.py").read_text() == old_source
    execution_id = durable_client.post("/durable-job/run-durable", json={"value": 7}).json()["execution_id"]
    assert wait_for_execution(durable_client, f"/durable-job/executions/{execution_id}", "completed")["result"] == {
        "value": 7
    }


@pytest.mark.skipif(not _HAYSTACK_V3, reason="Hayhooks durable wrappers require Haystack 3.1+")
def test_rollback_republishes_before_reactivating_the_old_deployment(
    durable_client, monkeypatch, reject_candidate_routes
):
    old_wrapper = wrapper_for(DurableWrapper)
    commit_prepared_pipeline(PreparedPipeline("durable-job", old_wrapper), app=durable_client.app)
    old_deployment = durable_client.app.state.durable_runtime._deployments["durable-job"]
    original_start = DurableDeployment.start
    reactivated = False

    async def assert_publication_then_start(deployment):
        nonlocal reactivated
        if deployment is old_deployment:
            assert registry.get("durable-job") is old_wrapper
            assert any(
                getattr(getattr(route, "include_context", None), "prefix", None) == "/durable-job"
                for route in durable_client.app.routes
            )
            reactivated = True
        await original_start(deployment)

    monkeypatch.setattr(DurableDeployment, "start", assert_publication_then_start)

    with pytest.raises(RuntimeError, match="route publication failed"):
        commit_prepared_pipeline(
            PreparedPipeline("durable-job", wrapper_for(DurableWrapperV2)),
            app=durable_client.app,
            overwrite=True,
        )

    assert reactivated
