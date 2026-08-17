"""Standalone contract tests for the public FastAPI durable adapter."""

from __future__ import annotations

import importlib.metadata
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Annotated

import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from haystack import Pipeline
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper
from hayhooks.durable import (
    DurableContext,
    DurableRuntime,
    DurableSettings,
    InMemoryExecutionStoreProvider,
    create_durable_router,
)

pytestmark = pytest.mark.skipif(
    not importlib.metadata.version("haystack-ai").startswith("3."), reason="durable execution requires Haystack 3"
)


class JobRequest(BaseModel):
    value: int


class JobResult(BaseModel):
    value: int
    owner_id: str | None


class ResumeInput(BaseModel):
    approved: bool


class JobWrapper(BasePipelineWrapper):
    durable_revision = "portable-job-v1"
    durable_resume_model = ResumeInput

    def setup(self) -> None:
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: JobRequest) -> JobResult:
        if request.value == 0 and context.resume_input is None:
            await context.suspend({"kind": "approval", "private": "hidden"})
        approved = context.take_resume_input()
        value = request.value if approved is None or ResumeInput.model_validate(approved).approved else -1
        return JobResult(value=value + 1, owner_id=context.owner_id)


def _require_principal(request: Request) -> str:
    if request.headers.get("X-Deny"):
        raise HTTPException(status_code=403, detail="Forbidden")
    return request.headers.get("X-Owner", "")


async def _dependency_owner_id(principal: Annotated[str, Depends(_require_principal)]) -> str:
    return principal


def _app(
    owner_dependency=None,
    *,
    middleware_auth: bool = False,
) -> tuple[FastAPI, DurableRuntime]:
    durable_settings = DurableSettings(durable_store="memory", durable_poll_interval=0.05)
    runtime = DurableRuntime(InMemoryExecutionStoreProvider(durable_settings=durable_settings))
    wrapper = JobWrapper()
    wrapper.setup()
    deployment = runtime.deployment("jobs", wrapper)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            await runtime.start()
            yield
        finally:
            await runtime.close()

    app = FastAPI(lifespan=lifespan)
    if middleware_auth:

        @app.middleware("http")
        async def authenticate(request: Request, call_next):
            owner = request.headers.get("X-Owner")
            if owner is None:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            request.state.principal = SimpleNamespace(owner_id=owner)
            return await call_next(request)

    outer = APIRouter(prefix="/api")
    outer.include_router(
        create_durable_router(deployment, owner_id_dependency=owner_dependency),
        prefix="/jobs",
    )
    app.include_router(outer)
    return app, runtime


def _wait(client: TestClient, url: str, expected: str) -> dict:
    for _ in range(200):
        response = client.get(url)
        if response.json()["status"] == expected:
            return response.json()
        time.sleep(0.01)
    pytest.fail(f"execution did not become {expected}")


def test_public_router_is_typed_prefix_safe_and_supports_all_routes() -> None:
    app, _ = _app(owner_dependency=None)
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 0})
        assert submitted.status_code == 202
        assert submitted.headers["Location"].startswith("/api/jobs/executions/")
        links = submitted.json()["links"]
        assert set(links) == {"self", "cancel", "resume"}
        waiting = _wait(client, links["self"], "waiting")
        assert waiting["waiting"] == {"kind": "approval"}
        assert client.post(links["resume"], json={"approved": "invalid"}).status_code == 422
        resumed = client.post(links["resume"], json={"approved": True})
        assert resumed.status_code == 202
        completed = _wait(client, links["self"], "completed")
        assert completed["result"] == {"value": 1, "owner_id": None}
        assert client.post(links["cancel"]).status_code == 200

    openapi = app.openapi()
    paths = openapi["paths"]
    assert "/api/jobs/run-durable" in paths
    assert "JobRequest" in str(paths["/api/jobs/run-durable"]["post"]["requestBody"])
    submit_response = paths["/api/jobs/run-durable"]["post"]["responses"]
    assert "JobsExecutionResult" in str(submit_response)
    assert "JobResult" in str(openapi["components"]["schemas"]["JobsExecutionResult"])
    assert "ResumeInput" in str(paths["/api/jobs/executions/{execution_id}/resume"]["post"]["requestBody"])


def test_middleware_owner_dependency_isolates_every_operation_and_idempotency() -> None:
    def current_owner_id(request: Request) -> str:
        return request.state.principal.owner_id

    app, _ = _app(current_owner_id, middleware_auth=True)
    alice = {"X-Owner": "alice", "Idempotency-Key": "same"}
    bob = {"X-Owner": "bob", "Idempotency-Key": "same"}
    with TestClient(app) as client:
        assert client.post("/api/jobs/run-durable", json={"value": 2}).status_code == 401
        submitted = client.post("/api/jobs/run-durable", json={"value": 2}, headers=alice)
        replay = client.post("/api/jobs/run-durable", json={"value": 2}, headers=alice)
        assert replay.headers["Idempotent-Replay"] == "true"
        url = submitted.json()["links"]["self"]
        assert client.get(url, headers={"X-Owner": "alice"}).status_code == 200
        assert client.get(url, headers={"X-Owner": "bob"}).status_code == 404
        assert client.post(f"{url}/cancel", headers={"X-Owner": "bob"}).status_code == 404
        assert client.post(f"{url}/resume", headers={"X-Owner": "bob"}, json={"approved": True}).status_code == 404
        conflict = client.post("/api/jobs/run-durable", json={"value": 3}, headers=alice)
        independent = client.post("/api/jobs/run-durable", json={"value": 2}, headers=bob)
        assert conflict.status_code == 409
        assert independent.status_code == 202
        assert independent.json()["execution_id"] != submitted.json()["execution_id"]


def test_owner_dependency_composes_with_dependencies_and_fails_closed() -> None:
    app, _ = _app(_dependency_owner_id)
    with TestClient(app) as client:
        assert client.get("/api/jobs/executions/missing", headers={"X-Deny": "yes"}).status_code == 403
        invalid = client.post("/api/jobs/run-durable", json={"value": 1})
        oversized = client.post(
            "/api/jobs/run-durable",
            json={"value": 1},
            headers={"X-Owner": "x" * 513},
        )
    assert invalid.status_code == 500
    assert oversized.status_code == 500


def test_unscoped_router_uses_bearer_execution_ids() -> None:
    app, _ = _app(owner_dependency=None)
    with TestClient(app) as client:
        submitted = client.post(
            "/api/jobs/run-durable",
            json={"value": 3},
            headers={"Idempotency-Key": "shared"},
        )
        replay = client.post(
            "/api/jobs/run-durable",
            json={"value": 3},
            headers={"Idempotency-Key": "shared"},
        )
        assert submitted.status_code == 202
        assert replay.headers["Idempotent-Replay"] == "true"
        assert client.get(submitted.json()["links"]["self"]).status_code == 200
