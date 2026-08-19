"""Standalone contract tests for the public FastAPI durable adapter."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Annotated, Any

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
from hayhooks.durable.models import DEFAULT_MAX_STREAM_CHUNK_BYTES, ExecutionStoreError
from tests.durable_helpers import read_sse_events, wait_for_status

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


_STREAMED_VALUE = 7
_OVERSIZED_CHUNK_VALUE = 8
_RETRIED_STREAM_VALUE = 9
_BROKEN_CHUNK_VALUE = 10


class _BrokenChunk:
    def to_dict(self) -> dict[str, Any]:
        msg = "broken chunk serializer"
        raise RuntimeError(msg)


async def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
    """Stand in for a chunk backend that is down."""
    msg = "chunk backend is down"
    raise ExecutionStoreError(msg)


class JobWrapper(BasePipelineWrapper):
    durable_revision = "portable-job-v1"
    durable_resume_model = ResumeInput

    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.stale_chunk_task: asyncio.Task[None] | None = None

    async def run_durable_async(self, context: DurableContext, request: JobRequest) -> JobResult:
        if request.value == 0 and context.resume_input is None:
            await context.suspend({"kind": "approval", "private": "hidden"})
        if request.value == _STREAMED_VALUE:
            for index in range(3):
                await context.stream_chunk({"index": index})
        if request.value == _OVERSIZED_CHUNK_VALUE:
            await context.stream_chunk({"blob": "x" * (DEFAULT_MAX_STREAM_CHUNK_BYTES + 1)})
        if request.value == _BROKEN_CHUNK_VALUE:
            await context.stream_chunk(_BrokenChunk())
        if request.value == _RETRIED_STREAM_VALUE:
            if context.attempt == 1:

                async def append_stale_chunk() -> None:
                    # The replacement attempt is already current, but has not
                    # emitted yet: highest-attempt-seen alone would leak this.
                    await asyncio.sleep(0.05)
                    await context.stream_chunk({"source": "stale"})

                self.stale_chunk_task = asyncio.create_task(append_stale_chunk())
                await context.retry("retry streaming", delay=0.01)
            await asyncio.sleep(0.1)
            await context.stream_chunk({"source": "current"})
            assert self.stale_chunk_task is not None
            await self.stale_chunk_task
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
    max_stream_chunks: int = 10_000,
    max_stream_chunk_bytes: int = DEFAULT_MAX_STREAM_CHUNK_BYTES,
) -> tuple[FastAPI, DurableRuntime]:
    durable_settings = DurableSettings(
        durable_store="memory",
        durable_poll_interval=0.05,
        durable_max_stream_chunks=max_stream_chunks,
        durable_max_stream_chunk_bytes=max_stream_chunk_bytes,
    )
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


def test_public_router_is_typed_prefix_safe_and_supports_all_routes() -> None:
    app, _ = _app(owner_dependency=None)
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 0})
        assert submitted.status_code == 202
        assert submitted.headers["Location"].startswith("/api/jobs/executions/")
        links = submitted.json()["links"]
        assert set(links) == {"self", "cancel", "resume", "stream"}
        waiting = wait_for_status(client, links["self"], "waiting")
        assert waiting["waiting"] == {"kind": "approval"}
        assert client.post(links["resume"], json={"approved": "invalid"}).status_code == 422
        resumed = client.post(links["resume"], json={"approved": True})
        assert resumed.status_code == 202
        completed = wait_for_status(client, links["self"], "completed")
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
        assert client.get(f"{url}/stream", headers={"X-Owner": "bob"}).status_code == 404
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


def test_execution_stream_replays_from_last_event_id_and_ends_with_the_terminal_event() -> None:
    app, _ = _app(owner_dependency=None)
    with TestClient(app) as client:
        links = client.post("/api/jobs/run-durable", json={"value": _STREAMED_VALUE}).json()["links"]
        assert client.get(links["stream"], headers={"Last-Event-ID": "not-a-cursor"}).status_code == 422

        first = read_sse_events(client, links["stream"], limit=1)[0]
        assert first["event"] == "chunk"
        assert json.loads(first["data"]) == {"attempt": 1, "payload": {"index": 0}}

        # Reattaching from the first entry ID must not repeat it, and the terminal
        # event has to arrive on the same stream after the trailing chunks.
        reattached = read_sse_events(client, links["stream"], headers={"Last-Event-ID": first["id"]})
        chunks = [event for event in reattached if event["event"] == "chunk"]
        assert [json.loads(event["data"])["payload"]["index"] for event in chunks] == [1, 2]
        assert reattached[-1]["event"] == "completed"
        assert json.loads(reattached[-1]["data"])["result"] == {"value": _STREAMED_VALUE + 1, "owner_id": None}


def test_zero_chunk_bound_disables_the_log_without_breaking_the_stream() -> None:
    app, _ = _app(owner_dependency=None, max_stream_chunks=0)
    with TestClient(app) as client:
        links = client.post("/api/jobs/run-durable", json={"value": _STREAMED_VALUE}).json()["links"]
        events = read_sse_events(client, links["stream"])
        assert [event["event"] for event in events] == ["completed"]
        assert json.loads(events[0]["data"])["result"] == {"value": _STREAMED_VALUE + 1, "owner_id": None}


def test_stream_signals_an_expired_cursor_and_replays_the_retained_tail() -> None:
    app, _ = _app(owner_dependency=None, max_stream_chunks=1)
    with TestClient(app) as client:
        links = client.post("/api/jobs/run-durable", json={"value": _STREAMED_VALUE}).json()["links"]
        events = read_sse_events(client, links["stream"], headers={"Last-Event-ID": "0-1"})

    assert [event["event"] for event in events] == ["gap", "chunk", "completed"]
    assert json.loads(events[1]["data"])["payload"] == {"index": 2}


def test_stream_ignores_chunks_from_an_attempt_older_than_the_execution() -> None:
    app, _ = _app(owner_dependency=None)
    with TestClient(app) as client:
        links = client.post("/api/jobs/run-durable", json={"value": _RETRIED_STREAM_VALUE}).json()["links"]
        events = read_sse_events(client, links["stream"])

    chunks = [json.loads(event["data"]) for event in events if event["event"] == "chunk"]
    assert chunks == [{"attempt": 2, "payload": {"source": "current"}}]


def test_a_failed_chunk_append_never_fails_or_replays_the_execution() -> None:
    """Chunks live outside the durable fence, so losing one must cost only the chunk."""
    app, runtime = _app(owner_dependency=None)
    with TestClient(app) as client:
        oversized = client.post("/api/jobs/run-durable", json={"value": _OVERSIZED_CHUNK_VALUE}).json()
        assert wait_for_status(client, oversized["links"]["self"], "completed")["error"] is None

        broken = client.post("/api/jobs/run-durable", json={"value": _BROKEN_CHUNK_VALUE}).json()
        assert wait_for_status(client, broken["links"]["self"], "completed")["error"] is None

        runtime.deployment("jobs").store.core.append_chunk = _unavailable
        streamed = client.post("/api/jobs/run-durable", json={"value": _STREAMED_VALUE}).json()
        assert wait_for_status(client, streamed["links"]["self"], "completed")["attempt"] == 1


def test_configured_chunk_byte_limit_drops_oversized_chunks() -> None:
    """The chunk byte cap is a durable setting, and dropping still never fails the run."""
    app, _ = _app(owner_dependency=None, max_stream_chunk_bytes=1_024)
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": _OVERSIZED_CHUNK_VALUE}).json()
        assert wait_for_status(client, submitted["links"]["self"], "completed")["error"] is None
        events = read_sse_events(client, submitted["links"]["stream"])
        assert [event["event"] for event in events] == ["completed"]


def test_stream_rejects_an_out_of_range_cursor_and_frames_a_mid_stream_failure() -> None:
    app, runtime = _app(owner_dependency=None)
    with TestClient(app) as client:
        links = client.post("/api/jobs/run-durable", json={"value": _STREAMED_VALUE}).json()["links"]
        # A 64-bit-overflowing cursor is caught before the response begins.
        assert client.get(links["stream"], headers={"Last-Event-ID": f"{2**64}-0"}).status_code == 422

        with client.stream("GET", links["stream"]) as response:
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["x-accel-buffering"] == "no"

        runtime.deployment("jobs").store.core.read_chunks = _unavailable
        # Headers are already sent, so the break has to arrive as an SSE event.
        assert read_sse_events(client, links["stream"]) == [
            {"event": "error", "data": '{"detail":"Execution stream interrupted"}'}
        ]
