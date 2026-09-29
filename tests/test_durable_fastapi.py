"""Portable durable FastAPI contract."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from hayhooks.durable import DurableContext, create_durable_router
from hayhooks.durable.engine import PayloadKind
from hayhooks.durable.runtime import DurableDeployment, RuntimeConfig
from hayhooks.durable.store import ExecutionStoreError, MemoryExecutionStore, StoreConfig, StreamChunk


class JobRequest(BaseModel):
    value: int
    action: Literal["complete", "wait", "stream", "oversized"] = "complete"


class JobResult(BaseModel):
    value: int
    owner_id: str | None


class ResumeInput(BaseModel):
    approved: bool


class LegacyResult(BaseModel):
    old: int


class CurrentResult(BaseModel):
    new: str


async def run_job(context: DurableContext, request: JobRequest) -> JobResult:
    resume_input = context.resume_input
    if request.action == "wait" and resume_input is None:
        await context.suspend({"kind": "approval", "message": "Continue?", "private": "hidden"})
    if request.action == "stream":
        for index in range(3):
            await context.stream_chunk({"index": index})
    if request.action == "oversized":
        await context.stream_chunk({"blob": "x" * 100_000})
    approved = resume_input is None or ResumeInput.model_validate(resume_input).approved
    return JobResult(value=request.value if approved else -1, owner_id=context.owner_id)


async def run_legacy(_context: DurableContext, _request: JobRequest) -> LegacyResult:
    return LegacyResult(old=1)


def owner_id(request: Request) -> str:
    return request.headers.get("X-Owner", "")


@pytest.fixture
def durable_app_factory() -> Iterator[Callable[..., tuple[FastAPI, DurableDeployment]]]:
    def create(
        owner_dependency=None,
        *,
        root_path: str = "",
        max_nonterminal: int = 0,
        max_stream_chunks: int = 10_000,
        max_stream_chunk_bytes: int = 64_000,
    ) -> tuple[FastAPI, DurableDeployment]:
        store = MemoryExecutionStore(
            "jobs",
            config=StoreConfig(
                lease_commit_safety_ms=10,
                max_nonterminal_executions=max_nonterminal,
                max_stream_chunks=max_stream_chunks,
                max_stream_chunk_bytes=max_stream_chunk_bytes,
            ),
        )
        deployment = DurableDeployment(
            "jobs",
            "v1",
            store,
            JobRequest,
            run_job,
            result_model=JobResult,
            resume_model=ResumeInput,
            config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300),
        )

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            await deployment.start()
            try:
                yield
            finally:
                await deployment.close()

        app = FastAPI(root_path=root_path, lifespan=lifespan)
        api = APIRouter(prefix="/api")
        api.include_router(
            create_durable_router(deployment, owner_id_dependency=owner_dependency),
            prefix="/jobs",
        )
        app.include_router(api)
        return app, deployment

    yield create


def read_sse(
    client: TestClient,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    events: list[dict[str, str]] = []
    comments = []
    current: dict[str, str] = {}
    with client.stream("GET", path, headers=headers) as response:
        response_headers = dict(response.headers)
        for line in response.iter_lines():
            if line.startswith(":"):
                comments.append(line)
            elif not line:
                if current:
                    events.append(current)
                    current = {}
                    if limit is not None and len(events) >= limit:
                        break
            else:
                key, _, value = line.partition(":")
                current[key] = value.lstrip()
    return events, comments, response_headers


def seed_stream_chunks(
    store: MemoryExecutionStore,
    run_id: str,
    chunks: list[tuple[int, object]],
) -> None:
    """Seed retained display history without pretending a terminal worker still owns a lease."""
    if not store.config.max_stream_chunks:
        return
    encoded = [
        StreamChunk(
            f"0-{index}",
            attempt,
            payload if isinstance(payload, bytes) else json.dumps(payload, separators=(",", ":")).encode(),
        )
        for index, (attempt, payload) in enumerate(chunks, start=1)
    ]
    store._chunks[run_id] = deque(encoded, maxlen=store.config.max_stream_chunks)


def test_router_is_typed_prefix_and_root_path_safe(durable_app_factory, wait_for_execution) -> None:
    app, _ = durable_app_factory(root_path="/root")
    with TestClient(app) as client:
        idempotent = client.post(
            "/api/jobs/run-durable",
            json={"value": 1},
            headers={"Idempotency-Key": "predictable"},
        )
        assert idempotent.status_code == 202
        wait_for_execution(client, idempotent.json()["links"]["self"], "completed")
        replay = client.post(
            "/api/jobs/run-durable",
            json={"value": 1},
            headers={"Idempotency-Key": "predictable"},
        )
        conflict = client.post(
            "/api/jobs/run-durable",
            json={"value": 2},
            headers={"Idempotency-Key": "predictable"},
        )
        assert replay.status_code == 200 and replay.headers["idempotent-replay"] == "true"
        assert replay.json()["execution_id"] == idempotent.json()["execution_id"]
        assert conflict.status_code == 409
        submitted = client.post("/api/jobs/run-durable", json={"value": 1, "action": "wait"})
        assert submitted.status_code == 202
        assert submitted.headers["location"].startswith("/root/api/jobs/executions/")
        links = {key: value.removeprefix("/root") for key, value in submitted.json()["links"].items()}
        assert set(links) == {"self", "cancel", "resume", "stream"}
        waiting = wait_for_execution(client, links["self"], "waiting")
        assert waiting["waiting"] == {"kind": "approval", "message": "Continue?"}
        assert client.post(links["resume"], json={"approved": "invalid"}).status_code == 422
        assert client.post(links["resume"], json={"approved": True}).status_code == 202
        completed = wait_for_execution(client, links["self"], "completed")
        assert completed["result"] == {"value": 1, "owner_id": None}
        assert client.post(links["resume"], json={"approved": True}).status_code == 409
        assert client.post(links["cancel"]).status_code == 200
        assert client.get(links["stream"], headers={"Last-Event-ID": "invalid"}).status_code == 422
        assert client.get(links["stream"], headers={"Last-Event-ID": ""}).status_code == 422
        events, comments, headers = read_sse(client, links["stream"])
        assert events[-1]["event"] == "completed"
        assert comments and headers["cache-control"] == "no-cache" and headers["x-accel-buffering"] == "no"

    openapi = app.openapi()
    paths = openapi["paths"]
    assert "JobRequest" in str(paths["/api/jobs/run-durable"]["post"]["requestBody"])
    assert "ExecutionResult" in str(paths["/api/jobs/run-durable"]["post"]["responses"])
    assert "JobsExecutionResult" not in openapi["components"]["schemas"]
    assert "ResumeInput" in str(paths["/api/jobs/executions/{execution_id}/resume"]["post"]["requestBody"])


def test_terminal_result_remains_readable_after_result_schema_revision() -> None:
    store = MemoryExecutionStore("jobs", config=StoreConfig(lease_commit_safety_ms=10))
    legacy = DurableDeployment(
        "jobs",
        "v1",
        store,
        JobRequest,
        run_legacy,
        result_model=LegacyResult,
        config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300),
    )

    async def complete_legacy_execution() -> str:
        await legacy.start()
        try:
            submitted = await legacy.submit({"value": 1})
            for _ in range(200):
                stored = await store.read(submitted.control.run_id)
                if stored is not None and stored.control.terminal:
                    return submitted.control.run_id
                await asyncio.sleep(0.005)
            raise AssertionError
        finally:
            await legacy.close()

    execution_id = asyncio.run(complete_legacy_execution())
    current = DurableDeployment(
        "jobs",
        "v2",
        store,
        JobRequest,
        run_job,
        result_model=CurrentResult,
        config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await current.start()
        try:
            yield
        finally:
            await current.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(create_durable_router(current, owner_id_dependency=None), prefix="/jobs")
    with TestClient(app) as client:
        response = client.get(f"/jobs/executions/{execution_id}")
        assert response.status_code == 200
        assert response.json()["result"] == {"old": 1}

        store._controls[execution_id] = replace(
            store._controls[execution_id],
            definition_revision="v2",
        )
        assert client.get(f"/jobs/executions/{execution_id}").status_code == 503


@pytest.mark.parametrize(
    ("method", "suffix", "body"),
    [
        pytest.param("get", "", None, id="inspect"),
        pytest.param("post", "/cancel", None, id="cancel"),
        pytest.param("post", "/resume", {"approved": True}, id="resume"),
        pytest.param("get", "/stream", None, id="stream"),
    ],
)
def test_owner_mismatch_is_always_hidden(durable_app_factory, method: str, suffix: str, body: object) -> None:
    app, _ = durable_app_factory(owner_id)
    with TestClient(app) as client:
        submitted = client.post(
            "/api/jobs/run-durable",
            json={"value": 1, "action": "wait"},
            headers={"X-Owner": "alice"},
        ).json()
        response = client.request(
            method,
            f"{submitted['links']['self']}{suffix}",
            json=body,
            headers={"X-Owner": "bob"},
        )
        assert response.status_code == 404


def test_owner_scopes_idempotency_and_invalid_values_fail_closed(durable_app_factory, wait_for_execution) -> None:
    app, _ = durable_app_factory(owner_id)
    alice = {"X-Owner": "alice", "Idempotency-Key": "same"}
    bob = {"X-Owner": "bob", "Idempotency-Key": "same"}
    with TestClient(app) as client:
        assert client.post("/api/jobs/run-durable", json={"value": 1}).status_code == 500
        submitted = client.post("/api/jobs/run-durable", json={"value": 1}, headers=alice)
        wait_for_execution(client, submitted.json()["links"]["self"], "completed", headers=alice)
        replay = client.post("/api/jobs/run-durable", json={"value": 1}, headers=alice)
        conflict = client.post("/api/jobs/run-durable", json={"value": 2}, headers=alice)
        independent = client.post("/api/jobs/run-durable", json={"value": 2}, headers=bob)
        wait_for_execution(client, independent.json()["links"]["self"], "completed", headers=bob)
        oversized = client.post(
            "/api/jobs/run-durable",
            json={"value": 1},
            headers={"X-Owner": "alice", "Idempotency-Key": "x" * 513},
        )
        assert replay.headers["idempotent-replay"] == "true"
        assert replay.status_code == 200
        assert replay.json()["execution_id"] == submitted.json()["execution_id"]
        assert conflict.status_code == 409
        assert independent.status_code == 202
        assert independent.json()["execution_id"] != submitted.json()["execution_id"]
        assert (
            client.post(independent.json()["links"]["resume"], json={"approved": True}, headers=bob).status_code == 409
        )
        assert oversized.status_code == 422


@pytest.mark.parametrize(
    ("max_chunks", "chunk_bytes", "chunks", "cursor", "expected_events", "expected_payloads"),
    [
        pytest.param(
            10,
            64_000,
            [(1, {"index": 0}), (1, {"index": 1}), (1, {"index": 2})],
            "0-1",
            ["chunk", "chunk", "completed"],
            [{"index": 1}, {"index": 2}],
            id="reconnect",
        ),
        pytest.param(
            1,
            64_000,
            [(1, {"index": 0}), (1, {"index": 1}), (1, {"index": 2})],
            "0-1",
            ["gap", "chunk", "completed"],
            [{"index": 2}],
            id="expired-cursor",
        ),
        pytest.param(
            10,
            64_000,
            [(0, {"source": "stale"}), (1, {"source": "current"})],
            None,
            ["chunk", "completed"],
            [{"source": "current"}],
            id="stale-attempt",
        ),
        pytest.param(
            10,
            64_000,
            [(1, b'{\n  "event": "forged",\n  "index": 0\n}')],
            None,
            ["chunk", "completed"],
            [{"event": "forged", "index": 0}],
            id="reframed-json",
        ),
        pytest.param(
            10,
            2_000_000,
            [(1, {"index": 0}), (1, {"index": 1}), (1, {"index": 2})],
            None,
            ["chunk", "chunk", "chunk", "completed"],
            [{"index": 0}, {"index": 1}, {"index": 2}],
            id="terminal-backlog",
        ),
        pytest.param(0, 64_000, [(1, {"index": 0})], None, ["completed"], [], id="disabled-log"),
    ],
)
def test_stream_resume_gap_fencing_and_drain(
    durable_app_factory,
    wait_for_execution,
    max_chunks: int,
    chunk_bytes: int,
    chunks: list[tuple[int, object]],
    cursor: str | None,
    expected_events: list[str],
    expected_payloads: list[dict[str, object]],
) -> None:
    app, deployment = durable_app_factory(
        max_stream_chunks=max_chunks,
        max_stream_chunk_bytes=chunk_bytes,
    )
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 1}).json()
        completed = wait_for_execution(client, submitted["links"]["self"], "completed")
        seed_stream_chunks(deployment.store, completed["execution_id"], chunks)
        headers = {"Last-Event-ID": cursor} if cursor is not None else None
        events, _, _ = read_sse(client, submitted["links"]["stream"], headers=headers)
        payloads = [json.loads(event["data"])["payload"] for event in events if event["event"] == "chunk"]
        assert [event["event"] for event in events] == expected_events
        assert payloads == expected_payloads


def test_stream_drains_chunk_committed_immediately_before_terminal(durable_app_factory, monkeypatch) -> None:
    app, deployment = durable_app_factory()
    chunk_written = threading.Event()
    release_runner = threading.Event()

    async def controlled_run(context: DurableContext, request: JobRequest) -> JobResult:
        await context.stream_chunk({"index": 0})
        chunk_written.set()
        await asyncio.to_thread(release_runner.wait)
        return JobResult(value=request.value, owner_id=context.owner_id)

    deployment.runner = controlled_run
    read_chunks = deployment.store.read_chunks
    missed_once = False

    async def miss_once(run_id: str, cursor: str):
        nonlocal missed_once
        if not missed_once:
            missed_once = True
            release_runner.set()
            for _ in range(200):
                stored = await deployment.store.read(run_id)
                if stored is not None and stored.control.terminal:
                    return []
                await asyncio.sleep(0.005)
            raise AssertionError("execution did not complete")
        return await read_chunks(run_id, cursor)

    monkeypatch.setattr(deployment.store, "read_chunks", miss_once)
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 1}).json()
        assert chunk_written.wait(timeout=1)
        events, _, _ = read_sse(client, submitted["links"]["stream"])

    assert [event["event"] for event in events] == ["chunk", "completed"]


def test_chunk_failures_are_display_only_and_midstream_errors_are_framed(
    durable_app_factory, monkeypatch, wait_for_execution
) -> None:
    app, deployment = durable_app_factory(max_stream_chunk_bytes=1_024)
    append_chunk = deployment.store.append_chunk
    with TestClient(app) as client:
        monkeypatch.setattr(deployment.store, "append_chunk", AsyncMock(side_effect=ExecutionStoreError("down")))
        dropped = client.post("/api/jobs/run-durable", json={"value": 1, "action": "stream"}).json()
        assert wait_for_execution(client, dropped["links"]["self"], "completed")["attempt"] == 1
        monkeypatch.setattr(deployment.store, "append_chunk", append_chunk)
        oversized = client.post("/api/jobs/run-durable", json={"value": 1, "action": "oversized"}).json()
        assert wait_for_execution(client, oversized["links"]["self"], "completed")["attempt"] == 1
        monkeypatch.setattr(deployment.store, "read_chunks", AsyncMock(side_effect=ExecutionStoreError("down")))
        events, _, _ = read_sse(client, oversized["links"]["stream"])
        assert events == [{"event": "error", "data": '{"detail":"Execution stream interrupted"}'}]


def test_admission_and_store_failures_are_service_unavailable(
    durable_app_factory, monkeypatch, wait_for_execution
) -> None:
    app, deployment = durable_app_factory(max_nonterminal=1)
    with TestClient(app) as client:
        first = client.post("/api/jobs/run-durable", json={"value": 1, "action": "wait"}).json()
        wait_for_execution(client, first["links"]["self"], "waiting")
        deployment.store._payloads[first["execution_id"]][PayloadKind.WAIT] = b"not-json"
        projected_corruption = client.get(first["links"]["self"])
        deployment.store._payloads[first["execution_id"]][PayloadKind.CHECKPOINT] = b"not-json"
        resumed_corruption = client.post(first["links"]["resume"], json={"approved": True})
        assert projected_corruption.status_code == resumed_corruption.status_code == 503
        assert (
            projected_corruption.json()
            == resumed_corruption.json()
            == {"detail": "Durable execution store is unavailable"}
        )
        admission = client.post("/api/jobs/run-durable", json={"value": 2})
        assert admission.status_code == 503 and admission.headers["retry-after"] == "1"
        monkeypatch.setattr(deployment.store, "read", AsyncMock(side_effect=ExecutionStoreError("down")))
        unavailable = client.get(first["links"]["self"])
        assert unavailable.status_code == 503
        assert unavailable.json() == {"detail": "Durable execution store is unavailable"}


def test_waiting_stream_disconnect_does_not_cancel(durable_app_factory, wait_for_execution) -> None:
    app, _ = durable_app_factory()
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 1, "action": "wait"}).json()
        wait_for_execution(client, submitted["links"]["self"], "waiting")
        messages = []

        async def disconnect() -> None:
            streamed = asyncio.Event()

            async def receive() -> dict[str, str]:
                await streamed.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                messages.append(message)
                if message["type"] == "http.response.body":
                    streamed.set()

            path = submitted["links"]["stream"]
            await app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.3"},
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode(),
                    "query_string": b"",
                    "root_path": "",
                    "headers": [],
                    "client": ("test", 1),
                    "server": ("testserver", 80),
                    "state": {},
                },
                receive,
                send,
            )

        asyncio.run(disconnect())
        assert any(message["type"] == "http.response.body" for message in messages)
        assert client.get(submitted["links"]["self"]).json()["status"] == "waiting"
