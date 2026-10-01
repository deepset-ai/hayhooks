"""Portable durable FastAPI contract."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from hayhooks.durable import DurableContext, create_durable_router
from hayhooks.durable.engine import (
    ExecutionLeaseLostError,
    ExecutionNotFoundError,
    ExecutionPayloadSizeError,
    InvalidExecutionTransitionError,
    PayloadKind,
)
from hayhooks.durable.runtime import DurableDeployment, RuntimeConfig
from hayhooks.durable.store import (
    ExecutionAdmissionError,
    ExecutionContentionError,
    ExecutionIdempotencyConflictError,
    ExecutionStoreCorruptionError,
    ExecutionStoreError,
    MemoryExecutionStore,
    StoreConfig,
    StreamChunk,
)

SKIPPED = object()


class CountingStore(MemoryExecutionStore):
    """Count the reads HTTP routes make; workers use none of these methods."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: Counter[str] = Counter()
        self.on_read_control: Callable[[int], None] | None = None

    async def submit(self, control, input_payload):
        self.calls["submit"] += 1
        return await super().submit(control, input_payload)

    async def read_public(self, run_id):
        self.calls["read_public"] += 1
        return await super().read_public(run_id)

    async def read_control(self, run_id):
        self.calls["read_control"] += 1
        if self.on_read_control is not None:
            self.on_read_control(self.calls["read_control"])
        return await super().read_control(run_id)


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
        store_class: type[MemoryExecutionStore] = MemoryExecutionStore,
    ) -> tuple[FastAPI, DurableDeployment]:
        store = store_class(
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
    """Seed retained display history, ending with the terminal marker, without faking a live lease."""
    if not store.config.max_stream_chunks:
        return
    encoded = [
        StreamChunk(f"0-{index}", attempt, b"", skipped=True)
        if payload is SKIPPED
        else StreamChunk(
            f"0-{index}",
            attempt,
            payload if isinstance(payload, bytes) else json.dumps(payload, separators=(",", ":")).encode(),
        )
        for index, (attempt, payload) in enumerate(chunks, start=1)
    ]
    marker = StreamChunk(f"0-{len(chunks) + 1}", 1, b"", terminal=True)
    store._chunks[run_id] = deque([*encoded, marker], maxlen=store.config.max_stream_chunks)


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
        rejected = client.get(f"/jobs/executions/{execution_id}")
        assert rejected.status_code == 500
        assert rejected.json() == {"detail": "Durable execution state is invalid"}


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
            2,
            64_000,
            [(1, {"index": 0}), (1, {"index": 1}), (1, {"index": 2})],
            "0-1",
            ["gap", "chunk", "completed"],
            [{"index": 2}],
            id="expired-cursor",
        ),
        pytest.param(
            10,
            2_000_000,
            [(1, {"index": 0}), (1, {"index": 1}), (1, {"index": 2}), (1, {"index": 3})],
            "0-1",
            ["chunk", "chunk", "chunk", "completed"],
            [{"index": 1}, {"index": 2}, {"index": 3}],
            id="paged-reconnect",
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
        pytest.param(
            10,
            64_000,
            [(1, {"index": 0}), (1, SKIPPED), (1, {"index": 2})],
            None,
            ["chunk", "chunk", "completed"],
            [{"index": 0}, {"index": 2}],
            id="skipped-entry",
        ),
        pytest.param(
            10,
            64_000,
            [(1, {"index": 0}), (1, SKIPPED), (1, {"index": 2})],
            "0-1",
            ["chunk", "completed"],
            [{"index": 2}],
            id="reconnect-past-skipped-entry",
        ),
        pytest.param(
            10,
            64_000,
            [(1, b"not-json"), (1, {"index": 1})],
            None,
            ["chunk", "completed"],
            [{"index": 1}],
            id="undecodable-chunk",
        ),
        pytest.param(
            10,
            16,
            [(1, {"blob": "x" * 1_000})],
            None,
            ["chunk", "completed"],
            [{"blob": "x" * 1_000}],
            id="lowered-chunk-limit",
        ),
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


def test_sse_delivers_a_large_chunk_after_the_write_limit_is_lowered(
    durable_app_factory, wait_for_execution, caplog
) -> None:
    app, deployment = durable_app_factory(max_stream_chunk_bytes=6_000_000)
    blob = "x" * 5_000_000

    async def stream_large_chunk(context: DurableContext, request: JobRequest) -> JobResult:
        await context.stream_chunk({"blob": blob})
        return JobResult(value=request.value, owner_id=context.owner_id)

    deployment.runner = stream_large_chunk
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 1}).json()
        wait_for_execution(client, submitted["links"]["self"], "completed")
        stream = deployment.store._chunks[submitted["execution_id"]]
        marker = stream.pop()
        stream.append(StreamChunk(marker.cursor, 1, b"not-json"))
        deployment.store._chunk_sequence += 1
        stream.append(replace(marker, cursor=f"0-{deployment.store._chunk_sequence}"))
        deployment.store.config = replace(deployment.store.config, max_stream_chunk_bytes=64 * 1024)

        events, _, _ = read_sse(client, submitted["links"]["stream"])

    chunks = [event for event in events if event["event"] == "chunk"]
    assert [event["event"] for event in events] == ["chunk", "completed"]
    assert chunks[0]["id"] == "0-1"
    assert json.loads(chunks[0]["data"])["payload"] == {"blob": blob}
    assert caplog.messages.count("Skipped an undecodable durable stream chunk") == 1


def test_projection_ignores_a_lowered_payload_limit(durable_app_factory, wait_for_execution) -> None:
    app, deployment = durable_app_factory()
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 1}).json()
        wait_for_execution(client, submitted["links"]["self"], "completed")
        deployment.store.config = replace(deployment.store.config, max_payload_bytes=8)
        response = client.get(submitted["links"]["self"])

    assert response.status_code == 200
    assert response.json()["result"] == {"value": 1, "owner_id": None}


def test_routes_read_only_what_they_project(durable_app_factory, wait_for_execution) -> None:
    app, deployment = durable_app_factory(store_class=CountingStore)
    store = cast(CountingStore, deployment.store)
    headers = {"Idempotency-Key": "same"}
    with TestClient(app) as client:
        created = client.post("/api/jobs/run-durable", json={"value": 1, "action": "wait"}, headers=headers)
        assert created.status_code == 202 and created.json()["status"] == "queued"
        assert store.calls == {"submit": 1}
        wait_for_execution(client, created.json()["links"]["self"], "waiting")

        store.calls.clear()
        replay = client.post("/api/jobs/run-durable", json={"value": 1, "action": "wait"}, headers=headers)
        assert replay.status_code == 202 and replay.headers["idempotent-replay"] == "true"
        assert replay.json()["status"] == "waiting"
        assert store.calls == {"submit": 1, "read_public": 1}

        store.calls.clear()
        canceled = client.post(created.json()["links"]["cancel"])
        assert canceled.status_code == 200 and canceled.json()["status"] == "canceled"
        assert store.calls == {"read_control": 1, "read_public": 1}


@pytest.mark.parametrize(
    ("action", "expected_events"),
    [
        pytest.param("stream", ["chunk", "completed"], id="final-chunk"),
        pytest.param("complete", ["completed"], id="no-chunks"),
    ],
)
def test_blocked_viewer_wakes_on_the_final_flush_and_terminal_marker(
    durable_app_factory, action: str, expected_events: list[str]
) -> None:
    app, deployment = durable_app_factory()
    release_runner = threading.Event()

    async def controlled_run(context: DurableContext, request: JobRequest) -> JobResult:
        await asyncio.to_thread(release_runner.wait)
        if request.action == "stream":
            await context.stream_chunk({"index": 0})
        return JobResult(value=request.value, owner_id=context.owner_id)

    deployment.runner = controlled_run
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 1, "action": action}).json()
        threading.Timer(0.2, release_runner.set).start()
        started = time.monotonic()
        events, _, _ = read_sse(client, submitted["links"]["stream"])

    assert [event["event"] for event in events] == expected_events
    assert time.monotonic() - started < 2


@pytest.mark.parametrize(
    ("cursor", "expected_events"),
    [
        pytest.param(None, ["completed"], id="fresh"),
        pytest.param("0-1", ["gap", "completed"], id="expired-cursor"),
    ],
)
def test_terminal_run_without_history_ends_without_blocking(
    durable_app_factory,
    monkeypatch,
    wait_for_execution,
    cursor: str | None,
    expected_events: list[str],
) -> None:
    app, deployment = durable_app_factory()
    monkeypatch.setattr(deployment, "wait_chunks", AsyncMock(side_effect=AssertionError("terminal streams never block")))
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 1}).json()
        wait_for_execution(client, submitted["links"]["self"], "completed")
        deployment.store._chunks.pop(submitted["execution_id"])
        headers = {"Last-Event-ID": cursor} if cursor is not None else None
        events, comments, _ = read_sse(client, submitted["links"]["stream"], headers=headers)

    assert [event["event"] for event in events] == expected_events
    assert comments == [": heartbeat"]


def test_idle_stream_reads_only_the_control_until_the_run_ends_without_a_marker(
    durable_app_factory, monkeypatch
) -> None:
    app, deployment = durable_app_factory(store_class=CountingStore)
    store = cast(CountingStore, deployment.store)
    monkeypatch.setattr("hayhooks.durable.fastapi._STREAM_BLOCK_SECONDS", 0.01)
    release_runner = threading.Event()

    async def controlled_run(context: DurableContext, request: JobRequest) -> JobResult:
        await asyncio.to_thread(release_runner.wait)
        return JobResult(value=request.value, owner_id=context.owner_id)

    deployment.runner = controlled_run
    wait_chunks = store.wait_chunks

    async def wait_without_markers(run_id: str, after: str, timeout: float) -> tuple[StreamChunk, ...]:
        return tuple(chunk for chunk in await wait_chunks(run_id, after, timeout) if not chunk.terminal)

    monkeypatch.setattr(store, "wait_chunks", wait_without_markers)
    store.on_read_control = lambda count: count == 3 and release_runner.set()
    fallback = threading.Timer(5, release_runner.set)
    fallback.start()
    try:
        with TestClient(app) as client:
            submitted = client.post("/api/jobs/run-durable", json={"value": 1}).json()
            store.calls.clear()
            events, comments, _ = read_sse(client, submitted["links"]["stream"])
    finally:
        fallback.cancel()

    assert [event["event"] for event in events] == ["completed"]
    assert store.calls["read_public"] == 1
    assert store.calls["read_control"] >= 4
    assert len(comments) == store.calls["read_control"]


@pytest.mark.parametrize(
    ("lifecycle", "expected_events"),
    [
        pytest.param("close", [], id="close-ends-streams"),
        pytest.param("quiesce", ["completed"], id="quiesce-keeps-streams-for-a-successor"),
    ],
)
def test_only_close_ends_blocked_streams(
    durable_app_factory, wait_for_execution, lifecycle: str, expected_events: list[str]
) -> None:
    app, deployment = durable_app_factory()
    # Quiescing is one-way; a new instance on the same store finishes the work that open streams follow.
    successor = DurableDeployment(
        deployment.name,
        deployment.revision,
        deployment.store,
        deployment.request_model,
        deployment.runner,
        result_model=deployment.result_model,
        resume_model=deployment.resume_model,
        config=deployment.config,
    )
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/run-durable", json={"value": 1, "action": "wait"}).json()
        wait_for_execution(client, submitted["links"]["self"], "waiting")

        async def interrupt() -> None:
            if lifecycle == "close":
                await deployment.close()
                return
            await deployment.quiesce()
            await successor.start()
            await successor.resume(submitted["execution_id"], {"approved": True}, enforce_owner=False)

        threading.Timer(0.2, client.portal.call, (interrupt,)).start()
        events, _, _ = read_sse(client, submitted["links"]["stream"])
        client.portal.call(successor.close)

    assert [event["event"] for event in events] == expected_events


def test_chunk_failures_are_display_only_and_midstream_errors_are_framed(
    durable_app_factory, monkeypatch, wait_for_execution
) -> None:
    app, deployment = durable_app_factory(max_stream_chunk_bytes=1_024)
    append_chunks = deployment.store.append_chunks
    with TestClient(app) as client:
        monkeypatch.setattr(deployment.store, "append_chunks", AsyncMock(side_effect=ExecutionStoreError("down")))
        dropped = client.post("/api/jobs/run-durable", json={"value": 1, "action": "stream"}).json()
        assert wait_for_execution(client, dropped["links"]["self"], "completed")["attempt"] == 1
        monkeypatch.setattr(deployment.store, "append_chunks", append_chunks)
        oversized = client.post("/api/jobs/run-durable", json={"value": 1, "action": "oversized"}).json()
        assert wait_for_execution(client, oversized["links"]["self"], "completed")["attempt"] == 1
        monkeypatch.setattr(deployment.store, "read_chunks", AsyncMock(side_effect=ExecutionStoreError("down")))
        events, _, _ = read_sse(client, oversized["links"]["stream"])
        assert events == [{"event": "error", "data": '{"detail":"Execution stream interrupted"}'}]


def test_corruption_is_an_internal_error_and_store_failures_are_unavailable(
    durable_app_factory, monkeypatch, wait_for_execution, caplog
) -> None:
    app, deployment = durable_app_factory(max_nonterminal=1)
    with TestClient(app) as client:
        first = client.post("/api/jobs/run-durable", json={"value": 1, "action": "wait"}).json()
        wait_for_execution(client, first["links"]["self"], "waiting")
        deployment.store._payloads[first["execution_id"]][PayloadKind.WAIT] = b"not-json"
        projected_corruption = client.get(first["links"]["self"])
        deployment.store._payloads[first["execution_id"]][PayloadKind.CHECKPOINT] = b"not-json"
        resumed_corruption = client.post(first["links"]["resume"], json={"approved": True})
        assert projected_corruption.status_code == resumed_corruption.status_code == 500
        assert (
            projected_corruption.json()
            == resumed_corruption.json()
            == {"detail": "Durable execution state is invalid"}
        )
        assert "stored checkpoint payload is invalid" in caplog.text
        admission = client.post("/api/jobs/run-durable", json={"value": 2})
        assert admission.status_code == 503 and admission.headers["retry-after"] == "1"
        monkeypatch.setattr(deployment.store, "read_public", AsyncMock(side_effect=ExecutionStoreError("down")))
        unavailable = client.get(first["links"]["self"])
        assert unavailable.status_code == 503 and "retry-after" not in unavailable.headers
        assert unavailable.json() == {"detail": "Durable execution store is unavailable"}


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        pytest.param(ExecutionNotFoundError("hidden"), 404, "Execution not found", id="not-found"),
        pytest.param(ExecutionIdempotencyConflictError("bound"), 409, "bound", id="idempotency"),
        pytest.param(InvalidExecutionTransitionError("state"), 409, "state", id="transition"),
        pytest.param(ExecutionPayloadSizeError("too big"), 422, "too big", id="payload-size"),
        pytest.param(ValueError("invalid"), 422, "invalid", id="value"),
        pytest.param(ExecutionAdmissionError("full"), 503, "full", id="admission"),
        pytest.param(
            ExecutionStoreCorruptionError("secret"), 500, "Durable execution state is invalid", id="corruption"
        ),
        pytest.param(
            ExecutionContentionError("secret"), 503, "Durable execution store is unavailable", id="contention"
        ),
        pytest.param(ExecutionStoreError("secret"), 503, "Durable execution store is unavailable", id="store"),
        pytest.param(
            ExecutionLeaseLostError("secret"), 503, "Durable execution service is unavailable", id="lease-lost"
        ),
        pytest.param(RuntimeError("secret"), 503, "Durable execution service is unavailable", id="runtime"),
    ],
)
def test_route_errors_map_to_stable_responses(
    durable_app_factory, monkeypatch, caplog, error: Exception, status_code: int, detail: str
) -> None:
    app, deployment = durable_app_factory()
    monkeypatch.setattr(deployment, "get", AsyncMock(side_effect=error))
    with TestClient(app) as client:
        response = client.get(f"/api/jobs/executions/{'a' * 32}")
    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert ("retry-after" in response.headers) is isinstance(error, ExecutionAdmissionError)
    assert "secret" not in response.text
    logged = detail in ("Durable execution state is invalid", "Durable execution service is unavailable")
    assert ("secret" in caplog.text) is logged


def test_closed_admission_is_retryable(durable_app_factory) -> None:
    app, deployment = durable_app_factory()
    with TestClient(app) as client:
        client.portal.call(deployment.quiesce)
        response = client.post("/api/jobs/run-durable", json={"value": 1})
    assert response.status_code == 503 and response.headers["retry-after"] == "1"
    assert response.json() == {"detail": "durable deployment 'jobs' is not accepting submissions"}


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
