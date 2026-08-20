"""Real-Redis contract tests for the isolated durable namespace."""

from __future__ import annotations

import asyncio
import os
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from haystack import Pipeline
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper
from hayhooks.durable import DurableContext, DurableRuntime, DurableSettings, create_durable_router
from hayhooks.durable.backend import CHUNK_CURSOR_START, ChunkCursorExpiredError
from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionLeaseLostError,
    ExecutionPayloadSizeError,
    Heartbeat,
    RecoverExpiredLease,
    RequestCancellation,
    ScheduleRetry,
)
from hayhooks.durable.manager import DurableExecutionManager
from hayhooks.durable.models import (
    ExecutionAdmissionError,
    ExecutionCheckpoint,
    ExecutionKind,
    ExecutionRecord,
    ExecutionStatus,
)
from hayhooks.durable.redis import RedisExecutionStore
from hayhooks.durable.store import ExecutionStore, RedisExecutionStoreProvider
from tests.durable_contract import assert_store_contract, contract_config, control
from tests.durable_helpers import wait_for_record, wait_until_async

pytestmark = pytest.mark.integration


@pytest.fixture
async def store(isolated_redis):
    redis, prefix = isolated_redis
    config = contract_config(key_prefix=f"{prefix}:durable", terminal_ttl_seconds=60)
    durable = RedisExecutionStore(redis, deployment="integration", config=config)
    await durable.initialize()
    yield redis, durable


async def _claim(durable: RedisExecutionStore, *, worker: str = "worker", lease_ms: int = 1_000):
    run_id = await durable.read_candidate()
    assert run_id is not None
    return run_id, await durable.transition(run_id, Claim(worker, 0, lease_ms, 3, "rev-1"), candidate=True)


async def test_redis_store_matches_shared_contract(store) -> None:
    _, durable = store
    await assert_store_contract(durable)


async def test_reading_candidate_is_non_destructive(store) -> None:
    _, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    assert await durable.read_candidate() == "run_1"
    assert await durable.read_candidate() == "run_1"


async def test_three_concurrent_claimers_create_one_live_owner(store) -> None:
    redis, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id = await durable.read_candidate()
    assert run_id is not None
    await asyncio.gather(
        *(
            durable.transition(run_id, Claim(f"worker-{index}", 0, 10_000, 3, "rev-1"), candidate=True)
            for index in range(3)
        )
    )
    current = await durable.get(run_id)
    assert current is not None and current.status is ExecutionStatus.RUNNING
    assert current.fence == current.run_attempt == 1
    assert await redis.zcard(durable.keys.lease_expiry) == 1
    assert await redis.zcard(durable.keys.runnable) == 0


async def test_delayed_work_is_invisible_until_the_redis_deadline(store) -> None:
    redis, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, claimed = await _claim(durable)
    await durable.transition(run_id, ScheduleRetry(claimed.next_control.fence, "worker", 0, 250, 2, b"retry"))
    seconds, micros = await redis.time()
    now_ms = int(seconds) * 1_000 + int(micros) // 1_000
    score = await redis.zscore(durable.keys.runnable, run_id)
    assert score is not None and score > now_ms
    assert await durable.read_candidate() is None
    await asyncio.sleep(0.3)
    assert await durable.read_candidate() == run_id


async def test_heartbeat_updates_only_the_lease_path(store, monkeypatch) -> None:
    _, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, claimed = await _claim(durable)
    before = claimed.next_control.lease_expires_at_ms
    monkeypatch.setattr(
        durable,
        "_apply_plan",
        lambda *_args, **_kwargs: pytest.fail("heartbeat rewrote the full execution control"),
    )

    heartbeat = await durable.transition(run_id, Heartbeat(1, "worker", 0, 2_000))

    assert heartbeat.next_control.lease_expires_at_ms is not None
    assert heartbeat.next_control.lease_expires_at_ms > before


async def test_hundred_concurrent_submissions_succeed_when_admission_is_disabled(store) -> None:
    _, durable = store

    async def submit(index: int):
        return await durable.submit(
            control(f"run_{index}", idempotency_digest=f"{index:064x}", binding_digest="b" * 64),
            b"{}",
            binding_digest="b" * 64,
        )

    results = await asyncio.gather(*(submit(index) for index in range(100)))
    assert all(result.created for result in results)
    assert (await durable.operational_counts())["nonterminal"] == 100


async def test_global_admission_allows_replay_and_releases_capacity(store) -> None:
    redis, durable = store
    limited = RedisExecutionStore(
        redis,
        deployment="integration",
        config=replace(
            durable.config,
            key_prefix=f"{durable.config.key_prefix}:limited",
            max_nonterminal_executions=1,
        ),
    )
    await limited.initialize()
    assert (await limited.submit(control(), b"{}", binding_digest="b" * 64)).created
    replay = await limited.submit(control("replay"), b"{}", binding_digest="b" * 64)
    assert not replay.created and replay.control.run_id == "run_1"

    second = control("run_2", idempotency_digest="c" * 64, binding_digest="d" * 64)
    with pytest.raises(ExecutionAdmissionError):
        await limited.submit(second, b"{}", binding_digest="d" * 64)
    await limited.transition("run_1", RequestCancellation(0, "done"))
    assert (await limited.submit(second, b"{}", binding_digest="d" * 64)).created


async def test_expired_lease_reenters_runnable(store) -> None:
    _, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, claimed = await _claim(durable, lease_ms=20)
    await asyncio.sleep(0.03)
    await durable.maintain(lambda fence, deadline: RecoverExpiredLease(0, fence, deadline, 3, "rev-1"))
    recovered = await durable.get(run_id)
    assert recovered is not None and recovered.status is ExecutionStatus.QUEUED
    assert await durable.read_candidate() == run_id
    assert claimed.next_control.fence == 1


async def test_terminal_ttl_retains_data_and_rejects_stale_fences(store) -> None:
    redis, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, _ = await _claim(durable)
    checkpointed = await durable.transition(run_id, Checkpoint(1, "worker", 0, 1_000, b"checkpoint"))
    await durable.append_chunk(run_id, 1, b'{"c":1}')
    with pytest.raises(ExecutionLeaseLostError):
        await durable.transition(run_id, Complete(0, "worker", 0, b"result"))
    terminal = await durable.transition(run_id, Complete(checkpointed.next_control.fence, "worker", 0, b"result"))
    assert terminal.next_control.terminal
    async with redis.pipeline(transaction=False) as pipe:
        for key in durable._execution_keys(run_id):
            pipe.pttl(key)
        ttls = await pipe.execute()
    assert all(ttl == -2 or ttl > 0 for ttl in ttls)
    assert await redis.pttl(durable.keys.idempotency("a" * 64)) > 0
    assert await redis.pttl(durable.keys.chunks(run_id)) > 0


async def test_stale_append_after_terminal_gets_a_ttl(store) -> None:
    """A zombie worker's first chunk after terminal completion must not leak the key."""
    redis, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, _ = await _claim(durable)
    claimed = await durable.transition(run_id, Checkpoint(1, "worker", 0, 1_000, b"checkpoint"))
    terminal = await durable.transition(run_id, Complete(claimed.next_control.fence, "worker", 0, b"result"))
    assert terminal.next_control.terminal
    # The run never streamed, so the terminal transaction had no chunks key to
    # expire; the stale append below creates it and must set the TTL itself.
    await durable.append_chunk(run_id, 1, b'{"late":true}')
    assert await redis.pttl(durable.keys.chunks(run_id)) > 0


async def test_live_chunk_log_has_a_rolling_ttl(store) -> None:
    """Best-effort display history must stay bounded even before terminal cleanup."""
    redis, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, _ = await _claim(durable)
    await durable.append_chunk(run_id, 1, b'{"live":true}')
    assert await redis.pttl(durable.keys.chunks(run_id)) > 0


async def test_append_after_the_control_expired_still_gets_a_ttl(store) -> None:
    """Past the terminal TTL there is no control left to consult, and no EXPIRE to come."""
    redis, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, _ = await _claim(durable)
    await redis.delete(durable.keys.control(run_id))
    await durable.append_chunk(run_id, 1, b'{"orphan":true}')
    assert await redis.pttl(durable.keys.chunks(run_id)) > 0


async def test_chunk_read_is_bounded_and_resumes_from_the_last_entry(store, monkeypatch) -> None:
    """One read must not fan in a whole log; the cursor carries the rest."""
    _, durable = store
    monkeypatch.setattr("hayhooks.durable.backend.CHUNK_READ_MAX_BYTES", 2 * durable.config.max_stream_chunk_bytes)
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, _ = await _claim(durable)
    for index in range(3):
        await durable.append_chunk(run_id, 1, b'{"i":%d}' % index)

    first = await durable.read_chunks(run_id, CHUNK_CURSOR_START)
    assert [data for _, _, data in first] == [b'{"i":0}', b'{"i":1}']
    rest = await durable.read_chunks(run_id, first[-1][0])
    assert [data for _, _, data in rest] == [b'{"i":2}']

    await durable.append_chunk(run_id, 1, b'{"i":3}')
    await durable.append_chunk(run_id, 1, b'{"i":4}')
    with pytest.raises(ChunkCursorExpiredError):
        await durable.read_chunks(run_id, first[0][0])
    with pytest.raises(ChunkCursorExpiredError):
        await durable.read_chunks(run_id, "9999999999999-0")


async def test_chunk_read_returns_immediately_instead_of_holding_a_connection(store) -> None:
    """A blocking read would pin one pool connection per attached viewer."""
    _, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, _ = await _claim(durable)
    started = time.monotonic()
    assert await durable.read_chunks(run_id, CHUNK_CURSOR_START) == []
    assert time.monotonic() - started < 0.2
    await durable.append_chunk(run_id, 1, b'{"live":true}')
    assert [data for _, _, data in await durable.read_chunks(run_id, CHUNK_CURSOR_START)] == [b'{"live":true}']


async def test_redis_rejects_oversized_payload_without_partial_write(store) -> None:
    _, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, claim = await _claim(durable)
    with pytest.raises(ExecutionPayloadSizeError):
        await durable.transition(run_id, Checkpoint(claim.next_control.fence, "worker", 0, 1_000, b"x" * 65))
    assert await durable.get(run_id) == claim.next_control


async def test_redis_adapter_runs_the_public_manager_contract(store) -> None:
    redis, durable = store
    config = replace(
        durable.config,
        key_prefix=f"{durable.config.key_prefix}:adapter",
        max_input_bytes=512,
        max_checkpoint_bytes=512,
        max_result_bytes=512,
        max_error_bytes=512,
        max_wait_bytes=512,
        max_progress_event_bytes=256,
    )
    adapter_store = ExecutionStore(
        RedisExecutionStore(redis, deployment="adapter", config=config),
        definition_revision="rev-1",
        lease_duration_ms=10_000,
        max_run_attempts=3,
        max_progress_events=2,
        max_record_bytes=512,
    )

    async def runner(context):
        context.record.application_state["phase"] = "running"
        context.record.checkpoint = ExecutionCheckpoint(ExecutionKind.PIPELINE, {"component": "search"})
        await context.checkpoint()
        await context.report_progress("started")
        return {"answer": "done"}

    manager = DurableExecutionManager("adapter", adapter_store, runner, adapter=object(), poll_interval=0.001)
    await manager.start()
    try:
        assert await adapter_store.submit(
            ExecutionRecord(
                execution_id="public-run",
                execution_kind=ExecutionKind.PIPELINE,
                deployment_name="adapter",
                definition_revision="rev-1",
                validated_input={"question": "hello"},
                operation_fingerprint="request-fingerprint",
                max_progress_events=2,
                max_record_bytes=512,
            )
        )
        record = await wait_for_record(
            adapter_store, "public-run", message="Redis adapter did not complete the public manager execution"
        )
    finally:
        await manager.close()

    assert record.status is ExecutionStatus.COMPLETED
    assert record.result == {"answer": "done"}


class _StreamJobRequest(BaseModel):
    value: int = 0


class _StreamJobResult(BaseModel):
    value: int


class _WaitingJobWrapper(BasePipelineWrapper):
    """Parks in ``waiting`` so an attached stream stays open for the whole test."""

    durable_revision = "stream-pressure-v1"

    def setup(self) -> None:
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: _StreamJobRequest) -> _StreamJobResult:
        if context.resume_input is None:
            await context.suspend({"kind": "approval"})
        return _StreamJobResult(value=request.value)


@asynccontextmanager
async def _streaming_server(prefix: str, *, max_connections: int):
    """
    Serve one Redis-backed durable deployment over real HTTP on a deliberately small pool.

    A real server rather than an ASGI transport because ``httpx.ASGITransport`` buffers
    the whole response body, and an execution stream only ends when the execution does.
    Uvicorn runs on the test's own loop so the engine and the routes share one Redis pool.
    """
    from redis.asyncio import Redis

    client = Redis.from_url(
        os.environ["HAYHOOKS_TEST_REDIS_URL"], decode_responses=False, max_connections=max_connections
    )
    durable_settings = DurableSettings(durable_store="redis", durable_poll_interval=0.05)
    provider = RedisExecutionStoreProvider(
        redis=client, key_prefix=f"{prefix}:pressure", close_redis=False, durable_settings=durable_settings
    )
    runtime = DurableRuntime(provider)
    wrapper = _WaitingJobWrapper()
    wrapper.setup()
    app = FastAPI()
    app.include_router(
        create_durable_router(runtime.deployment("jobs", wrapper), owner_id_dependency=None), prefix="/jobs"
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    serving = asyncio.create_task(server.serve())
    await runtime.start()
    try:
        await wait_until_async(lambda: server.started, "the test server never started")
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as http:
            yield http
    finally:
        server.should_exit = True
        await serving
        await runtime.close()
        await client.aclose()


async def _parked_execution(http: httpx.AsyncClient) -> dict:
    submitted = await http.post("/jobs/run-durable", json={"value": 1})
    assert submitted.status_code == 202, submitted.text
    links = submitted.json()["links"]

    async def parked() -> bool:
        return (await http.get(links["self"])).json()["status"] == "waiting"

    await wait_until_async(parked, "the execution never parked in waiting", delay=0.05)
    return links


async def _drain(http: httpx.AsyncClient, url: str, attached: asyncio.Event) -> str:
    """Attach one viewer and keep consuming, so its generator stays inside the chunk read."""
    try:
        async with http.stream("GET", url) as response:
            if response.status_code != 200:
                return f"HTTP {response.status_code}"
            async for line in response.aiter_lines():
                attached.set()
                if line.startswith("event: error"):
                    return "error event"
        return "stream ended"
    except Exception as error:
        return type(error).__name__
    finally:
        # A refused viewer has to release the barrier too, so the assertion below
        # reports why it failed instead of the whole test timing out.
        attached.set()


async def test_concurrent_streams_do_not_starve_the_engine_of_connections(isolated_redis) -> None:
    """A draining viewer must not pin a pool connection: the engine shares that pool."""
    pool = 8
    viewers = pool * 3
    _, prefix = isolated_redis
    async with _streaming_server(prefix, max_connections=pool) as http:
        links = await _parked_execution(http)
        readers: list[asyncio.Task[str]] = []
        try:
            # One at a time, so this measures connections held for the life of a
            # stream rather than a burst of simultaneous attaches, which any pool
            # smaller than the burst refuses whatever the stream does afterwards.
            for _ in range(viewers):
                attached = asyncio.Event()
                readers.append(asyncio.create_task(_drain(http, links["stream"], attached)))
                await asyncio.wait_for(attached.wait(), timeout=10)
            assert [reader.result() for reader in readers if reader.done()] == []
            # Every viewer is attached and draining; the engine must still get connections.
            assert (await _parked_execution(http))["self"] != links["self"]
        finally:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
