"""Real-Redis checks for cross-process store invariants."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel
from redis.asyncio import ConnectionPool, Redis
from redis.asyncio.client import Pipeline

from hayhooks.durable import DurableContext, create_durable_router
from hayhooks.durable.context import _ClaimedExecution
from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionControl,
    ExecutionLeaseLostError,
    ExecutionStatus,
    Fail,
    Heartbeat,
    InvalidExecutionTransitionError,
    PayloadKind,
    RecoverExpiredLease,
    ReleaseClaim,
    RequestCancellation,
    Resume,
    ScheduleRetry,
    Suspend,
)
from hayhooks.durable.redis import RedisExecutionStore, RedisKeys
from hayhooks.durable.runtime import DurableDeployment, RuntimeConfig
from hayhooks.durable.store import (
    CHUNK_CURSOR_START,
    PUBLIC_PAYLOAD_KINDS,
    ChunkCursorExpiredError,
    ExecutionAdmissionError,
    ExecutionIdempotencyConflictError,
    ExecutionProgressCorruptionError,
    ExecutionStoreCorruptionError,
    ExecutionStoreError,
)
from tests.durable_store_contract import (
    ATTEMPTS_ERROR,
    CONTRACT_CONFIG,
    assert_discard_progress_contract,
    assert_lost_lease_budget_contract,
    assert_lowered_limits_keep_data_readable,
    assert_maintenance_backlog_contract,
    assert_raced_recovery_contract,
    assert_revision_routing_contract,
    assert_store_contract,
    assert_terminal_markers_contract,
    contract_control,
)

pytestmark = pytest.mark.integration


class SSERequest(BaseModel):
    chunks: int


def store_prefix(store: RedisExecutionStore) -> str:
    return store.keys.base.rsplit(":{", maxsplit=1)[0]


@pytest.fixture
async def redis_store():
    redis_url = os.getenv("HAYHOOKS_TEST_REDIS_URL")
    if not redis_url:
        pytest.skip("set HAYHOOKS_TEST_REDIS_URL to run the real-Redis suite")
    redis = Redis.from_url(redis_url, decode_responses=False)
    prefix = f"hayhooks:test:{uuid.uuid4().hex}"
    store = RedisExecutionStore(
        redis,
        "jobs",
        config=replace(CONTRACT_CONFIG, terminal_ttl_seconds=1),
        key_prefix=prefix,
    )
    await store.initialize()
    try:
        yield redis, store
    finally:
        keys = [key async for key in redis.scan_iter(match=f"{prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


async def test_redis_store_matches_shared_contract(redis_store) -> None:
    _, store = redis_store
    await assert_store_contract(store)


async def test_black_holed_connection_loses_the_lease_within_its_window(redis_store) -> None:  # noqa: PLR0915
    redis, store = redis_store
    connection = redis.connection_pool.connection_kwargs
    black_holed = asyncio.Event()
    proxy_tasks: set[asyncio.Task] = set()

    async def proxy_connection(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        server_reader, server_writer = await asyncio.open_connection(connection["host"], connection["port"])

        async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                while data := await reader.read(64 * 1_024):
                    if black_holed.is_set():
                        continue
                    writer.write(data)
                    await writer.drain()
            finally:
                writer.close()

        tasks = {
            asyncio.create_task(forward(client_reader, server_writer)),
            asyncio.create_task(forward(server_reader, client_writer)),
        }
        proxy_tasks.update(tasks)
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            proxy_tasks.difference_update(tasks)

    proxy = await asyncio.start_server(proxy_connection, "127.0.0.1", 0)
    proxy_port = proxy.sockets[0].getsockname()[1]
    proxied_redis = Redis(
        host="127.0.0.1",
        port=proxy_port,
        db=connection["db"],
        decode_responses=False,
        socket_timeout=None,
        retry=None,
    )
    proxied_store = RedisExecutionStore(
        proxied_redis,
        "jobs",
        config=store.config,
        key_prefix=store_prefix(store),
    )
    claim = fresh_claim = None
    try:
        await store.submit(contract_control("jobs", "blackhole"), b"input")
        confirmed_at = time.monotonic()
        claimed = await proxied_store.claim(Claim("worker", 0, 600, 3, "v1", ATTEMPTS_ERROR))
        assert claimed is not None
        claim = _ClaimedExecution(
            proxied_store,
            claimed.next_control,
            "worker",
            600,
            confirmed_at=confirmed_at,
        )
        await claim.start()
        black_holed.set()

        await asyncio.wait_for(claim.lease_lost.wait(), 1)
        assert claim.owned is False
        await asyncio.wait_for(claim.stop(), 1)

        black_holed.clear()
        await asyncio.sleep(0.1)
        await asyncio.wait_for(store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR), 1)
        fresh_at = time.monotonic()
        reclaimed = await asyncio.wait_for(
            proxied_store.claim(Claim("worker-2", 0, 600, 3, "v1", ATTEMPTS_ERROR)),
            1,
        )
        assert reclaimed is not None and reclaimed.next_control.fence > claim.control.fence
        fresh_claim = _ClaimedExecution(
            proxied_store,
            reclaimed.next_control,
            "worker-2",
            600,
            confirmed_at=fresh_at,
        )
        await asyncio.wait_for(fresh_claim.start(), 1)
        assert fresh_claim.owned
    finally:
        if fresh_claim is not None:
            await asyncio.wait_for(fresh_claim.stop(), 1)
        if claim is not None:
            await asyncio.wait_for(claim.stop(), 1)
        await asyncio.wait_for(proxied_redis.aclose(), 1)
        proxy.close()
        for task in tuple(proxy_tasks):
            task.cancel()
        await asyncio.wait_for(asyncio.gather(*proxy_tasks, return_exceptions=True), 1)
        await asyncio.wait_for(proxy.wait_closed(), 1)


async def test_redis_store_routes_claims_by_revision(redis_store) -> None:
    _, store = redis_store
    await assert_revision_routing_contract(store)


@pytest.mark.parametrize("max_stream_chunks", [CONTRACT_CONFIG.max_stream_chunks, 0], ids=["chunks", "no-chunks"])
async def test_redis_store_marks_every_terminal_path(redis_store, max_stream_chunks: int) -> None:
    redis, store = redis_store
    await assert_terminal_markers_contract(
        RedisExecutionStore(
            redis,
            "jobs",
            config=replace(store.config, max_stream_chunks=max_stream_chunks),
            key_prefix=store_prefix(store),
        )
    )


async def test_redis_store_skips_raced_lease_recovery(redis_store) -> None:
    _, store = redis_store
    await assert_raced_recovery_contract(store)


async def test_redis_store_fails_on_the_last_lost_lease(redis_store) -> None:
    _, store = redis_store
    await assert_lost_lease_budget_contract(store)


async def test_redis_store_keeps_data_readable_after_lowering_limits(redis_store) -> None:
    _, store = redis_store
    await assert_lowered_limits_keep_data_readable(store)


async def test_redis_store_discards_progress_only_when_requested(redis_store) -> None:
    _, store = redis_store
    await assert_discard_progress_contract(store)


async def test_redis_store_recovers_a_backlog_larger_than_one_batch(redis_store) -> None:
    _, store = redis_store
    await assert_maintenance_backlog_contract(store)


@pytest.mark.parametrize("corruption", ["wrong-type", "malformed"])
async def test_invalid_data_failure_repairs_public_progress(redis_store, corruption: str) -> None:
    redis, store = redis_store
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    await store.transition("run_1", Checkpoint(1, "worker", 0, 10_000, b"checkpoint", (b"progress",)))
    progress_key = store.keys.progress("run_1")
    if corruption == "wrong-type":
        await redis.delete(progress_key)
        await redis.set(progress_key, b"wrong type")
    else:
        await redis.rpush(progress_key, b"bad")

    with pytest.raises(ExecutionProgressCorruptionError):
        await store.read_public("run_1")
    error = b'{"type":"stored_execution_invalid"}'
    await store.transition("run_1", Fail(1, "worker", 0, error, discard_progress=True))

    public = await store.read_public("run_1")
    assert public is not None and public.control.status is ExecutionStatus.FAILED
    assert public.control.progress_sequence == 0 and public.progress == ()
    assert public.payloads[PayloadKind.ERROR] == error
    assert await store.operational_counts(revision="v1") == {
        "nonterminal": 0,
        "revision_nonterminal": 0,
        "revision_runnable": 0,
        "lease_expiry": 0,
    }


async def test_stale_invalid_data_failure_does_not_clear_progress(redis_store) -> None:
    redis, store = redis_store
    control = await claim_one(store, lease_ms=10_000)
    await redis.set(store.keys.progress("run_1"), b"wrong type")
    await store.transition("run_1", ReleaseClaim(control.fence, "worker"))
    assert await store.claim(Claim("other", 0, 10_000, 3, "v1", ATTEMPTS_ERROR)) is not None

    with pytest.raises(ExecutionLeaseLostError):
        await store.transition("run_1", Fail(control.fence, "worker", 0, b"invalid", discard_progress=True))

    assert await redis.type(store.keys.progress("run_1")) == b"string"


async def test_cancellation_wins_invalid_data_failure_and_clears_progress(redis_store) -> None:
    redis, store = redis_store
    control = await claim_one(store, lease_ms=10_000)
    await store.transition("run_1", RequestCancellation(0, "cancel"))
    await redis.set(store.keys.progress("run_1"), b"wrong type")

    canceled = await store.transition(
        "run_1",
        Fail(control.fence, "worker", 0, b"invalid", discard_progress=True),
    )

    assert canceled.next_control.status is ExecutionStatus.CANCELED
    public = await store.read_public("run_1")
    assert public is not None and public.control.status is ExecutionStatus.CANCELED
    assert public.control.progress_sequence == 0 and public.progress == ()


async def test_chunk_reads_skip_undecodable_entries(redis_store) -> None:
    redis, store = redis_store
    reader = RedisExecutionStore(
        redis,
        "jobs",
        config=replace(store.config, max_stream_chunks=10),
        key_prefix=store_prefix(store),
    )
    key = store.keys.chunks("run_1")
    first = await redis.xadd(key, {"attempt": 1, "data": b"first"})
    skipped = await redis.xadd(key, {"bogus": b"entry"})
    oversized = await redis.xadd(key, {"attempt": 1, "data": b"x" * 100})
    last = await redis.xadd(key, {"attempt": 1, "data": b"last"})

    chunks = await reader.read_chunks("run_1", CHUNK_CURSOR_START)
    waited = await reader.wait_chunks("run_1", CHUNK_CURSOR_START, 0.05)

    assert [chunk.cursor for chunk in chunks] == [value.decode() for value in (first, skipped, oversized, last)]
    assert chunks == waited
    assert chunks[1].skipped and chunks[1].data == b""
    assert chunks[2].data == b"x" * 100
    assert await reader.read_chunks("run_1", skipped.decode()) == chunks[2:]


async def test_corrupt_chunk_stream_does_not_block_an_owned_failure(redis_store) -> None:
    redis, store = redis_store
    control = await claim_one(store, lease_ms=10_000)
    await redis.set(store.keys.chunks("run_1"), b"wrong type")

    failed = await store.transition("run_1", Fail(control.fence, "worker", 0, b"failed"))

    assert failed.next_control.status is ExecutionStatus.FAILED
    assert await redis.type(store.keys.chunks("run_1")) == b"stream"
    chunks = await store.read_chunks("run_1", CHUNK_CURSOR_START)
    assert [(chunk.terminal, chunk.attempt) for chunk in chunks] == [(True, 1)]
    assert (await store.operational_counts(revision="v1"))["nonterminal"] == 0


async def test_stale_failure_does_not_repair_a_corrupt_chunk_stream(redis_store) -> None:
    redis, store = redis_store
    control = await claim_one(store, lease_ms=10_000)
    await store.transition("run_1", ReleaseClaim(control.fence, "worker"))
    assert await store.claim(Claim("other", 0, 10_000, 3, "v1", ATTEMPTS_ERROR)) is not None
    await redis.set(store.keys.chunks("run_1"), b"wrong type")
    before = await dump_keys(redis, store)

    with pytest.raises(ExecutionLeaseLostError):
        await store.transition("run_1", Fail(control.fence, "worker", 0, b"stale"))

    assert await dump_keys(redis, store) == before


async def test_corrupt_chunk_stream_does_not_block_terminal_recovery(redis_store) -> None:
    redis, store = redis_store
    for run_id in ("run_bad", "run_a", "run_b"):
        await store.submit(contract_control("jobs", run_id, idempotency=run_id, binding=run_id), b"input")
        assert await store.claim(Claim("worker", 0, 50, 3, "v1", ATTEMPTS_ERROR)) is not None
    await redis.hset(store.keys.control("run_bad"), "lease_recoveries", 2)
    await redis.set(store.keys.chunks("run_bad"), b"wrong type")
    await asyncio.sleep(0.06)

    assert await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR) == 2
    bad = await store.read("run_bad")
    assert bad is not None and bad.control.status is ExecutionStatus.FAILED
    assert await redis.type(store.keys.chunks("run_bad")) == b"stream"
    assert await store.operational_counts(revision="v1") == {
        "nonterminal": 2,
        "revision_nonterminal": 2,
        "revision_runnable": 2,
        "lease_expiry": 0,
    }
    assert await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR) == 0

    race = replace(
        contract_control("jobs", "run_race", idempotency="run_race", binding="run_race"),
        definition_revision="v2",
    )
    await store.submit(race, b"input")
    assert await store.claim(Claim("worker", 0, 50, 3, "v2", ATTEMPTS_ERROR)) is not None
    await redis.hset(store.keys.control("run_race"), "lease_recoveries", 2)
    await redis.set(store.keys.chunks("run_race"), b"wrong type")
    contender = RedisExecutionStore(redis, "jobs", config=store.config, key_prefix=store_prefix(store))
    await asyncio.sleep(0.06)

    recovered = await asyncio.gather(
        store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR),
        contender.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR),
    )

    assert sum(recovered) == 0
    assert (await store.operational_counts(revision="v1"))["nonterminal"] == 2
    assert await redis.zcard(store.keys.lease_expiry) == 0
    assert await redis.type(store.keys.chunks("run_race")) == b"stream"


async def test_concurrent_submissions_and_claims_have_one_winner(redis_store) -> None:
    redis, store = redis_store
    submissions = await asyncio.gather(
        *(store.submit(contract_control("jobs", f"run_{index}"), b"input") for index in range(20))
    )
    assert sum(result.created for result in submissions) == 1
    assert {result.control.run_id for result in submissions} == {submissions[0].control.run_id}

    claims = await asyncio.gather(
        *(store.claim(Claim(f"worker-{index}", 0, 10_000, 3, "v1", ATTEMPTS_ERROR)) for index in range(20))
    )
    winners = [plan for plan in claims if plan is not None and plan.next_control.status is ExecutionStatus.RUNNING]
    assert len(winners) == 1
    assert await redis.zcard(store.keys.runnable_revision("v1")) == 0
    assert await redis.zcard(store.keys.lease_expiry) == 1


async def test_concurrent_submissions_are_admitted_without_contention(redis_store) -> None:
    _, store = redis_store
    controls = [
        contract_control("jobs", f"run_{index}", idempotency=f"idem_{index}", binding=f"binding_{index}")
        for index in range(100)
    ]

    results = await asyncio.gather(*(store.submit(control, b"input") for control in controls))

    assert all(result.created for result in results)
    assert await store.operational_counts(revision="v1") == {
        "nonterminal": 100,
        "revision_nonterminal": 100,
        "revision_runnable": 100,
        "lease_expiry": 0,
    }


@pytest.mark.parametrize("limit", [5, 0])
async def test_admission_limit_is_exact_under_concurrent_submissions(redis_store, limit: int) -> None:
    redis, store = redis_store
    limited = RedisExecutionStore(
        redis,
        "limited",
        config=replace(store.config, max_nonterminal_executions=limit),
        key_prefix=f"{store_prefix(store)}:limited-{limit}",
    )
    controls = [
        contract_control("limited", f"run_{index}", idempotency=f"idem_{index}", binding=f"binding_{index}")
        for index in range(30)
    ]

    results = await asyncio.gather(*(limited.submit(control, b"input") for control in controls), return_exceptions=True)

    expected = 30 if limit == 0 else limit
    assert sum(not isinstance(result, BaseException) for result in results) == expected
    assert sum(isinstance(result, ExecutionAdmissionError) for result in results) == 30 - expected
    assert (await limited.operational_counts(revision="v1"))["nonterminal"] == expected


@pytest.mark.parametrize("target", ["binding", "revision-index", "capacity", "control"])
async def test_submission_with_a_wrong_type_key_writes_nothing(redis_store, target: str) -> None:
    redis, store = redis_store
    control = contract_control("jobs")
    key = {
        "binding": store.keys.idempotency(control.idempotency_digest),
        "revision-index": store.keys.runnable_revision("v1"),
        "capacity": store.keys.capacity,
        "control": store.keys.control(control.run_id),
    }[target]
    await redis.set(key, b"wrong type")
    before = await dump_keys(redis, store)

    error = ExecutionIdempotencyConflictError if target == "control" else ExecutionStoreError
    with pytest.raises(error):
        await store.submit(control, b"input")

    assert await dump_keys(redis, store) == before


@pytest.mark.parametrize("field", ["nonterminal", "revision"])
async def test_submission_rejects_an_invalid_capacity_counter(redis_store, field: str) -> None:
    redis, store = redis_store
    field = RedisKeys.revision_nonterminal_field("v1") if field == "revision" else field
    await redis.hset(store.keys.capacity, field, "01")
    before = await dump_keys(redis, store)

    with pytest.raises(ExecutionStoreCorruptionError, match="counter"):
        await store.submit(contract_control("jobs"), b"input")

    assert await dump_keys(redis, store) == before


async def test_replay_that_races_terminal_expiry_creates_a_new_execution(redis_store, monkeypatch) -> None:
    redis, store = redis_store
    control = contract_control("jobs", "old")
    await store.submit(control, b"input")
    await store.transition("old", RequestCancellation(0, "done"))
    original = redis.hgetall
    called = False

    async def expire_once(key):
        nonlocal called
        if not called:
            called = True
            await redis.delete(store.keys.idempotency(control.idempotency_digest), store.keys.control("old"))
            return {}
        return await original(key)

    monkeypatch.setattr(redis, "hgetall", expire_once)
    replay = await store.submit(contract_control("jobs", "new"), b"input")
    assert replay.created and replay.control.run_id == "new"


async def test_replay_of_a_binding_without_its_execution_reports_corruption(redis_store) -> None:
    redis, store = redis_store
    broken = RedisExecutionStore(
        redis,
        "broken",
        config=store.config,
        key_prefix=f"{store_prefix(store)}:broken",
        transaction_backoff_ms=0,
    )
    control = contract_control("broken")
    await redis.hset(
        broken.keys.idempotency(control.idempotency_digest),
        mapping={"run_id": control.run_id, "binding": control.idempotency_binding_digest},
    )

    with pytest.raises(ExecutionStoreCorruptionError, match="missing execution"):
        await broken.submit(control, b"input")


async def test_claim_that_loses_a_race_takes_the_next_runnable_execution(redis_store) -> None:
    redis, store = redis_store
    for index in range(2):
        await store.submit(
            contract_control("jobs", f"run_{index}", idempotency=str(index), binding=str(index)), b"input"
        )
    first = await store.claim(Claim("worker-0", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))
    assert first is not None
    # A worker that read this candidate before worker-0 committed can still see it in the index.
    await redis.zadd(store.keys.runnable_revision("v1"), {first.next_control.run_id: 0})

    second = await store.claim(Claim("worker-1", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))

    assert second is not None and second.next_control.run_id != first.next_control.run_id


@pytest.mark.parametrize(
    ("member", "score"),
    [
        pytest.param("bad_inf", float("inf"), id="infinite"),
        pytest.param("bad_negative", -1, id="negative"),
        pytest.param("bad_fraction", 1.5, id="fraction"),
        pytest.param("bad:id", 0, id="invalid-id"),
    ],
)
async def test_claim_drops_invalid_runnable_entries(redis_store, member: str, score: float) -> None:
    redis, store = redis_store
    await store.submit(contract_control("jobs"), b"input")
    await redis.zadd(store.keys.runnable_revision("v1"), {member: score})

    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))

    assert claimed is not None and claimed.next_control.run_id == "run_1"
    assert await redis.zscore(store.keys.runnable_revision("v1"), member) is None


async def test_claim_ignores_executions_that_are_not_due_yet(redis_store) -> None:
    redis, store = redis_store
    submitted = await store.submit(contract_control("jobs"), b"input")
    future = submitted.control.updated_at_ms + 3_600_000
    await redis.zadd(store.keys.runnable_revision("v1"), {"run_1": future})

    assert await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR)) is None
    assert await redis.zscore(store.keys.runnable_revision("v1"), "run_1") == future
    assert await store.read_control("run_1") == submitted.control


async def test_noop_transitions_write_nothing(redis_store, monkeypatch) -> None:
    redis, store = redis_store
    await store.submit(contract_control("jobs"), b"input")
    await store.transition("run_1", RequestCancellation(0, "done"))
    before = await dump_keys(redis, store)
    apply = AsyncMock(wraps=store._apply_script)
    monkeypatch.setattr(store, "_apply_script", apply)

    plan = await store.transition("run_1", RequestCancellation(0, "again"))

    assert plan.next_control.status is ExecutionStatus.CANCELED
    apply.assert_not_awaited()
    assert await dump_keys(redis, store) == before


async def test_terminal_chunk_streams_expire_before_the_execution(redis_store) -> None:
    redis, fixture_store = redis_store
    store = RedisExecutionStore(
        redis,
        "jobs",
        config=replace(fixture_store.config, terminal_ttl_seconds=60, stream_ttl_seconds=1),
        key_prefix=store_prefix(fixture_store),
    )
    control = await claim_one(store, lease_ms=10_000)
    await store.append_chunks("run_1", 1, control.fence, "worker", [b"chunk"])
    await store.transition("run_1", Complete(control.fence, "worker", 0, b"done"))
    assert 0 < await redis.pttl(store.keys.chunks("run_1")) <= 1_000
    assert await redis.pttl(store.keys.control("run_1")) > 1_000

    short = RedisExecutionStore(
        redis,
        "short",
        config=replace(fixture_store.config, terminal_ttl_seconds=1),
        key_prefix=f"{store_prefix(fixture_store)}:short",
    )
    await short.submit(contract_control("short"), b"input")
    claimed = await short.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    await short.transition("run_1", Complete(1, "worker", 0, b"done"))
    assert 0 < await redis.pttl(short.keys.chunks("run_1")) <= 1_000


async def test_stepped_back_redis_time_keeps_controls_decodable(redis_store, monkeypatch) -> None:
    _, store = redis_store
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    updated_at_ms = claimed.next_control.updated_at_ms

    from hayhooks.durable import redis as redis_module

    real_milliseconds = redis_module._milliseconds
    monkeypatch.setattr(redis_module, "_milliseconds", lambda value: real_milliseconds(value) - 5_000)
    retried = await store.transition("run_1", ScheduleRetry(1, "worker", 0, 0, 3, b"retry"))
    assert retried.next_control.updated_at_ms == updated_at_ms
    assert (await store.read("run_1")).control.updated_at_ms == updated_at_ms

    monkeypatch.setattr(redis_module, "_milliseconds", real_milliseconds)
    assert await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR)) is not None


async def test_heartbeat_after_a_clock_step_renews_past_the_last_update(redis_store) -> None:
    redis, store = redis_store
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    updated_at_ms = claimed.next_control.lease_expires_at_ms - 1
    await redis.hset(store.keys.control("run_1"), "updated_at_ms", updated_at_ms)

    heartbeat = await store.transition("run_1", Heartbeat(1, "worker", 0, 50))

    assert heartbeat.next_control.updated_at_ms == updated_at_ms
    assert heartbeat.next_control.lease_expires_at_ms == updated_at_ms + 50


async def test_claim_drops_an_undecodable_head_and_claims_the_next(redis_store, caplog) -> None:
    redis, store = redis_store
    for run_id in ("run_a", "run_b"):
        await store.submit(contract_control("jobs", run_id, idempotency=run_id, binding=run_id), b"input")
    await redis.hset(store.keys.control("run_a"), "status", "bogus")

    claims = [await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR)) for _ in range(2)]

    assert [plan.next_control.run_id for plan in claims if plan is not None] == ["run_b"]
    assert await redis.zscore(store.keys.runnable_revision("v1"), "run_a") is None
    assert "Removed an undecodable durable execution from the runnable index" in caplog.messages
    with pytest.raises(ExecutionStoreCorruptionError):
        await store.read("run_a")


async def test_maintenance_drops_an_undecodable_lease_member_and_recovers_the_rest(redis_store, caplog) -> None:
    redis, store = redis_store
    for run_id in ("run_a", "run_b", "run_c", "run_d"):
        await store.submit(contract_control("jobs", run_id, idempotency=run_id, binding=run_id), b"input")
        assert await store.claim(Claim("worker", 0, 50, 3, "v1", ATTEMPTS_ERROR)) is not None
    created = await redis.hget(store.keys.control("run_a"), "created_at_ms")
    await redis.hset(store.keys.control("run_a"), "updated_at_ms", int(created) - 1)
    await asyncio.sleep(0.1)

    assert await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR) == 3
    assert await redis.zscore(store.keys.lease_expiry, RedisKeys.lease_member("run_a", 1)) is None
    assert "Removed an undecodable durable execution from the lease index" in caplog.messages
    assert await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR) == 0


async def test_maintenance_removes_invalid_lease_entries(redis_store) -> None:
    redis, store = redis_store
    members = {"run_1|1": float("inf"), "run_1": 0}
    await redis.zadd(store.keys.lease_expiry, members)

    assert await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR) == 0
    assert await redis.zcard(store.keys.lease_expiry) == 0


async def test_maintenance_ignores_leases_that_are_not_due(redis_store) -> None:
    redis, store = redis_store
    control = await claim_one(store, lease_ms=10_000)

    assert await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR) == 0
    assert (
        await redis.zscore(
            store.keys.lease_expiry,
            RedisKeys.lease_member("run_1", control.fence),
        )
        == control.lease_expires_at_ms
    )


async def test_concurrent_maintainers_recover_each_lease_once(redis_store, monkeypatch) -> None:
    redis, store = redis_store
    contender = RedisExecutionStore(redis, "jobs", config=store.config, key_prefix=store_prefix(store))
    for index in range(100):
        run_id = f"run_{index}"
        await store.submit(contract_control("jobs", run_id, idempotency=run_id, binding=run_id), b"input")
        assert await store.claim(Claim("worker", 0, 50, 3, "v1", ATTEMPTS_ERROR)) is not None
    await asyncio.sleep(0.06)
    calls = [0]

    def count_commits(target: RedisExecutionStore) -> None:
        original = target._commit

        async def counted(*args, **kwargs):
            calls[0] += 1
            return await original(*args, **kwargs)

        monkeypatch.setattr(target, "_commit", counted)

    count_commits(store)
    count_commits(contender)
    requeued = await asyncio.gather(
        store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR),
        contender.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR),
    )

    assert sum(requeued) == 100
    assert calls[0] < 150
    assert await redis.zcard(store.keys.lease_expiry) == 0


async def test_read_keeps_operational_command_errors_visible(redis_store, monkeypatch) -> None:
    _, store = redis_store
    await store.submit(contract_control("jobs"), b"input")
    original = store.redis.pipeline

    class BrokenPipeline:
        async def __aenter__(self):
            self.pipe = original(transaction=True)
            await self.pipe.__aenter__()
            return self

        async def __aexit__(self, *args):
            return await self.pipe.__aexit__(*args)

        def __getattr__(self, name):
            return getattr(self.pipe, name)

        async def execute(self, **_kwargs):
            replies = list(await self.pipe.execute())
            replies[-1] = ResponseError("NOPERM denied")
            return replies

    from redis.exceptions import ResponseError

    monkeypatch.setattr(store.redis, "pipeline", lambda **_kwargs: BrokenPipeline())
    with pytest.raises(ExecutionStoreError, match="ResponseError NOPERM"):
        await store.read("run_1")


async def test_concurrent_progress_and_cancellation_remain_atomic(redis_store) -> None:
    redis, store = redis_store
    control = contract_control("jobs")
    await store.submit(control, b"input")
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    fence = claimed.next_control.fence
    await asyncio.gather(
        *(
            store.transition(
                control.run_id,
                Checkpoint(fence, "worker", 0, 10_000, f"checkpoint-{index}".encode(), (str(index).encode(),)),
            )
            for index in range(10)
        )
    )
    await asyncio.gather(
        store.transition(control.run_id, RequestCancellation(0, "stop")),
        store.transition(control.run_id, Checkpoint(fence, "worker", 0, 10_000, b"final", (b"final",))),
    )
    snapshot = await store.read(control.run_id)
    assert snapshot is not None
    assert snapshot.control.cancel_requested_at_ms is not None
    assert [event.sequence for event in snapshot.progress] == list(range(10, 12))
    assert {event.data for event in snapshot.progress} <= {str(index).encode() for index in range(10)} | {b"final"}

    terminal = await store.transition(control.run_id, Complete(fence, "worker", 0, b"ignored"))
    assert terminal.next_control.status is ExecutionStatus.CANCELED
    assert (await store.operational_counts(revision="v1"))["nonterminal"] == 0
    with pytest.raises(ExecutionLeaseLostError):
        await store.append_chunks(control.run_id, 1, fence, "worker", [b"late"])
    assert [chunk.terminal for chunk in await store.read_chunks(control.run_id, CHUNK_CURSOR_START)] == [True]
    assert await redis.pttl(store.keys.idempotency(control.idempotency_digest)) > 0
    await asyncio.sleep(1.1)
    assert await store.read(control.run_id) is None
    assert not await redis.exists(store.keys.chunks(control.run_id))
    assert not await redis.exists(store.keys.idempotency(control.idempotency_digest))


async def test_concurrent_resume_commits_one_checkpoint(redis_store) -> None:
    _, store = redis_store
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    waiting = await store.transition(
        "run_1",
        Suspend(claimed.next_control.fence, "worker", 0, b"initial", b"wait"),
    )
    resumes = await asyncio.gather(
        *(
            store.transition("run_1", Resume(0, "v1", value, expected_version=waiting.next_control.version))
            for value in (b"first", b"second")
        ),
        return_exceptions=True,
    )
    winner = next(result for result in resumes if not isinstance(result, BaseException))
    assert sum(isinstance(result, InvalidExecutionTransitionError) for result in resumes) == 1
    snapshot = await store.read("run_1")
    assert snapshot is not None
    assert snapshot.payloads[PayloadKind.CHECKPOINT] == winner.payload_writes[0].data


@pytest.mark.parametrize("operation", ["read", "transition", "replay"])
async def test_control_key_identity_corruption_is_rejected(redis_store, operation: str) -> None:
    redis, store = redis_store
    control = contract_control("jobs")
    await store.submit(control, b"input")
    await redis.hset(store.keys.control(control.run_id), "run_id", "run_2")

    with pytest.raises(ExecutionStoreCorruptionError):
        if operation == "read":
            await store.read(control.run_id)
        elif operation == "transition":
            await store.transition(control.run_id, RequestCancellation(0, "stop"))
        else:
            await store.submit(control, b"input")

    assert not await redis.exists(store.keys.control("run_2"))
    assert await store.operational_counts(revision="v1") == {
        "nonterminal": 1,
        "revision_nonterminal": 1,
        "revision_runnable": 1,
        "lease_expiry": 0,
    }


async def test_admission_heartbeat_and_stale_lease_repair_are_transactional(redis_store, monkeypatch) -> None:
    redis, store = redis_store
    limited = RedisExecutionStore(
        redis,
        "limited",
        config=replace(CONTRACT_CONFIG, max_nonterminal_executions=1),
        key_prefix=f"{store_prefix(store)}:limited",
    )
    controls = (
        contract_control("limited", "run_1", idempotency="one", binding="one"),
        contract_control("limited", "run_2", idempotency="two", binding="two"),
    )
    results = await asyncio.gather(*(limited.submit(control, b"input") for control in controls), return_exceptions=True)
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, ExecutionAdmissionError) for result in results) == 1

    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    with monkeypatch.context() as patch:
        patch.setattr(store, "_plan_commands", lambda *_args, **_kwargs: pytest.fail("heartbeat rewrote control"))
        heartbeat = await store.transition("run_1", Heartbeat(1, "worker", 0, 10_000))
    assert heartbeat.next_control.version == claimed.next_control.version
    live_member = RedisKeys.lease_member("run_1", claimed.next_control.fence)
    await redis.zadd(store.keys.lease_expiry, {RedisKeys.lease_member("run_1", 0): 0})
    await store.maintain(
        max_run_attempts=3,
        attempts_error=ATTEMPTS_ERROR,
    )
    assert await redis.zrange(store.keys.lease_expiry, 0, -1) == [live_member.encode()]


LEASE_MS = 300


async def dump_keys(redis: Redis, store: RedisExecutionStore) -> dict[bytes, tuple[bytes, bool]]:
    """Capture every key's serialized value and whether it expires."""
    keys = [key async for key in redis.scan_iter(match=f"{store_prefix(store)}:*")]
    return {key: (await redis.dump(key), await redis.pttl(key) > 0) for key in sorted(keys)}


async def claim_one(store: RedisExecutionStore, lease_ms: int = LEASE_MS) -> ExecutionControl:
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, lease_ms, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    return claimed.next_control


def interfere_on_first_call(monkeypatch, store: RedisExecutionStore, script: str, interfere) -> list[tuple]:
    """Run ``interfere`` between a commit's preparation and its script, recording what the script changed."""
    original = getattr(store, script)
    calls: list[tuple] = []

    async def interfering(*args, **kwargs):
        if calls:
            return await original(*args, **kwargs)
        calls.append(())
        await interfere()
        before = await dump_keys(store.redis, store)
        outcome = await original(*args, **kwargs)
        calls[0] = (outcome, before, await dump_keys(store.redis, store))
        return outcome

    monkeypatch.setattr(store, script, interfering)
    return calls


OWNED_COMMITS = {
    "checkpoint": (
        "_apply_script",
        lambda store, fence: store.transition(
            "run_1", Checkpoint(fence, "worker", 0, LEASE_MS, b"checkpoint", (b"progress",))
        ),
    ),
    "complete": (
        "_apply_script",
        lambda store, fence: store.transition("run_1", Complete(fence, "worker", 0, b"done")),
    ),
    "heartbeat": (
        "_heartbeat_script",
        lambda store, fence: store.transition("run_1", Heartbeat(fence, "worker", 0, LEASE_MS)),
    ),
    "chunk": (
        "_append_chunks_script",
        lambda store, fence: store.append_chunks("run_1", 1, fence, "worker", [b"chunk"]),
    ),
}


@pytest.mark.parametrize("operation", OWNED_COMMITS)
@pytest.mark.parametrize("interference", ["expired", "stale-fence"])
async def test_delayed_owned_commit_writes_nothing(redis_store, monkeypatch, operation: str, interference: str) -> None:
    redis, store = redis_store
    control = await claim_one(store)
    assert control.lease_expires_at_ms is not None

    async def interfere() -> None:
        if interference == "expired":
            seconds, microseconds = await redis.time()
            now_ms = seconds * 1_000 + microseconds // 1_000
            safe_until_ms = control.lease_expires_at_ms - store.config.lease_commit_safety_ms
            await asyncio.sleep((safe_until_ms - now_ms) / 1_000 + 0.005)
        else:
            await store.transition("run_1", ReleaseClaim(control.fence, "worker"))
            assert await store.claim(Claim("other", 0, LEASE_MS, 3, "v1", ATTEMPTS_ERROR)) is not None

    script, commit = OWNED_COMMITS[operation]
    calls = interfere_on_first_call(monkeypatch, store, script, interfere)
    with pytest.raises(ExecutionLeaseLostError):
        await commit(store, control.fence)

    _, before, after = calls[0]
    assert after == before


@pytest.mark.parametrize("operation", ["checkpoint", "complete"])
async def test_changed_control_snapshot_retries_from_a_fresh_read(redis_store, monkeypatch, operation: str) -> None:
    _, store = redis_store
    control = await claim_one(store)
    script, commit = OWNED_COMMITS[operation]
    calls = interfere_on_first_call(
        monkeypatch,
        store,
        script,
        lambda: store.transition("run_1", RequestCancellation(0, "stop")),
    )

    plan = await commit(store, control.fence)

    outcome, before, after = calls[0]
    assert outcome == 0 and after == before
    assert plan.next_control.cancel_requested_at_ms is not None
    assert plan.next_control.status is (
        ExecutionStatus.CANCELED if operation == "complete" else ExecutionStatus.RUNNING
    )


@pytest.mark.parametrize(
    "key",
    [
        "progress",
        "revision",
        "lease_expiry",
        "capacity-fraction",
        "capacity-leading-zero",
        "capacity-revision",
    ],
)
async def test_corrupt_commit_targets_leave_every_key_unchanged(redis_store, key: str) -> None:
    redis, store = redis_store
    control = await claim_one(store, lease_ms=10_000)
    if key.startswith("capacity-"):
        field = RedisKeys.revision_nonterminal_field("v1") if key == "capacity-revision" else "nonterminal"
        await redis.hset(store.keys.capacity, field, "1.5" if key == "capacity-fraction" else "01")
    else:
        target = {
            "progress": store.keys.progress("run_1"),
            "revision": store.keys.runnable_revision("v1"),
            "lease_expiry": store.keys.lease_expiry,
        }[key]
        await redis.set(target, b"wrong type")
    before = await dump_keys(redis, store)

    with pytest.raises(ExecutionStoreError):
        await store.transition("run_1", Complete(control.fence, "worker", 0, b"done", (b"progress",)))

    assert await dump_keys(redis, store) == before


async def test_corrupt_chunks_are_not_repaired_when_another_commit_target_is_invalid(redis_store) -> None:
    redis, store = redis_store
    control = await claim_one(store, lease_ms=10_000)
    await redis.set(store.keys.chunks("run_1"), b"wrong type")
    await redis.set(store.keys.runnable_revision("v1"), b"wrong type")
    before = await dump_keys(redis, store)

    with pytest.raises(ExecutionStoreError):
        await store.transition("run_1", Complete(control.fence, "worker", 0, b"done", (b"progress",)))

    assert await dump_keys(redis, store) == before


async def test_heartbeat_rejects_a_corrupt_lease_index_without_renewing(redis_store) -> None:
    redis, store = redis_store
    control = await claim_one(store, lease_ms=10_000)
    await redis.set(store.keys.lease_expiry, b"wrong type")
    before = await dump_keys(redis, store)

    with pytest.raises(ExecutionStoreError):
        await store.transition("run_1", Heartbeat(control.fence, "worker", 0, 20_000))

    assert await dump_keys(redis, store) == before


async def test_claim_ignores_unrelated_submissions_during_its_commit(redis_store, monkeypatch) -> None:
    _, store = redis_store
    await store.submit(contract_control("jobs"), b"input")
    calls = interfere_on_first_call(
        monkeypatch,
        store,
        "_apply_script",
        lambda: store.submit(contract_control("jobs", "run_2", idempotency="two", binding="two"), b"input"),
    )

    claimed = await store.claim(Claim("worker", 0, LEASE_MS, 3, "v1", ATTEMPTS_ERROR))

    assert claimed is not None and claimed.next_control.status is ExecutionStatus.RUNNING
    assert calls[0][0] == 1
    assert (await store.operational_counts(revision="v1"))["revision_runnable"] == 1


@pytest.fixture
def round_trips(monkeypatch) -> list[list[tuple[str, ...]]]:
    """Record client round trips as their (command, key) pairs."""
    trips: list[list[tuple[str, ...]]] = []
    execute_command = Redis.execute_command
    execute_pipeline = Pipeline.execute

    async def record_command(self, *args, **options):
        trips.append([_command_name(args)])
        return await execute_command(self, *args, **options)

    async def record_pipeline(self, *args, **options):
        trips.append([_command_name(command) for command, _ in self.command_stack])
        return await execute_pipeline(self, *args, **options)

    monkeypatch.setattr(Redis, "execute_command", record_command)
    monkeypatch.setattr(Pipeline, "execute", record_pipeline)
    return trips


def _command_name(args: tuple) -> tuple[str, ...]:
    name = args[0].decode() if isinstance(args[0], bytes) else str(args[0])
    key = args[1] if name in {"GET", "HGETALL", "LRANGE", "XRANGE"} else None
    return (name,) if key is None else (name, key.decode() if isinstance(key, bytes) else str(key))


async def test_scheduling_paths_cost_one_round_trip_each(redis_store, round_trips) -> None:
    _, store = redis_store
    warm = replace(
        contract_control("jobs", "warm", idempotency="warm", binding="warm"),
        definition_revision="warm",
    )
    await store.submit(warm, b"input")
    assert await store.claim(Claim("warm", 0, 10_000, 3, "warm", ATTEMPTS_ERROR)) is not None

    control = contract_control("jobs")
    round_trips.clear()
    await store.submit(control, b"input")
    assert round_trips == [[("EVALSHA",)]]

    round_trips.clear()
    await store.submit(control, b"input")
    assert round_trips == [[("EVALSHA",)], [("HGETALL", store.keys.control("run_1"))]]

    round_trips.clear()
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    assert round_trips == [
        [("ZRANGE",), ("TIME",)],
        [("HGETALL", store.keys.control("run_1")), ("TIME",)],
        [("EVALSHA",)],
    ]

    round_trips.clear()
    assert await store.claim(Claim("worker", 0, 10_000, 3, "v1", ATTEMPTS_ERROR)) is None
    assert round_trips == [[("ZRANGE",), ("TIME",)]]

    round_trips.clear()
    assert await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR) == 0
    assert round_trips == [[("ZRANGE",), ("TIME",)]]

    await store.transition("run_1", Complete(1, "worker", 0, b"done"))
    round_trips.clear()
    await store.transition("run_1", RequestCancellation(0, "late"))
    assert round_trips == [[("HGETALL", store.keys.control("run_1")), ("TIME",)]]

    round_trips.clear()
    assert await store.read_control("run_1") is not None
    assert round_trips == [[("HGETALL", store.keys.control("run_1"))]]

    recovered = replace(
        contract_control("jobs", "recovered", idempotency="recovered", binding="recovered"),
        definition_revision="recovery",
    )
    await store.submit(recovered, b"input")
    claim = await store.claim(Claim("worker", 0, 50, 3, "recovery", ATTEMPTS_ERROR))
    assert claim is not None and claim.next_control.lease_expires_at_ms is not None
    await asyncio.sleep(0.06)
    await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR)
    command = RecoverExpiredLease(
        0,
        claim.next_control.fence,
        claim.next_control.lease_expires_at_ms,
        3,
        ATTEMPTS_ERROR,
    )
    round_trips.clear()
    await store.transition("recovered", command)
    assert round_trips == [
        [("HGETALL", store.keys.control("recovered")), ("TIME",), ("ZSCORE",)],
    ]


async def test_hot_paths_cost_one_round_trip(redis_store, round_trips) -> None:
    _, store = redis_store
    control = await claim_one(store, lease_ms=10_000)
    heartbeat = Heartbeat(control.fence, "worker", 0, 10_000)
    checkpoint = Checkpoint(control.fence, "worker", 0, 10_000, b"checkpoint")
    for warm in (
        store.transition("run_1", heartbeat),
        store.transition("run_1", checkpoint),
        store.append_chunks("run_1", 1, control.fence, "worker", [b"warm"]),
    ):
        await warm
    cursor = (await store.read_chunks("run_1", CHUNK_CURSOR_START))[-1].cursor

    round_trips.clear()
    await store.transition("run_1", heartbeat)
    await store.append_chunks("run_1", 1, control.fence, "worker", [b"one", b"two"])
    await store.transition("run_1", checkpoint)
    assert round_trips == [
        [("EVALSHA",)],
        [("EVALSHA",)],
        [("HGETALL", store.keys.control("run_1")), ("TIME",)],
        [("EVALSHA",)],
    ]

    chunks = await store.read_chunks("run_1", cursor)
    round_trips.clear()
    assert await store.wait_chunks("run_1", chunks[-1].cursor, 0.2) == ()
    waiting = asyncio.create_task(store.wait_chunks("run_1", chunks[-1].cursor, 5))
    await asyncio.sleep(0.1)
    await store.append_chunks("run_1", 1, control.fence, "worker", [b"three"])
    assert [chunk.data for chunk in await waiting] == [b"three"]
    wake_up = [("XREAD",), ("XRANGE", store.keys.chunks("run_1"))]
    assert round_trips == [wake_up, wake_up, [("EVALSHA",)]]

    round_trips.clear()
    public = await store.read_public("run_1")
    assert public is not None and not {PayloadKind.INPUT, PayloadKind.CHECKPOINT} & public.payloads.keys()
    assert round_trips == [
        [
            ("HGETALL", store.keys.control("run_1")),
            *(("GET", store.keys.payload("run_1", kind)) for kind in PUBLIC_PAYLOAD_KINDS),
            ("LRANGE", store.keys.progress("run_1")),
        ]
    ]


@pytest.mark.parametrize("read", ["read", "read_public"])
@pytest.mark.parametrize("control_present", [False, True], ids=["orphan", "present"])
async def test_wrong_type_keys_fail_closed_only_for_existing_executions(
    redis_store, read: str, control_present: bool
) -> None:
    redis, store = redis_store
    if control_present:
        await store.submit(contract_control("jobs"), b"input")
    await redis.hset(store.keys.payload("run_1", PayloadKind.RESULT), "wrong", "type")
    await redis.set(store.keys.progress("run_1"), b"wrong type")

    if control_present:
        with pytest.raises(ExecutionStoreCorruptionError, match="invalid types"):
            await getattr(store, read)("run_1")
    else:
        assert await getattr(store, read)("run_1") is None


async def test_blocking_reads_use_only_the_viewer_client(redis_store) -> None:
    redis, store = redis_store
    viewer = Redis(connection_pool=ConnectionPool.from_url(os.environ["HAYHOOKS_TEST_REDIS_URL"], max_connections=1))
    viewing = RedisExecutionStore(
        redis, "jobs", viewer_client=viewer, config=store.config, key_prefix=store_prefix(store)
    )
    try:
        control = await claim_one(viewing, lease_ms=10_000)
        blocked = asyncio.create_task(viewing.wait_chunks("run_1", CHUNK_CURSOR_START, 5))
        await asyncio.sleep(0.05)

        with pytest.raises(ExecutionStoreError):
            await viewing.wait_chunks("run_1", CHUNK_CURSOR_START, 5)
        await viewing.transition("run_1", Heartbeat(control.fence, "worker", 0, 10_000))
        await viewing.append_chunks("run_1", 1, control.fence, "worker", [b"chunk"])
        assert [chunk.data for chunk in await blocked] == [b"chunk"]

        deployment = DurableDeployment("jobs", "v1", viewing, SSERequest, lambda _context, _request: None)
        cancelled = asyncio.create_task(deployment.wait_chunks("run_1", (await blocked)[0].cursor, 5))
        await asyncio.sleep(0.05)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await deployment.close()
        assert await viewer.ping()
        assert await viewing.wait_chunks("run_1", CHUNK_CURSOR_START, 0.05) == await blocked
    finally:
        await viewer.aclose()


async def test_wait_detects_history_trimmed_while_blocked(redis_store) -> None:
    _, store = redis_store
    control = await claim_one(store, lease_ms=10_000)
    await store.append_chunks("run_1", 1, control.fence, "worker", [b"old"])
    cursor = (await store.read_chunks("run_1", CHUNK_CURSOR_START))[-1].cursor
    waiting = asyncio.create_task(store.wait_chunks("run_1", cursor, 5))
    await asyncio.sleep(0.05)

    await store.append_chunks(
        "run_1",
        1,
        control.fence,
        "worker",
        [str(index).encode() for index in range(2 * store.config.max_stream_chunks)],
    )

    with pytest.raises(ChunkCursorExpiredError):
        await waiting


async def test_commits_beyond_the_lua_argument_limit(redis_store) -> None:
    redis, store = redis_store
    many = RedisExecutionStore(
        redis,
        "jobs",
        config=replace(store.config, max_progress_events=10_000),
        key_prefix=store_prefix(store),
    )
    control = await claim_one(many, lease_ms=10_000)
    events = tuple(str(index).encode() for index in range(8_000))

    await many.transition("run_1", Checkpoint(control.fence, "worker", 0, 10_000, b"checkpoint", events))

    stored = await many.read("run_1")
    assert stored is not None and [event.data for event in stored.progress] == list(events)


async def test_sse_streams_through_redis_viewer_client(redis_store) -> None:
    redis, store = redis_store
    viewer = Redis.from_url(os.environ["HAYHOOKS_TEST_REDIS_URL"])
    viewing = RedisExecutionStore(
        redis,
        "jobs",
        viewer_client=viewer,
        config=replace(store.config, max_stream_chunks=100),
        key_prefix=store_prefix(store),
    )

    async def stream(context: DurableContext, request: SSERequest) -> dict[str, int]:
        for index in range(request.chunks):
            await context.stream_chunk({"index": index})
        return {"chunks": request.chunks}

    deployment = DurableDeployment(
        "jobs",
        "v1",
        viewing,
        SSERequest,
        stream,
        config=RuntimeConfig(poll_interval_seconds=0.05, lease_duration_ms=500),
    )
    app = FastAPI()
    app.include_router(create_durable_router(deployment, owner_id_dependency=None))
    await deployment.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            submitted = (await client.post("/run-durable", json={"chunks": 5})).json()
            body = (await client.get(submitted["links"]["stream"])).text
    finally:
        await deployment.close()
        await viewer.aclose()

    events = [line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")]
    assert events == ["chunk"] * 5 + ["completed"]
