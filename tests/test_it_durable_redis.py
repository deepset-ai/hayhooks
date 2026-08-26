"""Real-Redis checks for cross-process store invariants."""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import replace

import pytest
from redis.asyncio import Redis

from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionStatus,
    Heartbeat,
    InvalidExecutionTransitionError,
    PayloadKind,
    RequestCancellation,
    Resume,
    Suspend,
)
from hayhooks.durable.redis import RedisExecutionStore, RedisKeys
from hayhooks.durable.store import ExecutionAdmissionError, ExecutionStoreCorruptionError
from tests.durable_store_contract import (
    ATTEMPTS_ERROR,
    CONTRACT_CONFIG,
    REVISION_ERROR,
    assert_store_contract,
    contract_control,
)

pytestmark = pytest.mark.integration


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


async def test_concurrent_submissions_and_claims_have_one_winner(redis_store) -> None:
    redis, store = redis_store
    submissions = await asyncio.gather(
        *(store.submit(contract_control("jobs", f"run_{index}"), b"input") for index in range(20))
    )
    assert sum(result.created for result in submissions) == 1
    assert {result.control.run_id for result in submissions} == {submissions[0].control.run_id}

    claims = await asyncio.gather(
        *(
            store.claim(Claim(f"worker-{index}", 0, 10_000, 3, "v1", REVISION_ERROR, ATTEMPTS_ERROR))
            for index in range(20)
        )
    )
    winners = [plan for plan in claims if plan is not None and plan.next_control.status is ExecutionStatus.RUNNING]
    assert len(winners) == 1
    assert await redis.zcard(store.keys.runnable) == 0
    assert await redis.zcard(store.keys.lease_expiry) == 1


async def test_concurrent_progress_and_cancellation_remain_atomic(redis_store) -> None:
    redis, store = redis_store
    control = contract_control("jobs")
    await store.submit(control, b"input")
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", REVISION_ERROR, ATTEMPTS_ERROR))
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
    assert (await store.operational_counts())["nonterminal"] == 0
    await store.append_chunk(control.run_id, 1, b"late")
    assert await redis.pttl(store.keys.chunks(control.run_id)) > 0
    assert await redis.pttl(store.keys.idempotency(control.idempotency_digest)) > 0
    await asyncio.sleep(1.1)
    assert await store.read(control.run_id) is None
    assert not await redis.exists(store.keys.chunks(control.run_id))
    assert not await redis.exists(store.keys.idempotency(control.idempotency_digest))


async def test_concurrent_resume_commits_one_checkpoint(redis_store) -> None:
    _, store = redis_store
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", REVISION_ERROR, ATTEMPTS_ERROR))
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
    assert await store.operational_counts() == {"nonterminal": 1, "runnable": 1, "lease_expiry": 0}


async def test_admission_heartbeat_and_stale_lease_repair_are_transactional(redis_store, monkeypatch) -> None:
    redis, store = redis_store
    limited = RedisExecutionStore(
        redis,
        "limited",
        config=replace(CONTRACT_CONFIG, max_nonterminal_executions=1),
        key_prefix=f"{store.keys.base.rsplit(':{', maxsplit=1)[0]}:limited",
    )
    controls = (
        contract_control("limited", "run_1", idempotency="one", binding="one"),
        contract_control("limited", "run_2", idempotency="two", binding="two"),
    )
    results = await asyncio.gather(*(limited.submit(control, b"input") for control in controls), return_exceptions=True)
    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, ExecutionAdmissionError) for result in results) == 1

    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 10_000, 3, "v1", REVISION_ERROR, ATTEMPTS_ERROR))
    assert claimed is not None
    with monkeypatch.context() as patch:
        patch.setattr(store, "_apply_plan", lambda *_args, **_kwargs: pytest.fail("heartbeat rewrote control"))
        heartbeat = await store.transition("run_1", Heartbeat(1, "worker", 0, 10_000))
    assert heartbeat.next_control.version == claimed.next_control.version
    live_member = RedisKeys.lease_member("run_1", claimed.next_control.fence)
    await redis.zadd(store.keys.lease_expiry, {RedisKeys.lease_member("run_1", 0): 0})
    await store.maintain(
        max_run_attempts=3,
        worker_revision="v1",
        revision_error=REVISION_ERROR,
        attempts_error=ATTEMPTS_ERROR,
    )
    assert await redis.zrange(store.keys.lease_expiry, 0, -1) == [live_member.encode()]
