"""Real-Redis contract tests for the isolated durable namespace."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from hayhooks.durable.backend import CHUNK_CURSOR_START
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
from hayhooks.durable.store import ExecutionStore
from tests.durable_contract import assert_store_contract, contract_config, control
from tests.durable_helpers import wait_for_record

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
    monkeypatch.setattr("hayhooks.durable.redis.CHUNK_READ_COUNT", 2)
    _, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, _ = await _claim(durable)
    for index in range(3):
        await durable.append_chunk(run_id, 1, b'{"i":%d}' % index)

    first = await durable.read_chunks(run_id, CHUNK_CURSOR_START, block_ms=0)
    assert [data for _, _, data in first] == [b'{"i":0}', b'{"i":1}']
    rest = await durable.read_chunks(run_id, first[-1][0], block_ms=0)
    assert [data for _, _, data in rest] == [b'{"i":2}']


async def test_blocking_chunk_read_wakes_on_append_and_times_out_empty(store) -> None:
    _, durable = store
    await durable.submit(control(), b"{}", binding_digest="b" * 64)
    run_id, _ = await _claim(durable)
    assert await durable.read_chunks(run_id, CHUNK_CURSOR_START, block_ms=20) == []
    reader = asyncio.create_task(durable.read_chunks(run_id, CHUNK_CURSOR_START, block_ms=5_000))
    await asyncio.sleep(0.05)
    await durable.append_chunk(run_id, 1, b'{"live":true}')
    assert [data for _, _, data in await reader] == [b'{"live":true}']


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
