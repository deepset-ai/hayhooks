import asyncio
import os
import shutil
import socket
import sys
import time
import uuid
from dataclasses import replace

import pytest
from redis.asyncio import Redis

from hayhooks.execution import (
    DurableExecutionManager,
    ExecutionError,
    ExecutionKind,
    ExecutionLeaseLostError,
    ExecutionRecord,
    ExecutionRetiredError,
    ExecutionStatus,
    ExecutionStoreError,
    RetryableExecutionError,
)
from hayhooks.redis_execution import EXECUTION_GROUP, RedisExecutionStore
from hayhooks.server.a2a.redis_task_store import RedisTaskStore

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_REDIS_URL_ENV = "HAYHOOKS_TEST_REDIS_URL"
_PROCESS_CLAIM_SCRIPT = """
import asyncio
import sys

from redis.asyncio import Redis

from hayhooks.redis_execution import RedisExecutionStore


async def main():
    redis = Redis.from_url(sys.argv[1], decode_responses=False)
    store = RedisExecutionStore(
        redis,
        key_prefix=sys.argv[2],
        claim_idle_ms=10_000,
        queue_block_ms=1,
        reclaim_interval=0,
        delayed_promotion_interval=0,
    )
    await store.initialize()
    claim = await store.claim_next(sys.argv[3])
    print("owned" if claim is not None else "none", flush=True)
    await redis.aclose()


asyncio.run(main())
"""


@pytest.fixture
async def redis_store():
    redis_url = os.getenv(_REDIS_URL_ENV)
    if not redis_url:
        pytest.skip(f"set {_REDIS_URL_ENV} to run the real-Redis durable contract suite")
    redis = Redis.from_url(redis_url, decode_responses=False)
    try:
        await redis.ping()
    except Exception as error:
        await redis.aclose()
        pytest.fail(f"{_REDIS_URL_ENV} was explicitly configured but Redis is unavailable: {error}")

    prefix = f"hayhooks:test:durable:{uuid.uuid4().hex}"
    store = RedisExecutionStore(
        redis,
        key_prefix=prefix,
        claim_idle_ms=50,
        queue_block_ms=1,
        reclaim_interval=0,
        terminal_ttl_seconds=60,
        max_stream_length=1,
        delayed_promotion_interval=0,
    )
    await store.initialize()
    try:
        yield redis, store
    finally:
        keys = [key async for key in redis.scan_iter(match=f"{prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


def _record(execution_id: str) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=execution_id,
        execution_kind=ExecutionKind.PIPELINE,
        deployment_name="integration",
        definition_revision="revision",
        validated_input={"execution_id": execution_id},
    )


async def _complete_next(store: RedisExecutionStore, worker: str) -> None:
    claim = await store.claim_next(worker)
    assert claim is not None
    async with claim:
        claim.record.status = ExecutionStatus.COMPLETED
        claim.record.result = {"execution_id": claim.record.execution_id}
        await claim.complete()


async def test_blocking_queue_read_wakes_immediately_for_new_work(redis_store) -> None:
    _, store = redis_store
    store.queue_block_ms = 1_000
    waiting_claim = asyncio.create_task(store.claim_next("blocking-reader"))
    await asyncio.sleep(0.05)

    started = time.monotonic()
    assert await store.submit(_record("wake-reader"))
    claim = await asyncio.wait_for(waiting_claim, timeout=0.5)

    assert claim is not None
    assert time.monotonic() - started < 0.5
    async with claim:
        claim.record.status = ExecutionStatus.COMPLETED
        await claim.complete()


async def test_pending_delivery_survives_backlog_larger_than_deprecated_stream_limit(redis_store) -> None:
    redis, store = redis_store
    assert await store.submit(_record("pending"))
    pending = await store.claim_next("original")
    assert pending is not None
    store.claim_idle_ms = 10_000

    for index in range(25):
        execution_id = f"later-{index}"
        assert await store.submit(_record(execution_id))
        await _complete_next(store, f"completion-{index}")

    assert await redis.xlen(store.stream_key) == 1
    store.claim_idle_ms = 1_000
    await redis.xclaim(
        store.stream_key,
        EXECUTION_GROUP,
        "original",
        min_idle_time=0,
        message_ids=[pending.delivery.entry_id],
        idle=2_000,
    )
    await redis.delete(store._lease_key("pending"))
    recovered = await store.claim_next("recovery")
    assert recovered is not None
    assert recovered.record.execution_id == "pending"
    async with recovered:
        recovered.record.status = ExecutionStatus.COMPLETED
        await recovered.complete()


async def test_stale_owner_cannot_write_any_fenced_transition(redis_store) -> None:
    redis, store = redis_store
    assert await store.submit(_record("stale"))
    claim = await store.claim_next("worker")
    assert claim is not None
    await redis.set(store._lease_key("stale"), "replacement", px=1_000)

    claim.record.status = ExecutionStatus.WAITING
    with pytest.raises(ExecutionLeaseLostError):
        await claim.checkpoint()
    with pytest.raises(ExecutionLeaseLostError):
        await claim.suspend()
    with pytest.raises(ExecutionLeaseLostError):
        await claim.retry(ExecutionError(type="Retry", message="again"), delay=0)
    claim.record.status = ExecutionStatus.COMPLETED
    with pytest.raises(ExecutionLeaseLostError):
        await claim.complete()


async def test_retirement_cannot_be_overwritten_by_an_already_claimed_worker(redis_store) -> None:
    redis, store = redis_store
    record = _record("retired-after-claim")
    record.definition_revision = "old-revision"
    assert await store.submit(record)
    claim = await store.claim_next("worker")
    assert claim is not None

    assert await store.retire_incompatible("new-revision") == 1
    claim.record.status = ExecutionStatus.RUNNING
    with pytest.raises(ExecutionRetiredError):
        await claim.checkpoint()

    persisted = await store.get("retired-after-claim")
    assert persisted is not None
    assert persisted.status is ExecutionStatus.FAILED
    assert persisted.error is not None
    assert persisted.error.code == "definition_revision_conflict"
    assert await redis.ttl(store._record_key("retired-after-claim")) > 0


async def test_persisted_cancellation_wins_before_claim_and_terminal_race(redis_store) -> None:
    _, store = redis_store
    assert await store.submit(_record("before-claim"))
    assert await store.request_cancel("before-claim")
    claim = await store.claim_next("worker")
    assert claim is not None
    async with claim:
        assert await claim.cancellation_requested()
        claim.record.status = ExecutionStatus.COMPLETED
        await claim.complete()
    canceled = await store.get("before-claim")
    assert canceled is not None
    assert canceled.status is ExecutionStatus.CANCELED
    assert canceled.cancel_requested_at is not None

    assert await store.submit(_record("race"))
    race = await store.claim_next("worker")
    assert race is not None
    async with race:
        race.record.status = ExecutionStatus.COMPLETED
        assert await store.request_cancel("race")
        await race.complete()
    raced = await store.get("race")
    assert raced is not None
    assert raced.status is ExecutionStatus.CANCELED


async def test_nonterminal_cancellation_has_no_expiry_and_terminal_transition_sets_retention(redis_store) -> None:
    redis, store = redis_store
    store.terminal_ttl_seconds = 30
    assert await store.submit(_record("cancellation-retention"))
    claim = await store.claim_next("worker")
    assert claim is not None

    assert await store.request_cancel("cancellation-retention")
    assert await redis.pttl(store._record_key("cancellation-retention")) == -1

    async with claim:
        claim.record.status = ExecutionStatus.COMPLETED
        await claim.complete()

    ttl = await redis.ttl(store._record_key("cancellation-retention"))
    assert 0 < ttl <= store.terminal_ttl_seconds
    record = await store.get("cancellation-retention")
    assert record is not None
    assert record.status is ExecutionStatus.CANCELED


async def test_retry_deletes_old_delivery_and_promotes_exactly_once(redis_store) -> None:
    redis, store = redis_store
    assert await store.submit(_record("retry"))
    claim = await store.claim_next("worker")
    assert claim is not None
    old_entry_id = claim.delivery.entry_id
    async with claim:
        await claim.retry(ExecutionError(type="Retry", message="temporary", retryable=True), delay=0.03)

    assert await redis.xrange(store.stream_key, min=old_entry_id, max=old_entry_id) == []
    pending = await redis.xpending(store.stream_key, EXECUTION_GROUP)
    assert pending["pending"] == 0
    assert await store.claim_next("early") is None
    await asyncio.sleep(0.04)
    promoted = await store.claim_next("later")
    assert promoted is not None
    assert promoted.record.execution_id == "retry"
    assert promoted.record.error is None
    assert await redis.zscore(store.delayed_key, "retry") is None
    async with promoted:
        promoted.record.status = ExecutionStatus.COMPLETED
        await promoted.complete()


async def test_waiting_is_not_reclaimed_and_concurrent_resume_enqueues_once(redis_store) -> None:
    redis, store = redis_store
    assert await store.submit(_record("waiting"))
    claim = await store.claim_next("worker")
    assert claim is not None
    async with claim:
        claim.record.status = ExecutionStatus.WAITING
        claim.record.wait = {"kind": "approval", "message": "Approve"}
        await claim.suspend()

    assert await store.claim_next("other") is None
    results = await asyncio.gather(*(store.resume("waiting", {"approved": True}) for _ in range(2)))
    assert sorted(results) == [False, True]
    entries = await redis.xrange(store.stream_key)
    assert len(entries) == 1


async def test_two_store_instances_never_own_same_execution(redis_store) -> None:
    redis, first = redis_store
    first.claim_idle_ms = 1_000
    second = RedisExecutionStore(
        redis,
        key_prefix=first.key_prefix,
        claim_idle_ms=1_000,
        queue_block_ms=1,
        reclaim_interval=0,
        delayed_promotion_interval=0,
    )
    await second.initialize()
    assert await first.submit(_record("single-owner"))

    claims = await asyncio.gather(first.claim_next("first"), second.claim_next("second"))
    assert sum(claim is not None for claim in claims) == 1
    owned = next(claim for claim in claims if claim is not None)
    async with owned:
        owned.record.status = ExecutionStatus.COMPLETED
        await owned.complete()


async def test_two_worker_processes_never_own_same_execution(redis_store) -> None:
    _, store = redis_store
    assert await store.submit(_record("process-owner"))

    processes = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _PROCESS_CLAIM_SCRIPT,
            os.environ[_REDIS_URL_ENV],
            store.key_prefix,
            worker,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for worker in ("process-a", "process-b")
    ]
    results = await asyncio.gather(*(process.communicate() for process in processes))
    assert all(process.returncode == 0 for process in processes), [stderr.decode() for _, stderr in results]
    ownership = sorted(stdout.decode().strip() for stdout, _ in results)
    assert ownership == ["none", "owned"]


async def test_a2a_projection_lease_version_and_active_index_use_real_lua(redis_store) -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from a2a.types import Task, TaskState

    redis, execution_store = redis_store
    prefix = f"{execution_store.key_prefix}:a2a"
    first = RedisTaskStore(redis, "agent", key_prefix=prefix, terminal_ttl_seconds=60)
    second = RedisTaskStore(redis, "agent", key_prefix=prefix, terminal_ttl_seconds=60)
    context = SimpleNamespace(user=SimpleNamespace(user_name="owner"))
    task = Task(id="task", context_id="context")
    task.status.state = TaskState.TASK_STATE_WORKING
    task.status.timestamp.FromDatetime(datetime.now(timezone.utc))
    await first.save(task, context)

    batch, cursor = await first.recoverable_task_batch(0, 10)
    assert [item.id for item in batch] == ["task"]
    assert cursor is None

    tokens = await asyncio.gather(
        first.acquire_projection(task.id, lease_ms=1_000),
        second.acquire_projection(task.id, lease_ms=1_000),
    )
    assert sum(token is not None for token in tokens) == 1
    token = next(token for token in tokens if token is not None)
    version = await first.projection_version(task.id)
    assert await first.save_projection(task, token, version) == version + 1
    assert await first.save_projection(task, token, version) == -2

    task.status.state = TaskState.TASK_STATE_COMPLETED
    assert await first.save_projection(task, token, version + 1) == version + 2
    batch, _ = await first.recoverable_task_batch(0, 10)
    assert batch == []
    await redis.zadd(first._terminal_tasks_key(), {task.id: time.time() - 1})
    assert await first.cleanup_expired_tasks() == 1
    assert await first.get_for_execution(task.id) is None
    assert await first.projection_version(task.id) == 0


async def test_two_a2a_replicas_recover_only_owned_projections_and_take_over_lease_loss(
    redis_store,
    monkeypatch,
) -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from a2a.types import Task, TaskState

    from hayhooks.server.a2a.executor import DurableAgentExecutor

    redis, execution_store = redis_store
    prefix = f"{execution_store.key_prefix}:a2a-replicas"
    first_store = RedisTaskStore(redis, "agent", key_prefix=prefix, terminal_ttl_seconds=60)
    second_store = RedisTaskStore(redis, "agent", key_prefix=prefix, terminal_ttl_seconds=60)
    context = SimpleNamespace(user=SimpleNamespace(user_name="owner"))
    task = Task(id="replicated-task", context_id="context")
    task.status.state = TaskState.TASK_STATE_WORKING
    task.status.timestamp.FromDatetime(datetime.now(timezone.utc))
    await first_store.save(task, context)
    record = SimpleNamespace(
        status=ExecutionStatus.RUNNING,
        progress=[],
        result=None,
        error=None,
        updated_at=datetime.now(timezone.utc),
        sequence=0,
    )

    async def get_changed(known_sequences):
        return {
            execution_id: record for execution_id, sequence in known_sequences.items() if sequence != record.sequence
        }

    deployment = SimpleNamespace(
        get=lambda _task_id: asyncio.sleep(0, result=record),
        get_changed=get_changed,
    )
    monkeypatch.setattr("hayhooks.server.a2a.executor.durable_runtime.deployment", lambda *_args: deployment)
    first = DurableAgentExecutor(SimpleNamespace(), "agent", first_store)
    second = DurableAgentExecutor(SimpleNamespace(), "agent", second_store)
    await first.start()
    await second.start()
    try:
        owners = [executor for executor in (first, second) if task.id in executor._projections]
        assert len(owners) == 1
        owner = owners[0]
        standby = second if owner is first else first
        projection = owner._projections[task.id]
        await redis.set(first_store._projection_lease_key(task.id), "replacement", px=1_000)
        projection.lease_renew_at = 0
        await owner._reconcile_one(task.id, projection)
        assert projection.lease_token is None

        await owner.close()
        await redis.delete(first_store._projection_lease_key(task.id))
        standby._next_recovery_scan_at = 0
        standby._wake.set()
        for _ in range(100):
            if task.id in standby._projections:
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("standby A2A replica did not acquire the abandoned projection")
        assert standby._projections[task.id].lease_token is not None
    finally:
        await first.close()
        await second.close()


async def test_real_redis_revision_retirement_rechecks_work_requeued_after_drain(redis_store) -> None:
    _, store = redis_store
    waiting_record = _record("waiting")
    waiting_record.definition_revision = "old"
    assert await store.submit(waiting_record)
    waiting = await store.claim_next("worker")
    assert waiting is not None
    async with waiting:
        waiting.record.status = ExecutionStatus.WAITING
        waiting.record.wait = {"kind": "input"}
        await waiting.suspend()

    running_record = _record("running")
    running_record.definition_revision = "old"
    assert await store.submit(running_record)
    running = await store.claim_next("worker")
    assert running is not None
    async with running:
        queued_record = _record("queued")
        queued_record.definition_revision = "old"
        assert await store.submit(queued_record)
        cancel_requested = _record("cancel-requested")
        cancel_requested.definition_revision = "old"
        assert await store.submit(cancel_requested)
        assert await store.request_cancel(cancel_requested.execution_id)
        assert await store.retire_incompatible("current") == 3
        assert (await store.get("waiting")).status is ExecutionStatus.FAILED
        assert (await store.get("queued")).status is ExecutionStatus.FAILED
        assert (await store.get("cancel-requested")).status is ExecutionStatus.CANCELED
        assert (await store.get("running")).status is ExecutionStatus.RUNNING
        running.record.status = ExecutionStatus.WAITING
        running.record.wait = {"kind": "input"}
        await running.suspend()

    assert await store.retire_incompatible("current") == 1
    retired = await store.get("running")
    assert retired is not None
    assert retired.status is ExecutionStatus.FAILED
    assert retired.error is not None
    assert retired.error.code == "definition_revision_conflict"
    counts = await store.operational_counts()
    assert counts["failed"] == 3
    assert counts["waiting"] == 0


async def test_operational_counts_follow_every_real_redis_state_transition(redis_store) -> None:
    redis, store = redis_store
    assert await store.submit(_record("counted"))
    assert (await store.operational_counts())["queued"] == 1

    claim = await store.claim_next("worker")
    assert claim is not None
    assert (await store.operational_counts())["running"] == 1
    async with claim:
        claim.record.status = ExecutionStatus.WAITING
        claim.record.wait = {"kind": "input"}
        await claim.suspend()
    assert (await store.operational_counts())["waiting"] == 1

    assert await store.resume("counted")
    assert (await store.operational_counts())["queued"] == 1
    resumed = await store.claim_next("worker")
    assert resumed is not None
    async with resumed:
        resumed.record.status = ExecutionStatus.COMPLETED
        await resumed.complete()
    counts = await store.operational_counts()
    assert counts["completed"] == 1
    assert counts["queued"] == counts["running"] == counts["waiting"] == 0

    await redis.delete(store._record_key("counted"))
    await redis.zadd(store.terminal_count_index_key, {"counted": time.time() - 1})
    assert (await store.operational_counts())["completed"] == 0

    assert await store.submit(_record("canceled"))
    canceled_claim = await store.claim_next("worker")
    assert canceled_claim is not None
    async with canceled_claim:
        canceled_claim.record.status = ExecutionStatus.WAITING
        canceled_claim.record.wait = {"kind": "input"}
        await canceled_claim.suspend()
    assert await store.request_cancel("canceled")
    counts = await store.operational_counts()
    assert counts["canceled"] == 1
    assert counts["waiting"] == 0


async def test_sequence_index_avoids_unchanged_reads_and_active_index_drops_terminal_work(redis_store) -> None:
    redis, store = redis_store
    record = _record("indexed")
    assert await store.submit(record)
    assert await redis.hget(store.active_revisions_key, record.execution_id) == record.definition_revision.encode()

    assert await store.get_changed({record.execution_id: record.sequence}) == {}
    assert await store.request_cancel(record.execution_id)
    changed = await store.get_changed({record.execution_id: record.sequence})
    assert changed[record.execution_id] is not None
    assert changed[record.execution_id].cancel_requested_at is not None

    claim = await store.claim_next("indexed-worker")
    assert claim is not None
    async with claim:
        claim.record.mark_canceled()
        await claim.complete()

    assert await redis.hget(store.active_revisions_key, record.execution_id) is None


async def test_same_redis_task_store_rejects_two_stale_in_process_snapshots(redis_store) -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from a2a.types import Task, TaskState
    from a2a.utils.errors import InvalidParamsError

    redis, execution_store = redis_store
    store = RedisTaskStore(redis, "agent", key_prefix=f"{execution_store.key_prefix}:a2a")
    context = SimpleNamespace(user=SimpleNamespace(user_name="owner"))
    task = Task(id="snapshot", context_id="context")
    task.status.state = TaskState.TASK_STATE_WORKING
    task.status.timestamp.FromDatetime(datetime.now(timezone.utc))
    await store.save(task, context)
    first = await store.get(task.id, context)
    stale = await store.get(task.id, context)
    assert first is not None
    assert stale is not None

    first.status.state = TaskState.TASK_STATE_COMPLETED
    await store.save(first, context)
    stale.status.state = TaskState.TASK_STATE_FAILED
    with pytest.raises(InvalidParamsError, match="stale projection version"):
        await store.save(stale, context)

    persisted = await store.get(task.id, context)
    assert persisted is not None
    assert persisted.status.state == TaskState.TASK_STATE_COMPLETED


@pytest.mark.parametrize(
    ("operation", "script_name"),
    [
        ("checkpoint", "checkpoint"),
        ("complete", "complete"),
        ("retry", "retry"),
        ("suspend", "suspend"),
        ("release", "release_lease"),
    ],
)
async def test_manager_recovers_from_each_real_redis_transition_failure(
    redis_store,
    operation: str,
    script_name: str,
) -> None:
    _, store = redis_store
    delegate = getattr(store._scripts, script_name)

    class FailOnceScript:
        failed = False

        async def __call__(self, **kwargs):
            if not self.failed:
                self.failed = True
                msg = f"injected {operation} connection failure"
                raise ConnectionError(msg)
            return await delegate(**kwargs)

    store._scripts = replace(store._scripts, **{script_name: FailOnceScript()})

    async def runner(context):
        if context.attempt == 1:
            if operation == "checkpoint":
                await context.checkpoint()
            elif operation == "retry":
                raise RetryableExecutionError("retry after injected failure")
            elif operation == "suspend":
                await context.suspend({"kind": "test"})
            elif operation == "release":
                raise ExecutionStoreError("release after injected failure")
        return {"recovered": True}

    manager = DurableExecutionManager(
        "integration",
        store,
        runner,
        adapter=object(),
        poll_interval=0.005,
        retry_base_delay=0,
    )
    await manager.start()
    try:
        assert await manager.submit(_record(f"transition-{operation}"))
        for _ in range(1_000):
            persisted = await store.get(f"transition-{operation}")
            if persisted is not None and persisted.terminal:
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail(f"manager did not recover from {operation} failure")
    finally:
        await manager.close()

    assert persisted is not None
    assert persisted.status is ExecutionStatus.COMPLETED
    assert persisted.result == {"recovered": True}
    assert manager.health["store_error_count"] >= 1


async def test_redis_restart_with_aof_keeps_nonterminal_work_recoverable(tmp_path) -> None:
    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server executable is required for the restart contract")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        executable,
        "--bind",
        "127.0.0.1",
        "--port",
        str(port),
        "--dir",
        str(tmp_path),
        "--appendonly",
        "yes",
        "--appendfsync",
        "always",
        "--save",
        "",
    ]

    async def start_server():
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        client = Redis.from_url(f"redis://127.0.0.1:{port}/0", decode_responses=False)
        for _ in range(100):
            try:
                if await client.ping():
                    return process, client
            except Exception:
                await asyncio.sleep(0.01)
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5)
        pytest.fail("temporary Redis server did not start")

    process, redis = await start_server()
    prefix = f"hayhooks:test:restart:{uuid.uuid4().hex}"
    try:
        store = RedisExecutionStore(redis, key_prefix=prefix, claim_idle_ms=500)
        await store.initialize()
        assert await store.submit(_record("survives-restart"))
        await redis.shutdown(now=False, force=False, abort=False)
        await asyncio.wait_for(process.wait(), timeout=5)
        await redis.aclose()

        process, redis = await start_server()
        recovered_store = RedisExecutionStore(redis, key_prefix=prefix, claim_idle_ms=500)
        await recovered_store.initialize()
        claim = await recovered_store.claim_next("after-restart")
        assert claim is not None
        assert claim.record.execution_id == "survives-restart"
        async with claim:
            claim.record.status = ExecutionStatus.COMPLETED
            await claim.complete()
    finally:
        await redis.aclose()
        if process.returncode is None:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5)
