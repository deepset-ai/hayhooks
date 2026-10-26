"""Removal receipts against the reference store and isolated real Valkey namespaces."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import textwrap
import threading
import uuid
from dataclasses import FrozenInstanceError, replace

import pytest
from pydantic import BaseModel

from hayhooks.durable import RemovalScope
from hayhooks.durable.engine import (
    Claim,
    ExecutionLeaseLostError,
    Heartbeat,
    ReleaseClaim,
    RequestCancellation,
    initial_control,
)
from hayhooks.durable.runtime import DurableDeployment, RuntimeConfig
from hayhooks.durable.store import ExecutionAdmissionError, ExecutionStoreError, MemoryExecutionStore, StoreConfig


class Request(BaseModel):
    value: int


@pytest.fixture(params=["memory", pytest.param("redis", marks=pytest.mark.integration)])
async def removal_store(request):
    config = StoreConfig(lease_commit_safety_ms=10, max_nonterminal_executions=1000)
    if request.param == "memory":
        yield MemoryExecutionStore("jobs", config=config)
        return
    url = os.getenv("HAYHOOKS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set HAYHOOKS_TEST_REDIS_URL for isolated Valkey tests")
    from redis.asyncio import Redis
    from hayhooks.durable.redis import RedisExecutionStore

    redis = Redis.from_url(url)
    prefix = f"removal-test:{uuid.uuid4().hex}"
    store = RedisExecutionStore(redis, "jobs", key_prefix=prefix, config=config)
    await store.initialize()
    try:
        yield store
    finally:
        keys = [key async for key in redis.scan_iter(match=f"{prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


def deployment(store, runner, *, family=None, release_running=False):
    return DurableDeployment(
        "jobs",
        "v1",
        store,
        Request,
        runner,
        owner_family=family,
        config=RuntimeConfig(
            lease_duration_ms=300,
            poll_interval_seconds=0.005,
            maintenance_interval_seconds=0.01,
            shutdown_grace_seconds=0.02,
            release_running_on_close=release_running,
        ),
    )


async def complete_proof(member):
    async with asyncio.timeout(4):
        while True:
            proof = await member.quiesce_for_removal(0.05)
            if proof.complete:
                return proof
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("stage", ["before_commit", "after_commit"])
async def test_delayed_claim_cannot_enter_application_after_removal(removal_store, stage):
    store = removal_store
    claiming, release, invoked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = store.claim

    async def delayed(command):
        if stage == "before_commit":
            claiming.set()
            await release.wait()
        result = await original(command)
        if stage == "after_commit" and result is not None:
            claiming.set()
            await release.wait()
        return result

    async def runner(context, request):
        invoked.set()
        return request.value

    store.claim = delayed
    member = deployment(store, runner)
    try:
        await member.start()
        if stage == "before_commit":
            await asyncio.wait_for(claiming.wait(), 1)
        await member.submit({"value": 1})
        await asyncio.wait_for(claiming.wait(), 1)
        proof = await member.quiesce_for_removal(0.02)
        assert not proof.complete and proof.worker_tasks == 1
        release.set()
        proof = await complete_proof(member)
        assert not invoked.is_set()
        assert proof.owned_leases == 0 and proof.store_snapshot.nonterminal == 1
        assert await member.quiesce_for_removal() is proof
        with pytest.raises(FrozenInstanceError):
            proof.complete = False
    finally:
        release.set()
        await member.close()
        await member.wait_drained()


async def test_existing_execution_renews_and_finishes_through_removal(removal_store):
    entered, release = asyncio.Event(), asyncio.Event()

    async def runner(context, request):
        entered.set()
        await release.wait()
        return request.value

    member = deployment(removal_store, runner, family="pod-uid-active")
    try:
        await member.start()
        await member.submit({"value": 1})
        await asyncio.wait_for(entered.wait(), 1)
        proof = await member.quiesce_for_removal(0.02)
        assert not proof.complete and proof.owned_leases == 1
        await asyncio.sleep(0.4)
        snapshot = await removal_store.removal_snapshot(scope=proof.scope)
        assert snapshot.owned_leases == 1
        release.set()
        proof = await complete_proof(member)
        assert proof.store_snapshot.nonterminal == 0
    finally:
        release.set()
        await member.close()
        await member.wait_drained()


async def test_cancelled_submit_retains_write_and_fences_all_public_mutations(removal_store):
    entered, release = asyncio.Event(), asyncio.Event()
    original = removal_store.submit

    async def blocked(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    async def runner(context, request):
        return request.value

    removal_store.submit = blocked
    member = deployment(removal_store, runner)
    try:
        await member.start()
        submit = asyncio.create_task(member.submit({"value": 1}))
        await asyncio.wait_for(entered.wait(), 1)
        submit.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submit
        proof = await member.quiesce_for_removal(0.02)
        assert not proof.complete and proof.pending_writes >= 1
        for operation in (member.submit({"value": 2}), member.cancel("missing"), member.resume("missing")):
            with pytest.raises(ExecutionAdmissionError):
                await operation
        release.set()
        proof = await complete_proof(member)
        assert proof.store_snapshot.nonterminal == 1
    finally:
        release.set()
        await member.close()
        await member.wait_drained()


async def test_released_threads_still_block_removal(removal_store):
    entered, release = threading.Event(), threading.Event()

    def runner(context, request):
        entered.set()
        release.wait(4)
        return request.value

    member = deployment(removal_store, runner, release_running=True)
    try:
        await member.start()
        await member.submit({"value": 1})
        assert await asyncio.to_thread(entered.wait, 1)
        await member.close()
        await asyncio.wait_for(member.wait_drained(), 1)
        proof = await member.quiesce_for_removal(0.02)
        assert not proof.complete and proof.local_active >= 1
        release.set()
        assert (await complete_proof(member)).complete
    finally:
        release.set()
        await member.close()
        await member.wait_drained()


async def test_same_family_overlap_and_restart_cannot_escape_fence(removal_store):
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    def runner_for(index):
        async def runner(context, request):
            entered[index].set()
            await release[index].wait()
            return request.value

        return runner

    first = deployment(removal_store, runner_for(0), family="trusted-pod-uid")
    second = deployment(removal_store, runner_for(1), family="trusted-pod-uid")
    try:
        await first.start()
        await first.submit({"value": 1})
        await asyncio.wait_for(entered[0].wait(), 1)
        await second.start()
        await second.submit({"value": 2})
        await asyncio.wait_for(entered[1].wait(), 1)
        await first.quiesce_for_removal(0.02)
        release[0].set()
        await asyncio.sleep(0.03)
        proof = await first.quiesce_for_removal(0.02)
        assert not proof.complete and proof.owned_leases == 1
        release[1].set()
        proof = await complete_proof(first)
        assert proof.scope.kind == "owner_family"
        restarted = deployment(removal_store, runner_for(0), family="trusted-pod-uid")
        with pytest.raises(ExecutionAdmissionError, match="fenced"):
            await restarted.start()
    finally:
        for event in release:
            event.set()
        await second.quiesce_for_removal(0.1)
        await asyncio.gather(first.close(), second.close())
        await asyncio.gather(first.wait_drained(), second.wait_drained())


async def add_claim(store, index, worker, *, lease_ms=5000):
    run_id = f"{index:032x}"
    digest = hashlib.sha256(run_id.encode()).hexdigest()
    control = initial_control(
        run_id=run_id,
        idempotency_digest=digest,
        idempotency_binding_digest=digest,
        deployment="jobs",
        definition_revision="v1",
        owner_id=None,
        kind="pipeline",
        now_ms=0,
    )
    await store.submit(control, b"{}")
    return await store.transition(run_id, Claim(worker, 0, lease_ms, 3, "v1", b"{}"))


async def test_scope_probe_exact_bound_saturation_and_namespace_counts(removal_store):
    scope = RemovalScope("owner_family", "a" * 64)
    other = f"hh1:{'b' * 64}:{'c' * 16}-0"
    for index in range(200):
        await add_claim(removal_store, index, other)
    snapshot = await removal_store.removal_snapshot(scope=scope)
    assert snapshot.nonterminal == 200 and snapshot.active_leases == 200
    assert snapshot.owned_leases == 0 and not snapshot.blockers
    extra = await add_claim(removal_store, 200, other)
    snapshot = await removal_store.removal_snapshot(scope=scope)
    assert snapshot.owned_leases is None and snapshot.blockers == ("lease_probe_saturated",)
    namespace = await removal_store.removal_snapshot()
    assert namespace.nonterminal == namespace.active_leases == 201
    await removal_store.transition(extra.next_control.run_id, ReleaseClaim(extra.next_control.fence, other))
    assert (await removal_store.removal_snapshot(scope=scope)).owned_leases == 0


async def test_legacy_ownership_is_unknown_for_family_proof(removal_store):
    scope = RemovalScope("owner_family", "a" * 64)
    plan = await add_claim(removal_store, 1, "old-unknown-worker")
    snapshot = await removal_store.removal_snapshot(scope=scope)
    assert snapshot.owned_leases is None
    assert snapshot.blockers == ("legacy_lease_ownership_unknown",)
    await removal_store.transition(
        plan.next_control.run_id, ReleaseClaim(plan.next_control.fence, "old-unknown-worker")
    )
    assert (await removal_store.removal_snapshot(scope=scope)).owned_leases == 0


async def test_fenced_family_heartbeat_stays_visible_and_cannot_resurrect_expired_lease(removal_store):
    scope = RemovalScope("owner_family", "a" * 64)
    worker = f"hh1:{scope.identity}:{'b' * 16}-0"
    plan = await add_claim(removal_store, 1, worker, lease_ms=180)
    await removal_store.fence_owner_scope(scope)
    await asyncio.sleep(0.08)
    renewed = await removal_store.transition(
        plan.next_control.run_id, Heartbeat(plan.next_control.fence, worker, 0, 240)
    )
    await asyncio.sleep(0.13)
    snapshot = await removal_store.removal_snapshot(scope=scope)
    assert snapshot.owned_leases == 1
    await asyncio.sleep(0.14)
    assert (await removal_store.removal_snapshot(scope=scope)).owned_leases == 0
    with pytest.raises(ExecutionLeaseLostError):
        await removal_store.transition(renewed.next_control.run_id, Heartbeat(plan.next_control.fence, worker, 0, 240))


async def test_unconfirmed_admitted_write_blocks_removal_even_after_store_recovers(removal_store):
    original = removal_store.submit

    async def unconfirmed(*args):
        await original(*args)
        raise ExecutionStoreError("the submission committed but its reply was lost")

    async def runner(context, request):
        return request.value

    removal_store.submit = unconfirmed
    member = deployment(removal_store, runner)
    try:
        await member.start()
        with pytest.raises(ExecutionStoreError):
            await member.submit({"value": 1})
        proof = await member.quiesce_for_removal(0.1)
        assert not proof.complete
        assert "engine_write_unconfirmed" in proof.blockers
        assert not (await member.quiesce_for_removal(0.1)).complete
    finally:
        await member.close()
        await member.wait_drained()


async def test_removal_during_preparation_returns_claim_without_invoking_runner(removal_store):
    preparing, release, invoked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = removal_store.read

    async def blocked_read(*args):
        preparing.set()
        await release.wait()
        return await original(*args)

    async def runner(context, request):
        invoked.set()
        return request.value

    removal_store.read = blocked_read
    member = deployment(removal_store, runner)
    try:
        await member.start()
        await member.submit({"value": 1})
        await asyncio.wait_for(preparing.wait(), 1)
        assert not (await member.quiesce_for_removal(0.02)).complete
        release.set()
        proof = await complete_proof(member)
        assert not invoked.is_set()
        assert proof.store_snapshot.nonterminal == 1
    finally:
        release.set()
        await member.close()
        await member.wait_drained()


async def test_removal_overlapping_start_cannot_reactivate_deployment(removal_store):
    initializing, release = asyncio.Event(), asyncio.Event()
    original = removal_store.owner_scope_fenced

    async def blocked_fence_read(scope):
        result = await original(scope)
        initializing.set()
        await release.wait()
        return result

    async def runner(context, request):
        return request.value

    removal_store.owner_scope_fenced = blocked_fence_read
    member = deployment(removal_store, runner)
    starting = asyncio.create_task(member.start())
    try:
        await asyncio.wait_for(initializing.wait(), 1)
        await complete_proof(member)
        release.set()
        with pytest.raises(ExecutionAdmissionError, match="removal"):
            await starting
        await asyncio.wait_for(member.wait_drained(), 1)
    finally:
        release.set()
        await asyncio.gather(starting, return_exceptions=True)
        await member.close()


async def test_cancelled_removal_observer_does_not_cancel_the_durable_fence(removal_store):
    fencing, release = asyncio.Event(), asyncio.Event()
    original = removal_store.fence_owner_scope

    async def blocked(scope):
        fencing.set()
        await release.wait()
        await original(scope)

    async def runner(context, request):
        return request.value

    removal_store.fence_owner_scope = blocked
    member = deployment(removal_store, runner)
    try:
        await member.start()
        observing = asyncio.create_task(member.quiesce_for_removal(10))
        await asyncio.wait_for(fencing.wait(), 1)
        observing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observing
        proof = await member.quiesce_for_removal(0.01)
        assert not proof.complete and "owner_fence_pending" in proof.blockers
        release.set()
        assert (await complete_proof(member)).complete
    finally:
        release.set()
        await member.close()
        await member.wait_drained()


async def test_unavailable_snapshot_is_unknown_then_recovers(removal_store):
    original = removal_store.removal_snapshot

    async def unavailable(**kwargs):
        raise ExecutionStoreError("the ownership snapshot is unavailable")

    async def runner(context, request):
        return request.value

    removal_store.removal_snapshot = unavailable
    member = deployment(removal_store, runner)
    try:
        await member.start()
        proof = await member.quiesce_for_removal(0.1)
        assert not proof.complete and proof.owned_leases is None
        assert "store_probe_unavailable" in proof.blockers
        removal_store.removal_snapshot = original
        assert (await complete_proof(member)).complete
    finally:
        await member.close()
        await member.wait_drained()


async def test_unsupported_store_cannot_produce_removal_receipt():
    class LegacyStore:
        def __init__(self):
            self.delegate = MemoryExecutionStore("jobs", config=StoreConfig(lease_commit_safety_ms=10))

        def __getattr__(self, name):
            if name in {"owner_scope_fenced", "fence_owner_scope", "removal_snapshot"}:
                raise AttributeError(name)
            return getattr(self.delegate, name)

    async def runner(context, request):
        return request.value

    member = deployment(LegacyStore(), runner)
    try:
        await member.start()
        proof = await member.quiesce_for_removal(0.02)
        assert not proof.complete and proof.owned_leases is None
        assert "store_removal_unsupported" in proof.blockers
    finally:
        await member.close()
        await member.wait_drained()


async def test_cancelled_mutation_is_retained_through_close_and_wait_drained(removal_store):
    running, cancelling, release_cancel = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = removal_store.transition

    async def blocked_cancel(run_id, command):
        if isinstance(command, RequestCancellation):
            cancelling.set()
            await release_cancel.wait()
        return await original(run_id, command)

    async def runner(context, request):
        running.set()
        await asyncio.Event().wait()

    removal_store.transition = blocked_cancel
    member = deployment(removal_store, runner)
    try:
        await member.start()
        submitted = await member.submit({"value": 1})
        await asyncio.wait_for(running.wait(), 1)
        cancellation = asyncio.create_task(member.cancel(submitted.control.run_id))
        await asyncio.wait_for(cancelling.wait(), 1)
        cancellation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancellation
        await member.close()
        draining = asyncio.create_task(member.wait_drained())
        await asyncio.sleep(0.02)
        assert not draining.done()
        release_cancel.set()
        await asyncio.wait_for(draining, 1)
    finally:
        release_cancel.set()
        await member.close()
        await member.wait_drained()


@pytest.mark.integration
@pytest.mark.parametrize("removal_store", ["redis"], indirect=True)
async def test_family_fence_survives_a_fresh_process(removal_store):
    async def runner(context, request):
        return request.value

    member = deployment(removal_store, runner, family="trusted-pod-uid-restart")
    try:
        await member.start()
        assert (await complete_proof(member)).complete
        program = textwrap.dedent("""
            import asyncio
            import sys
            from pydantic import BaseModel
            from redis.asyncio import Redis
            from hayhooks.durable.redis import RedisExecutionStore
            from hayhooks.durable.runtime import DurableDeployment
            from hayhooks.durable.store import ExecutionAdmissionError

            class Request(BaseModel):
                value: int

            async def runner(context, request):
                raise AssertionError("a fenced process must never execute")

            async def main():
                redis = Redis.from_url(sys.argv[1])
                store = RedisExecutionStore(redis, "jobs", key_prefix=sys.argv[2])
                member = DurableDeployment(
                    "jobs", "v1", store, Request, runner,
                    owner_family="trusted-pod-uid-restart",
                )
                try:
                    try:
                        await member.start()
                    except ExecutionAdmissionError:
                        print("fenced")
                    else:
                        raise AssertionError("a restarted family escaped its fence")
                finally:
                    await member.close()
                    await member.wait_drained()
                    await redis.aclose()

            asyncio.run(main())
        """)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", program,
            os.environ["HAYHOOKS_TEST_REDIS_URL"],
            removal_store.keys.base.rsplit(":{", 1)[0],
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, stderr.decode()
        assert stdout.decode().strip() == "fenced"
    finally:
        await member.close()
        await member.wait_drained()
