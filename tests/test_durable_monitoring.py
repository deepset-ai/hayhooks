"""Owner-scoped, control-only bounded enumeration in Memory and real Redis."""

import asyncio
import base64
import json
import os
import uuid
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel
from redis.asyncio import Redis

from hayhooks.durable.engine import Claim, Complete, RequestCancellation, Suspend, ExecutionStatus
from hayhooks.durable.fastapi import create_durable_router
from hayhooks.durable.redis import RedisExecutionStore
from hayhooks.durable.runtime import DurableDeployment
from hayhooks.durable.store import ControlQuery, MemoryExecutionStore, StoreConfig, ExecutionStoreError, creation_member, monitoring_scope
from tests.durable_store_contract import contract_control


@pytest.fixture(params=["memory", pytest.param("redis", marks=pytest.mark.integration)])
async def monitored_store(request):
    config = StoreConfig(monitoring_enabled=True, lease_commit_safety_ms=10, terminal_ttl_seconds=1,
                         max_nonterminal_executions=0)
    if request.param == "memory":
        yield MemoryExecutionStore("jobs", config=config, clock=lambda: 1000)
        return
    url = os.environ.get("HAYHOOKS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set HAYHOOKS_TEST_REDIS_URL for real Redis")
    redis = Redis.from_url(url, decode_responses=False)
    prefix = f"hayhooks:monitoring:test:{uuid.uuid4().hex}"
    store = RedisExecutionStore(redis, "jobs", config=config, key_prefix=prefix)
    await store.initialize()
    try:
        yield store
    finally:
        keys = [key async for key in redis.scan_iter(match=f"{prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


async def test_collection_contract_filters_replays_and_read_only(monitored_store):
    store = monitored_store
    for i in range(205):
        control = replace(contract_control("jobs", f"run_{i:03}", idempotency=f"idem_{i}"),
                          definition_revision="old" if i == 204 else "v1", kind="agent" if i == 204 else "pipeline")
        await store.submit(control, b"input")
    private = replace(contract_control("jobs", "other_owner", idempotency="other"), owner_id="someone_else")
    await store.submit(private, b"input")
    replay = await store.submit(contract_control("jobs", "ignored", idempotency="idem_0"), b"input")
    assert not replay.created and replay.control.run_id == "run_000"
    query = ControlQuery(owner_id="owner", limit=50)
    page = await store.list_controls(query)
    assert len(page.controls) == 50 and page.has_more
    assert page.next_cursor and "other_owner" not in base64.urlsafe_b64decode(page.next_cursor).decode()
    assert {c.owner_id for c in page.controls} == {"owner"}
    with pytest.raises(ValueError):
        ControlQuery(owner_id="someone_else", cursor=page.next_cursor)
    observed = list(page.controls)
    while page.has_more:
        page = await store.list_controls(replace(query, cursor=page.next_cursor))
        observed.extend(page.controls)
    assert len({c.run_id for c in observed}) == 205
    filtered = await store.list_controls(ControlQuery(owner_id="owner", search="204", revision="old"))
    assert not filtered.controls and filtered.has_more  # candidate budget, not an exact filtered total
    filtered = await store.list_controls(ControlQuery(owner_id="owner", search="204", revision="old", cursor=filtered.next_cursor))
    assert filtered.controls[0].kind == "agent" and filtered.controls[0].definition_revision == "old"
    before = await store.read("run_204")
    counts = await store.operational_counts(revision="old")
    await store.list_controls(ControlQuery(owner_id="owner", status=ExecutionStatus.RUNNING))
    assert before == await store.read("run_204")
    assert counts == await store.operational_counts(revision="old")


async def test_creation_ties_concurrent_insertion_and_waiting(monitored_store):
    store = monitored_store
    submitted = await store.submit(contract_control("jobs", "waiting", idempotency="waiting"), b"input")
    claim = await store.transition("waiting", Claim("worker", 0, 30000, 3, "v1", b"attempts"))
    await store.transition("waiting", Suspend(claim.next_control.fence, "worker", 0, b"checkpoint", b'{"kind":"approval"}'))
    for name in ("a", "c", "b"):
        await store.submit(contract_control("jobs", name, idempotency=name), b"input")
    page = await store.list_controls(ControlQuery(owner_id="owner", limit=1))
    first = page.controls[0]
    await store.submit(contract_control("jobs", "later", idempotency="later"), b"input")
    seen = [first.run_id]
    while page.has_more:
        page = await store.list_controls(ControlQuery(owner_id="owner", limit=1, cursor=page.next_cursor))
        seen.extend(c.run_id for c in page.controls)
    assert len(set(seen)) == len(seen)
    assert "waiting" in seen and "later" in seen
    assert (await store.read_control("waiting")).created_at_ms == submitted.control.created_at_ms
    if isinstance(store, MemoryExecutionStore):
        assert seen == sorted(seen)


async def test_ttl_cleanup_keeps_live_records(monitored_store):
    store = monitored_store
    for name in ("old_live", "terminal"):
        await store.submit(contract_control("jobs", name, idempotency=name), b"input")
    await store.transition("terminal", RequestCancellation(0))
    if isinstance(store, MemoryExecutionStore):
        store._clock = lambda: 2001
    else:
        await asyncio.sleep(1.1)
    page = await store.list_controls(ControlQuery(owner_id="owner"))
    assert [c.run_id for c in page.controls] == ["old_live"]
    assert await store.read_public("terminal") is None
    if isinstance(store, RedisExecutionStore):
        assert await store.redis.zcard(store.keys.execution_expiry) == 0


async def test_redis_index_type_is_checked_before_any_submission_write(monitored_store):
    store = monitored_store
    if not isinstance(store, RedisExecutionStore):
        return
    index = store.keys.executions(monitoring_scope("owner"))
    await store.redis.set(index, "wrong-type")
    control = contract_control("jobs")
    with pytest.raises(ExecutionStoreError):
        await store.submit(control, b"input")
    assert not await store.redis.exists(store.keys.control(control.run_id), store.keys.idempotency(control.idempotency_digest), store.keys.capacity)


async def test_redis_corrupt_and_expired_controls_advance_cursor(monitored_store):
    store = monitored_store
    if not isinstance(store, RedisExecutionStore):
        return
    for i in range(205):
        item = await store.submit(contract_control("jobs", f"run_{i:03}", idempotency=f"i_{i}"), b"input")
        if i < 200:
            await store.redis.hset(store.keys.control(item.control.run_id), "status", "broken")
    first = await store.list_controls(ControlQuery(owner_id="owner"))
    assert not first.controls and first.has_more and first.skipped_controls == 200
    second = await store.list_controls(ControlQuery(owner_id="owner", cursor=first.next_cursor))
    assert len(second.controls) == 5 and not second.has_more


@pytest.mark.parametrize("cursor", ["bad", "!", base64.urlsafe_b64encode(b'{}').decode(), base64.urlsafe_b64encode(b'[1, null]').decode(), "a"*1025])
def test_malformed_cursor_is_rejected(cursor):
    with pytest.raises(ValueError):
        ControlQuery(cursor=cursor)


class RequestModel(BaseModel):
    value: int


async def no_run(context, request):
    return request


def owner_dependency(request: Request):
    if not request.headers.get("X-Owner"):
        raise HTTPException(401)
    return request.headers["X-Owner"]


async def test_collection_authorization_and_public_projection():
    store = MemoryExecutionStore("jobs", config=StoreConfig(monitoring_enabled=True))
    for name, owner in (("alice_1", "alice"), ("bob_1", "bob"), ("alice_2", "alice")):
        await store.submit(replace(contract_control("jobs", name, idempotency=name), owner_id=owner), b"private-input")
    deployment = DurableDeployment("jobs", "v2", store, RequestModel, no_run)
    app = FastAPI()
    app.include_router(create_durable_router(deployment, owner_id_dependency=owner_dependency, monitoring_enabled=True), prefix="/jobs")
    store.read = AsyncMock(side_effect=AssertionError("private read"))
    read_public = store.read_public
    store.read_public = AsyncMock(side_effect=AssertionError("payload read"))
    with TestClient(app) as client:
        assert client.get("/jobs/executions").status_code == 401
        response = client.get("/jobs/executions?limit=1", headers={"X-Owner": "alice"})
        assert response.status_code == 200
        page = response.json()
        assert page["executions"][0]["execution_id"] == "alice_1"
        assert page["executions"][0]["revision"] == "v1"
        assert not {"owner_id", "lease_owner", "fence", "idempotency_digest"}.intersection(page["executions"][0])
        assert "bob_1" not in response.text
        cursor = page["next_cursor"]
        assert client.get("/jobs/executions", params={"cursor": cursor}, headers={"X-Owner": "bob"}).status_code == 422
        store.read_public = read_public
        assert client.get("/jobs/executions/bob_1", headers={"X-Owner": "alice"}).status_code == 404
        for params in ({"limit": 101}, {"status": "unknown"}, {"cursor": "bad"}):
            assert client.get("/jobs/executions", params=params, headers={"X-Owner": "alice"}).status_code == 422


@pytest.mark.parametrize("operator", [None, lambda: False, lambda: True])
async def test_unscoped_enumeration_requires_explicit_operator(operator):
    store = MemoryExecutionStore("jobs", config=StoreConfig(monitoring_enabled=True))
    await store.submit(replace(contract_control("jobs"), owner_id=None), b"input")
    await store.submit(replace(contract_control("jobs", "owned", idempotency="owned"), owner_id="owner"), b"input")
    deployment = DurableDeployment("jobs", "v1", store, RequestModel, no_run)
    app = FastAPI()
    app.include_router(create_durable_router(deployment, owner_id_dependency=None, monitoring_enabled=True, monitoring_dependency=operator))
    with TestClient(app) as client:
        response = client.get("/executions")
        assert response.status_code == (200 if operator and operator() is True else 403)
        if response.status_code == 200:
            assert [c["execution_id"] for c in response.json()["executions"]] == ["run_1"]


def test_disabled_route_and_no_historical_backfill():
    store = MemoryExecutionStore("jobs")
    deployment = DurableDeployment("jobs", "v1", store, RequestModel, no_run)
    app = FastAPI()
    app.include_router(create_durable_router(deployment, owner_id_dependency=None))
    with TestClient(app) as client:
        assert client.get("/executions").status_code == 404


async def test_monitoring_observes_runtime_lifecycle_without_writes(monitored_store):
    from hayhooks.durable.runtime import RuntimeConfig
    from typing import Literal

    class LifecycleRequest(BaseModel):
        action: Literal["hold", "wait", "complete", "fail"]

    release = asyncio.Event()
    entered = asyncio.Event()

    async def runner(context, request):
        if request.action == "hold":
            entered.set()
            await release.wait()
        if request.action == "wait":
            await context.suspend({"kind": "approval", "message": "Continue?"})
        if request.action == "fail":
            raise ValueError("intentional qualification failure")
        return {"done": True}

    store = monitored_store
    deployment = DurableDeployment("jobs", "v1", store, LifecycleRequest, runner,
        config=RuntimeConfig(poll_interval_seconds=0.01, shutdown_grace_seconds=0.1))
    await deployment.start()
    try:
        running = await deployment.submit({"action": "hold"})
        await asyncio.wait_for(entered.wait(), 2)
        waiting = await deployment.submit({"action": "wait"})
        completed = await deployment.submit({"action": "complete"})
        failed = await deployment.submit({"action": "fail"})
        canceled = await deployment.submit({"action": "complete"})
        await deployment.cancel(canceled.control.run_id)
        first = await store.list_controls(ControlQuery())
        statuses = {item.status for item in first.controls}
        assert {ExecutionStatus.QUEUED, ExecutionStatus.RUNNING, ExecutionStatus.CANCELED} <= statuses
        release.set()
        for _ in range(100):
            final = await store.list_controls(ControlQuery())
            statuses = {item.status for item in final.controls}
            if {ExecutionStatus.WAITING, ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELED} <= statuses:
                break
            await asyncio.sleep(0.02)
        else:
            assert False, statuses
        before = [await store.read(item.run_id) for item in final.controls]
        await store.list_controls(ControlQuery(search=waiting.control.run_id))
        assert before == [await store.read(item.run_id) for item in final.controls]
    finally:
        release.set()
        await deployment.close()
        await deployment.wait_drained()


async def test_terminal_expiry_index_validation_and_incremental_cleanup(monitored_store):
    store = monitored_store
    for i in range(205):
        await store.submit(contract_control("jobs", f"terminal_{i:03}", idempotency=f"t_{i}"), b"input")
        await store.transition(f"terminal_{i:03}", RequestCancellation(0))
    if isinstance(store, MemoryExecutionStore):
        store._clock = lambda: 2001
    else:
        await asyncio.sleep(1.1)
    # Even if the expired owner's collection is never visited, maintenance of the index stays bounded.
    for _ in range(3):
        await store.list_controls(ControlQuery(owner_id="another-owner"))
    if isinstance(store, RedisExecutionStore):
        assert await store.redis.zcard(store.keys.executions(monitoring_scope("owner"))) == 0
        assert await store.redis.zcard(store.keys.execution_expiry) == 0
        await store.redis.set(store.keys.execution_expiry, "wrong-type")
        with pytest.raises(ExecutionStoreError):
            await store.submit(contract_control("jobs", "new", idempotency="new"), b"input")
        assert await store.read_control("new") is None
    else:
        assert not store._execution_index[monitoring_scope("owner")]


async def test_no_backfill_or_replay_backfill(monitored_store):
    from dataclasses import replace
    store = monitored_store
    store.config = replace(store.config, monitoring_enabled=False)
    old = await store.submit(contract_control("jobs", "before_monitoring", idempotency="old"), b"input")
    store.config = replace(store.config, monitoring_enabled=True)
    replay = await store.submit(contract_control("jobs", "ignored", idempotency="old"), b"input")
    assert not replay.created and replay.control == old.control
    assert not (await store.list_controls(ControlQuery(owner_id="owner"))).controls
    assert await store.read_public("before_monitoring") is not None
