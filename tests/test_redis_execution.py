import asyncio

import pytest

from hayhooks.execution import ExecutionKind, ExecutionRecord, ExecutionStoreError
from hayhooks.redis_execution import RedisExecutionStore, RedisExecutionStoreProvider
from hayhooks.settings import settings


class FakeRegisteredScript:
    def __init__(self, redis, source):
        self.redis = redis
        self.source = source
        self.name = source.splitlines()[0].removeprefix("-- hayhooks:")
        self.calls = []
        self.fail_next = False

    async def __call__(self, *, keys=None, args=None, client=None):
        assert client is None
        self.calls.append((list(keys or []), list(args or [])))
        await asyncio.sleep(0)
        if self.fail_next:
            self.fail_next = False
            msg = "temporary Redis failure"
            raise ConnectionError(msg)
        return self.redis.results.get(self.name, 1)


class FakeRedis:
    def __init__(self):
        self.scripts = {}
        self.results = {}
        self.closed = False

    def register_script(self, source):
        script = FakeRegisteredScript(self, source)
        self.scripts[script.name] = script
        return script

    async def aclose(self):
        self.closed = True


class InitializingFakeRedis(FakeRedis):
    def __init__(self, version):
        super().__init__()
        self.version = version
        self.group_created = False

    async def ping(self):
        return True

    async def info(self, section):
        assert section == "server"
        return {"redis_version": self.version}

    async def config_get(self, *names):
        assert names == ("maxmemory-policy", "appendonly", "save")
        return {"maxmemory-policy": "noeviction", "appendonly": "yes", "save": ""}

    async def xgroup_create(self, *args, **kwargs):
        self.group_created = True


class DeliveryFakeRedis(FakeRedis):
    def __init__(self):
        super().__init__()
        self.xautoclaim_calls = []
        self.xreadgroup_calls = []
        self.claimed = [b"0-0", [], []]
        self.fail_next_xautoclaim = False

    async def xautoclaim(self, *args, **kwargs):
        self.xautoclaim_calls.append((args, kwargs))
        if self.fail_next_xautoclaim:
            self.fail_next_xautoclaim = False
            msg = "temporary reclaim failure"
            raise ConnectionError(msg)
        return self.claimed

    async def xreadgroup(self, *args, **kwargs):
        self.xreadgroup_calls.append((args, kwargs))
        return []


def _record(execution_id="execution"):
    return ExecutionRecord(
        execution_id=execution_id,
        execution_kind=ExecutionKind.PIPELINE,
        deployment_name="test",
        definition_revision="revision",
        validated_input={"value": 1},
    )


def test_redis_execution_scripts_are_packaged_and_registered():
    redis = FakeRedis()

    RedisExecutionStore(redis, key_prefix="test")

    assert set(redis.scripts) == {
        "acknowledge_delivery",
        "cancel",
        "checkpoint",
        "cleanup_expired_counts",
        "complete",
        "delay_delivery",
        "promote_delayed",
        "release_lease",
        "retire_incompatible",
        "renew_lease",
        "resume",
        "retry",
        "submit",
        "suspend",
    }
    assert all(script.source.strip() for script in redis.scripts.values())


@pytest.mark.asyncio
async def test_redis_execution_requires_server_6_2_or_newer():
    supported = InitializingFakeRedis(b"6.2.0")
    await RedisExecutionStore(supported).initialize()
    assert supported.group_created

    unsupported = InitializingFakeRedis("6.0.20")
    with pytest.raises(ExecutionStoreError, match="requires Redis server 6.2 or newer"):
        await RedisExecutionStore(unsupported).initialize()
    assert not unsupported.group_created


def test_redis_execution_provider_uses_app_settings(monkeypatch):
    monkeypatch.setattr(settings, "durable_redis_key_prefix", "configured:durable:")
    monkeypatch.setattr(settings, "durable_redis_claim_idle_ms", 45_000)
    monkeypatch.setattr(settings, "durable_redis_queue_block_ms", 2_500)
    monkeypatch.setattr(settings, "durable_redis_reclaim_interval", 2.5)
    monkeypatch.setattr(settings, "durable_terminal_ttl_seconds", 600)
    monkeypatch.setattr(settings, "durable_redis_cancellation_ttl_seconds", 120)
    monkeypatch.setattr(settings, "durable_redis_stream_max_length", 2_500)
    monkeypatch.setattr(settings, "durable_max_progress_events", 25)
    monkeypatch.setattr(settings, "durable_max_record_bytes", 200_000)
    monkeypatch.setattr(settings, "durable_redis_delayed_promotion_interval", 0.75)
    monkeypatch.setattr(settings, "durable_redis_delayed_promotion_batch_size", 250)

    provider = RedisExecutionStoreProvider(redis=FakeRedis(), close_redis=False)
    store = provider.create_execution_store("agent/name")

    assert store.key_prefix == "configured:durable:deployment:agent%2Fname"
    assert store.claim_idle_ms == 45_000
    assert store.queue_block_ms == 2_500
    assert store.reclaim_interval == 2.5
    assert store.terminal_ttl_seconds == 600
    assert store.cancellation_ttl_seconds == 120
    assert store.max_stream_length == 2_500
    assert store.max_progress_events == 25
    assert store.max_record_bytes == 200_000
    assert store.delayed_promotion_interval == 0.75
    assert store.delayed_promotion_batch_size == 250


@pytest.mark.asyncio
async def test_redis_execution_uses_registered_script_handles():
    redis = FakeRedis()
    store = RedisExecutionStore(redis, key_prefix="test")
    record = _record()

    assert await store.submit(record)

    keys, args = redis.scripts["submit"].calls[0]
    assert keys == ["test:execution:execution:record", "test:queue", "test:state-counts"]
    assert args == [record.to_json(), "execution"]


@pytest.mark.asyncio
async def test_delayed_promotion_is_throttled_and_retries_after_failure(monkeypatch):
    redis = FakeRedis()
    store = RedisExecutionStore(redis, key_prefix="test")
    clock = [100.0]
    monkeypatch.setattr("hayhooks.redis_execution.time.monotonic", lambda: clock[0])

    await asyncio.gather(*(store._promote_delayed() for _ in range(20)))
    assert len(redis.scripts["promote_delayed"].calls) == 1

    clock[0] += 0.1
    await store._promote_delayed()
    assert len(redis.scripts["promote_delayed"].calls) == 1

    clock[0] += 0.2
    redis.scripts["promote_delayed"].fail_next = True
    with pytest.raises(ExecutionStoreError, match="delayed promotion"):
        await store._promote_delayed()
    await store._promote_delayed()
    assert len(redis.scripts["promote_delayed"].calls) == 3


@pytest.mark.asyncio
async def test_idle_delivery_read_blocks_and_reclaim_scans_are_throttled(monkeypatch):
    redis = DeliveryFakeRedis()
    store = RedisExecutionStore(
        redis,
        key_prefix="test",
        queue_block_ms=1_250,
        reclaim_interval=1.0,
    )
    clock = [100.0]
    monkeypatch.setattr("hayhooks.redis_execution.time.monotonic", lambda: clock[0])

    assert await store._next_delivery("worker") is None
    clock[0] += 0.5
    assert await store._next_delivery("worker") is None
    clock[0] += 0.5
    assert await store._next_delivery("worker") is None

    assert len(redis.xautoclaim_calls) == 2
    assert len(redis.xreadgroup_calls) == 3
    assert all(call[1]["block"] == 1_250 for call in redis.xreadgroup_calls)


@pytest.mark.asyncio
async def test_reclaimed_delivery_skips_new_delivery_read():
    redis = DeliveryFakeRedis()
    redis.claimed = [b"42-0", [(b"41-0", {b"execution_id": b"recovered"})], []]
    store = RedisExecutionStore(redis, key_prefix="test")

    delivery = await store._next_delivery("worker")

    assert delivery is not None
    assert delivery.execution_id == "recovered"
    assert store._reclaim_cursors["worker"] == "42-0"
    assert redis.xreadgroup_calls == []


@pytest.mark.asyncio
async def test_failed_reclaim_scan_is_immediately_eligible_for_retry():
    redis = DeliveryFakeRedis()
    redis.fail_next_xautoclaim = True
    store = RedisExecutionStore(redis, key_prefix="test", reclaim_interval=10.0)

    with pytest.raises(ExecutionStoreError, match="delivery failed"):
        await store._next_delivery("worker")
    assert await store._next_delivery("worker") is None

    assert len(redis.xautoclaim_calls) == 2


@pytest.mark.asyncio
async def test_redis_execution_store_cleanup_only_closes_owned_client():
    shared_redis = FakeRedis()
    shared_store = RedisExecutionStore(shared_redis, close_redis=False)
    await shared_store.close()
    assert not shared_redis.closed

    owned_redis = FakeRedis()
    owned_store = RedisExecutionStore(owned_redis, close_redis=True)
    await owned_store.close()
    assert owned_redis.closed

    provider_redis = FakeRedis()
    provider = RedisExecutionStoreProvider(redis=provider_redis, close_redis=True)
    assert provider.create_execution_store("agent") is provider.create_execution_store("agent")
    await provider.close()
    assert provider_redis.closed
