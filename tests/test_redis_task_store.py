import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from redis.cluster import key_slot

from hayhooks.a2a import RedisTaskStore, RedisTaskStoreProvider, TaskStoreProvider
from hayhooks.durable.backend import ExecutionStoreCorruptionError
from hayhooks.durable.runtime import execution_id_for
from hayhooks.server.a2a.durable_executor import DurableTaskStore
from hayhooks.server.a2a.runtime import create_task_store_provider


def _context(owner: str):
    return SimpleNamespace(user=SimpleNamespace(is_authenticated=True, user_name=owner))


def test_default_owner_distinguishes_unauthenticated_and_anonymous_user():
    store = RedisTaskStore(AsyncMock(), "agent", key_prefix="test:a2a")
    unauthenticated = SimpleNamespace(user=SimpleNamespace(is_authenticated=False, user_name=""))

    assert store.owner_id_for_context(unauthenticated) == "anonymous"
    assert store.owner_id_for_context(_context("anonymous")) == "user:anonymous"


def test_durable_task_store_delegates_custom_owner_resolution():
    store = RedisTaskStore(AsyncMock(), "agent", key_prefix="test:a2a", owner_resolver=lambda _context: "tenant:alice")

    assert DurableTaskStore(store, object()).owner_id_for_context(_context("ignored")) == "tenant:alice"


def test_execution_id_for_accepts_arbitrary_a2a_task_ids():
    owner = "tenant:alice"
    task_ids = ["", "client.task", "folder/task", "λ" * 200]
    execution_ids = [execution_id_for(owner, task_id) for task_id in task_ids]

    assert all(len(execution_id) == 64 and execution_id.isalnum() for execution_id in execution_ids)
    assert len(set(execution_ids)) == len(task_ids)
    assert execution_id_for("tenant:bob", "client.task") != execution_id_for(owner, "client.task")


def test_redis_task_keys_share_one_cluster_slot_and_store_uses_no_eval():
    store = RedisTaskStore(AsyncMock(), "agent", key_prefix="test:a2a")
    keys = (
        store._task_key("task"),
        store._updates_key("tenant:alice"),
        store._key("active"),
        store._key("terminal-expiry"),
    )

    assert len({key_slot(key.encode()) for key in keys}) == 1
    assert ".eval(" not in inspect.getsource(RedisTaskStore)


def test_redis_task_store_normalizes_corrupt_protobuf_snapshots():
    with pytest.raises(ExecutionStoreCorruptionError, match="invalid Redis snapshot"):
        RedisTaskStore._decode_snapshot(
            "task",
            {
                b"owner": b"owner",
                b"payload": b"not-a-protobuf",
                b"version": b"1",
                b"terminal_expiry_ms": b"0",
            },
        )


async def test_redis_task_store_provider_health_pings_redis():
    redis = AsyncMock()
    provider = RedisTaskStoreProvider(redis=redis, close_redis=False)

    await provider.initialize()
    assert (await provider.health())["healthy"]

    redis.ping.side_effect = ConnectionError("offline")
    health = await provider.health()
    assert not health["healthy"]
    assert health["error"] == "ConnectionError"


async def test_redis_task_store_provider_is_cached_and_closes_redis():
    redis = AsyncMock()
    provider = RedisTaskStoreProvider(redis=redis, key_prefix="test:a2a")

    first = provider.create_task_store("agent")
    assert provider.create_task_store("agent") is first
    assert isinstance(provider, TaskStoreProvider)

    await provider.close()
    redis.aclose.assert_awaited_once()


def test_redis_task_store_defaults_use_app_settings(monkeypatch):
    from hayhooks.settings import settings

    monkeypatch.setattr(settings, "a2a_redis_key_prefix", "configured:a2a:")
    monkeypatch.setattr(settings, "a2a_redis_socket_timeout", 3.5)
    monkeypatch.setattr(settings, "a2a_redis_socket_connect_timeout", 2.5)
    monkeypatch.setattr(settings, "a2a_redis_health_check_interval", 20)
    redis = AsyncMock()

    direct_store = RedisTaskStore(redis, "direct")
    provider_store = RedisTaskStoreProvider(redis=redis, close_redis=False).create_task_store("provided")

    assert direct_store.key_prefix == "configured:a2a"
    assert provider_store.key_prefix == "configured:a2a"

    provider = RedisTaskStoreProvider(redis=redis, close_redis=False)
    assert provider.socket_timeout == 3.5
    assert provider.socket_connect_timeout == 2.5
    assert provider.health_check_interval == 20


def test_create_task_store_provider_selects_builtin_backends():
    memory = create_task_store_provider()
    assert type(memory).__name__ == "InMemoryTaskStoreProvider"

    redis = create_task_store_provider(backend="redis", redis_url="redis://localhost:6379/2")
    assert isinstance(redis, RedisTaskStoreProvider)
