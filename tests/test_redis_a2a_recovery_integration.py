"""Real-Redis regression coverage for durable A2A task recovery."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from a2a.server.context import ServerCallContext
from a2a.types import ListTasksRequest, Task, TaskState

from hayhooks.durable.models import ExecutionStatus
from hayhooks.durable.runtime import execution_id_for
from hayhooks.server.a2a.durable_executor import DurableAgentExecutor
from hayhooks.server.a2a.imports import InvalidParamsError
from hayhooks.server.a2a.redis_task_store import RedisTaskStore

pytestmark = pytest.mark.integration


class _CompletedDeployment:
    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id
        self.record = SimpleNamespace(
            status=ExecutionStatus.COMPLETED,
            progress=[],
            result={"last_message": {"content": "recovered"}},
            error=None,
            sequence=1,
        )

    async def get(self, execution_id: str, **_kwargs):
        if execution_id != self.execution_id:
            raise KeyError(execution_id)
        return self.record


@pytest.fixture
async def redis_task_store(isolated_redis):
    redis, prefix = isolated_redis
    store = RedisTaskStore(redis, "agent", key_prefix=f"{prefix}:a2a")
    yield redis, store


def _task(task_id: str, seconds: int = 0):
    task = Task(id=task_id, context_id=f"context-{task_id}")
    task.status.state = TaskState.TASK_STATE_WORKING
    task.status.timestamp.FromDatetime(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds))
    return task


def _context(owner: str):
    return SimpleNamespace(user=SimpleNamespace(is_authenticated=True, user_name=owner))


async def test_redis_a2a_recovery_uses_atomic_projection_and_owner(redis_task_store) -> None:
    redis, store = redis_task_store
    context = ServerCallContext()
    task = _task("client.task/" + "x" * 160)
    execution_id = execution_id_for("anonymous", task.id)
    await store.save(task, context)

    # A replacement executor sees the task through Redis, not a live request queue.
    first = DurableAgentExecutor("agent", store, _CompletedDeployment(execution_id))
    await first.close()
    second = DurableAgentExecutor("agent", store, _CompletedDeployment(execution_id))
    await second.start()

    recovered = await store.get(task.id, context)
    assert recovered is not None and recovered.status.state == TaskState.TASK_STATE_COMPLETED
    version = store._loaded_task_version(recovered)
    assert version == 2

    stale = type(recovered)()
    stale.CopyFrom(recovered)
    assert not await store.save_projection(stale, "anonymous", version - 1)

    await store.delete(task.id, context)
    assert await store.get(task.id, context) is None
    assert await redis.zscore(store._key("active"), task.id) is None


async def test_redis_a2a_read_through_persists_late_completion(redis_task_store) -> None:
    redis, store = redis_task_store
    context = ServerCallContext()
    task = _task("late-completion")
    execution_id = execution_id_for("anonymous", task.id)
    deployment = _CompletedDeployment(execution_id)
    deployment.record.status = ExecutionStatus.RUNNING
    await store.save(task, context)

    executor = DurableAgentExecutor("agent", store, deployment)
    await executor.start()
    deployment.record.status = ExecutionStatus.COMPLETED
    completed = await executor.task_store.get(task.id, context)
    persisted = await store.get(task.id, context)

    assert completed.status.state == TaskState.TASK_STATE_COMPLETED
    assert persisted.status.state == TaskState.TASK_STATE_COMPLETED
    assert await redis.zscore(store._key("active"), task.id) is None
    assert await redis.zscore(store._key("terminal-expiry"), task.id) is not None


async def test_redis_task_store_isolates_agents_and_owners(redis_task_store) -> None:
    redis, base_store = redis_task_store
    store = RedisTaskStore(redis, "agent/one", key_prefix=base_store.key_prefix)
    other_agent = RedisTaskStore(redis, "agent/two", key_prefix=base_store.key_prefix)
    owner = _context("alice@example.com")
    other_owner = _context("bob@example.com")
    task = _task("task-1", 1)
    await store.save(task, owner)

    assert (await store.get(task.id, owner)).id == task.id
    recoverable, _ = await store.recoverable_task_batch(0, 1)
    assert recoverable[0][1:] == ("user:alice@example.com", 1)
    assert await store.get(task.id, other_owner) is None
    assert await other_agent.get(task.id, owner) is None
    task.status.state = TaskState.TASK_STATE_COMPLETED
    await store.save(task, owner)
    assert (await store.get(task.id, owner)).status.state == TaskState.TASK_STATE_COMPLETED

    with pytest.raises(InvalidParamsError, match="belongs to another owner"):
        await store.save(_task(task.id, 2), other_owner)
    assert (await store.get(task.id, owner)).status.state == TaskState.TASK_STATE_COMPLETED

    await store.delete(task.id, owner)
    assert await store.get(task.id, owner) is None


async def test_redis_task_store_uses_one_owner_resolver_for_all_paths(redis_task_store) -> None:
    redis, base_store = redis_task_store
    store = RedisTaskStore(
        redis,
        "custom-owner-agent",
        key_prefix=base_store.key_prefix,
        owner_resolver=lambda _context: "tenant:alice",
    )
    context = _context("ignored")
    task = _task("client.task/with unicode-€", 1)

    await store.save(task, context)

    assert await store.get(task.id, context) is not None
    recoverable, _ = await store.recoverable_task_batch(0, 1)
    assert recoverable[0][1] == "tenant:alice"


async def test_all_task_store_writes_reject_a_stale_loaded_version(redis_task_store) -> None:
    redis, base_store = redis_task_store
    first = RedisTaskStore(redis, "stale-writes", key_prefix=base_store.key_prefix)
    second = RedisTaskStore(redis, "stale-writes", key_prefix=base_store.key_prefix)
    context = _context("alice@example.com")
    task = _task("task", 1)
    await first.save(task, context)
    stale = await second.get(task.id, context)
    assert stale is not None

    task.status.timestamp.FromDatetime(datetime(2026, 1, 2, tzinfo=timezone.utc))
    await first.save(task, context)
    with pytest.raises(InvalidParamsError, match="stale projection version"):
        await second.save(stale, context)


async def test_same_store_tracks_versions_per_loaded_task_snapshot(redis_task_store) -> None:
    _, store = redis_task_store
    context = _context("alice@example.com")
    await store.save(_task("task", 1), context)
    first = await store.get("task", context)
    stale = await store.get("task", context)
    assert first is not None
    assert stale is not None

    first.status.state = TaskState.TASK_STATE_COMPLETED
    await store.save(first, context)
    stale.status.state = TaskState.TASK_STATE_FAILED
    with pytest.raises(InvalidParamsError, match="stale projection version"):
        await store.save(stale, context)

    persisted = await store.get("task", context)
    assert persisted is not None
    assert persisted.status.state == TaskState.TASK_STATE_COMPLETED


async def test_concurrent_projection_writers_use_one_version_fence(redis_task_store) -> None:
    redis, base_store = redis_task_store
    first = RedisTaskStore(redis, "projection-race", key_prefix=base_store.key_prefix)
    second = RedisTaskStore(redis, "projection-race", key_prefix=base_store.key_prefix)
    context = _context("alice@example.com")
    owner = "user:alice@example.com"
    await first.save(_task("task", 1), context)
    left = await first.get("task", context)
    right = await second.get("task", context)
    assert left is not None and right is not None
    left.metadata["winner"] = "left"
    right.metadata["winner"] = "right"

    results = await asyncio.gather(
        first.save_projection(left, owner, 1),
        second.save_projection(right, owner, 1),
    )

    assert sorted(results) == [False, True]
    persisted = await first.get("task", context)
    assert persisted is not None and persisted.metadata["winner"] in {"left", "right"}


async def test_cleanup_preserves_a_task_whose_terminal_ttl_was_extended(monkeypatch, redis_task_store) -> None:
    redis, base_store = redis_task_store
    cleanup = RedisTaskStore(redis, "cleanup-race", key_prefix=base_store.key_prefix, terminal_ttl_seconds=60)
    writer = RedisTaskStore(redis, "cleanup-race", key_prefix=base_store.key_prefix, terminal_ttl_seconds=60)
    context = _context("alice@example.com")
    task = _task("task", 1)
    task.status.state = TaskState.TASK_STATE_COMPLETED
    await cleanup.save(task, context)
    current = await writer.get(task.id, context)
    assert current is not None
    await redis.zadd(cleanup._key("terminal-expiry"), {task.id: 0})

    candidate_selected = asyncio.Event()
    allow_delete = asyncio.Event()
    delete_payload = cleanup._delete_payload

    async def delayed_delete(*args, **kwargs):
        candidate_selected.set()
        await allow_delete.wait()
        return await delete_payload(*args, **kwargs)

    monkeypatch.setattr(cleanup, "_delete_payload", delayed_delete)
    cleanup_call = asyncio.create_task(cleanup.cleanup_expired_tasks())
    await candidate_selected.wait()
    current.metadata["updated"] = "yes"
    await writer.save(current, context)
    allow_delete.set()

    assert await cleanup_call == 0
    persisted = await cleanup.get(task.id, context)
    assert persisted is not None and persisted.metadata["updated"] == "yes"
    assert await redis.zscore(cleanup._key("terminal-expiry"), task.id) is not None
    await redis.hset(cleanup._task_key(task.id), "terminal_expiry_ms", 1)
    await redis.zadd(cleanup._key("terminal-expiry"), {task.id: 1})
    assert await cleanup.cleanup_expired_tasks() == 1
    assert await cleanup.get(task.id, context) is None
    assert await redis.zscore(cleanup._updates_key("user:alice@example.com"), task.id) is None


async def test_redis_task_store_lists_with_filters_and_page_tokens(redis_task_store) -> None:
    _, store = redis_task_store
    context = _context("alice@example.com")
    for index in range(3):
        await store.save(_task(f"task-{index}", index), context)

    first_page = await store.list(ListTasksRequest(page_size=2), context)
    assert [task.id for task in first_page.tasks] == ["task-2", "task-1"]
    assert first_page.total_size == 3
    assert first_page.next_page_token

    second_page = await store.list(
        ListTasksRequest(page_size=2, page_token=first_page.next_page_token),
        context,
    )
    assert [task.id for task in second_page.tasks] == ["task-0"]
    assert second_page.next_page_token == ""

    filtered_page = await store.list(ListTasksRequest(context_id="context-task-2"), context)
    assert [task.id for task in filtered_page.tasks] == ["task-2"]

    with pytest.raises(InvalidParamsError, match="base64-encoded cursor"):
        await store.list(ListTasksRequest(page_token="invalid"), context)  # noqa: S106


async def test_redis_task_store_compares_status_timestamps_as_timestamps(redis_task_store) -> None:
    _, store = redis_task_store
    context = _context("alice@example.com")
    task = _task("task")
    task.status.timestamp.FromDatetime(datetime(2026, 1, 1, 0, 0, 0, 100_000, tzinfo=timezone.utc))
    await store.save(task, context)

    request = ListTasksRequest()
    request.status_timestamp_after.FromDatetime(datetime(2026, 1, 1, tzinfo=timezone.utc))
    page = await store.list(request, context)

    assert [item.id for item in page.tasks] == ["task"]


async def test_filtered_task_listing_and_snapshot_cache_are_globally_bounded(monkeypatch, redis_task_store) -> None:
    from hayhooks.settings import settings

    monkeypatch.setattr(settings, "a2a_list_scan_batch_size", 2)
    monkeypatch.setattr(settings, "a2a_task_snapshot_cache_size", 3)
    redis, store = redis_task_store
    load_snapshots = AsyncMock(side_effect=store._load_snapshots)
    monkeypatch.setattr(store, "_load_snapshots", load_snapshots)
    monkeypatch.setattr(redis, "hvals", AsyncMock(side_effect=AssertionError("unbounded task scan")))
    context = _context("alice@example.com")
    for index in range(6):
        await store.save(_task(f"task-{index}", index), context)
        await store.get(f"task-{index}", context)

    page = await store.list(ListTasksRequest(context_id="context-task-5"), context)

    assert [task.id for task in page.tasks] == ["task-5"]
    assert all(len(call.args[0]) <= 2 for call in load_snapshots.await_args_list)
    assert len(store._loaded_task_versions) <= 3
