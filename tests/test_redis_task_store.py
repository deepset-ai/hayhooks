from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from hayhooks.a2a import RedisTaskStore, RedisTaskStoreProvider, TaskStoreProvider
from hayhooks.server.a2a.imports import InvalidParamsError
from hayhooks.server.a2a.runtime import create_task_store_provider


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.sorted_sets = {}
        self.strings = {}
        self.closed = False
        self.hvals_calls = 0
        self.hgetall_calls = 0

    async def eval(self, script, numkeys, *args):  # noqa: C901, PLR0911, PLR0912, PLR0915
        keys = args[:numkeys]
        values = list(args[numkeys:])
        if "load-task" in script:
            payload = self.hashes.get(keys[0], {}).get(values[0])
            if payload is None:
                return None
            version = self.hashes.get(keys[1], {}).get(values[0], 0)
            return [payload, version]
        if "renew-projection" in script:
            return int(self.strings.get(keys[0]) == values[0])
        if "release-projection" in script:
            if self.strings.get(keys[0]) != values[0]:
                return 0
            self.strings.pop(keys[0], None)
            return 1

        tasks_key, task_owners_key, updates_key, active_key, terminal_key, versions_key = keys[:6]
        task_id, owner, *task_values = values
        existing_owner = self.hashes.get(task_owners_key, {}).get(task_id)

        if "save-task" in script:
            payload, score, terminal, expiry, expected_version = task_values
            if existing_owner is not None and existing_owner != owner:
                return 0
            current_version = self.hashes.setdefault(versions_key, {}).get(task_id, 0)
            if existing_owner is not None and (int(expected_version) < 0 or current_version != int(expected_version)):
                return -2
            self.hashes.setdefault(tasks_key, {})[task_id] = payload
            self.hashes.setdefault(task_owners_key, {})[task_id] = owner
            self.sorted_sets.setdefault(updates_key, {})[task_id] = float(score)
            version = current_version + 1
            self.hashes[versions_key][task_id] = version
            if terminal == "1":
                self.sorted_sets.setdefault(active_key, {}).pop(task_id, None)
                self.sorted_sets.setdefault(terminal_key, {})[task_id] = float(expiry)
            else:
                self.sorted_sets.setdefault(active_key, {})[task_id] = float(score)
                self.sorted_sets.setdefault(terminal_key, {}).pop(task_id, None)
            return version

        if "save-projection" in script:
            payload, score, terminal, expiry, token, expected_version = task_values
            if self.strings.get(keys[6]) != token:
                return -1
            version = self.hashes.setdefault(versions_key, {}).get(task_id, 0)
            if version != int(expected_version):
                return -2
            self.hashes.setdefault(tasks_key, {})[task_id] = payload
            self.sorted_sets.setdefault(updates_key, {})[task_id] = float(score)
            version += 1
            self.hashes[versions_key][task_id] = version
            if terminal == "1":
                self.sorted_sets.setdefault(active_key, {}).pop(task_id, None)
                self.sorted_sets.setdefault(terminal_key, {})[task_id] = float(expiry)
            else:
                self.sorted_sets.setdefault(active_key, {})[task_id] = float(score)
            return version

        if "delete-task" in script:
            if existing_owner != owner:
                return 0
            self.hashes.get(tasks_key, {}).pop(task_id, None)
            self.hashes.get(task_owners_key, {}).pop(task_id, None)
            self.sorted_sets.get(updates_key, {}).pop(task_id, None)
            self.sorted_sets.get(active_key, {}).pop(task_id, None)
            self.sorted_sets.get(terminal_key, {}).pop(task_id, None)
            self.hashes.get(versions_key, {}).pop(task_id, None)
            return 1

        msg = "Unexpected Redis script"
        raise AssertionError(msg)

    async def hget(self, name, key):
        return self.hashes.get(name, {}).get(key)

    async def set(self, name, value, *, nx=False, px=None):
        if nx and name in self.strings:
            return False
        self.strings[name] = value
        return True

    async def hvals(self, name):
        self.hvals_calls += 1
        return list(self.hashes.get(name, {}).values())

    async def hgetall(self, name):
        self.hgetall_calls += 1
        return dict(self.hashes.get(name, {}))

    async def hmget(self, name, keys):
        return [self.hashes.get(name, {}).get(key) for key in keys]

    async def zcard(self, name):
        return len(self.sorted_sets.get(name, {}))

    async def zrevrank(self, name, key):
        task_ids = sorted(
            self.sorted_sets.get(name, {}), key=lambda task_id: (self.sorted_sets[name][task_id], task_id), reverse=True
        )
        try:
            return task_ids.index(key)
        except ValueError:
            return None

    async def zrevrange(self, name, start, end):
        task_ids = sorted(
            self.sorted_sets.get(name, {}), key=lambda task_id: (self.sorted_sets[name][task_id], task_id), reverse=True
        )
        return task_ids[start : end + 1]

    async def zrange(self, name, start, end):
        task_ids = sorted(
            self.sorted_sets.get(name, {}), key=lambda task_id: (self.sorted_sets[name][task_id], task_id)
        )
        return task_ids[start : end + 1]

    async def zscan(self, name, cursor=0, *, count=None):
        entries = sorted(self.sorted_sets.get(name, {}).items())
        limit = count or len(entries)
        page = entries[cursor : cursor + limit]
        next_cursor = cursor + len(page)
        return (0 if next_cursor >= len(entries) else next_cursor), page

    async def zrangebyscore(self, name, minimum, maximum, *, start=0, num=None):
        lower = float("-inf") if minimum == "-inf" else float(minimum)
        upper = float(maximum)
        task_ids = [
            task_id
            for task_id, score in sorted(self.sorted_sets.get(name, {}).items(), key=lambda item: (item[1], item[0]))
            if lower <= score <= upper
        ]
        return task_ids[start : start + num if num is not None else None]

    async def zrem(self, name, task_id):
        return int(self.sorted_sets.get(name, {}).pop(task_id, None) is not None)

    async def aclose(self):
        self.closed = True


def _context(owner: str):
    return SimpleNamespace(user=SimpleNamespace(user_name=owner))


def _task(task_id: str, seconds: int):
    from a2a.types import Task, TaskState

    task = Task(id=task_id, context_id=f"context-{task_id}")
    task.status.state = TaskState.TASK_STATE_WORKING
    task.status.timestamp.FromDatetime(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds))
    return task


@pytest.mark.asyncio
async def test_redis_task_store_isolates_agents_and_owners_and_supports_recovery():
    from a2a.types import TaskState

    redis = FakeRedis()
    store = RedisTaskStore(redis, "agent/one", key_prefix="test:a2a")
    other_agent = RedisTaskStore(redis, "agent/two", key_prefix="test:a2a")
    owner = _context("alice@example.com")
    other_owner = _context("bob@example.com")
    task = _task("task-1", 1)

    await store.save(task, owner)

    assert (await store.get(task.id, owner)).id == task.id
    assert await store.get(task.id, other_owner) is None
    assert await other_agent.get(task.id, owner) is None
    assert (await store.get_for_execution(task.id)).id == task.id

    task.status.state = TaskState.TASK_STATE_COMPLETED
    await store.save_for_execution(task)
    assert (await store.get(task.id, owner)).status.state == TaskState.TASK_STATE_COMPLETED

    with pytest.raises(InvalidParamsError, match="belongs to another owner"):
        await store.save(_task(task.id, 2), other_owner)
    assert (await store.get(task.id, owner)).status.state == TaskState.TASK_STATE_COMPLETED

    await store.delete(task.id, owner)
    assert await store.get(task.id, owner) is None
    assert await store.get_for_execution(task.id) is None


@pytest.mark.asyncio
async def test_redis_task_store_lists_only_nonterminal_recovery_candidates():
    from a2a.types import TaskState

    redis = FakeRedis()
    store = RedisTaskStore(redis, "agent", key_prefix="test:a2a")
    owner = _context("alice@example.com")
    working = _task("working", 1)
    completed = _task("completed", 2)
    completed.status.state = TaskState.TASK_STATE_COMPLETED
    await store.save(working, owner)
    await store.save(completed, owner)

    recovered = await store.recoverable_tasks()

    assert [task.id for task in recovered] == ["working"]
    assert redis.hgetall_calls == 0


@pytest.mark.asyncio
async def test_projection_lease_and_version_fence_stale_replica():
    redis = FakeRedis()
    store = RedisTaskStore(redis, "agent", key_prefix="test:a2a")
    task = _task("task", 1)
    await store.save(task, _context("alice@example.com"))

    token = await store.acquire_projection(task.id, lease_ms=1_000)
    assert token is not None
    version = await store.projection_version(task.id)
    assert await store.save_projection(task, token, version) == version + 1
    assert await store.save_projection(task, token, version) == -2
    await store.release_projection(task.id, token)
    assert await store.save_projection(task, token, version + 1) == -1


@pytest.mark.asyncio
async def test_all_task_store_writes_reject_a_stale_loaded_version():
    redis = FakeRedis()
    first = RedisTaskStore(redis, "agent", key_prefix="test:a2a")
    second = RedisTaskStore(redis, "agent", key_prefix="test:a2a")
    context = _context("alice@example.com")
    task = _task("task", 1)
    await first.save(task, context)
    stale = await second.get(task.id, context)
    assert stale is not None

    task.status.timestamp.FromDatetime(datetime(2026, 1, 2, tzinfo=timezone.utc))
    await first.save(task, context)
    with pytest.raises(InvalidParamsError, match="stale projection version"):
        await second.save(stale, context)


@pytest.mark.asyncio
async def test_same_store_tracks_versions_per_loaded_task_snapshot():
    from a2a.types import TaskState

    redis = FakeRedis()
    store = RedisTaskStore(redis, "agent", key_prefix="test:a2a")
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


@pytest.mark.asyncio
async def test_redis_task_store_lists_with_filters_and_page_tokens():
    redis = FakeRedis()
    store = RedisTaskStore(redis, "agent", key_prefix="test:a2a")
    context = _context("alice@example.com")
    for index in range(3):
        await store.save(_task(f"task-{index}", index), context)

    from a2a.types import ListTasksRequest

    first_page = await store.list(ListTasksRequest(page_size=2), context)
    assert [task.id for task in first_page.tasks] == ["task-2", "task-1"]
    assert first_page.total_size == 3
    assert first_page.next_page_token
    assert redis.hvals_calls == 0

    second_page = await store.list(
        ListTasksRequest(page_size=2, page_token=first_page.next_page_token),
        context,
    )
    assert [task.id for task in second_page.tasks] == ["task-0"]
    assert second_page.next_page_token == ""

    filtered_page = await store.list(ListTasksRequest(context_id="context-task-2"), context)
    assert [task.id for task in filtered_page.tasks] == ["task-2"]
    assert redis.hvals_calls == 1

    from a2a.utils.errors import InvalidParamsError

    cursor = "invalid"
    with pytest.raises(InvalidParamsError, match="base64-encoded cursor"):
        await store.list(ListTasksRequest(page_token=cursor), context)


@pytest.mark.asyncio
async def test_redis_task_store_provider_is_cached_and_closes_redis():
    redis = FakeRedis()
    provider = RedisTaskStoreProvider(redis=redis, key_prefix="test:a2a")

    first = provider.create_task_store("agent")
    assert provider.create_task_store("agent") is first
    assert isinstance(provider, TaskStoreProvider)

    await provider.close()
    assert redis.closed


def test_redis_task_store_defaults_use_app_settings(monkeypatch):
    from hayhooks.settings import settings

    monkeypatch.setattr(settings, "a2a_redis_key_prefix", "configured:a2a:")
    monkeypatch.setattr(settings, "a2a_redis_socket_timeout", 3.5)
    monkeypatch.setattr(settings, "a2a_redis_socket_connect_timeout", 2.5)
    monkeypatch.setattr(settings, "a2a_redis_health_check_interval", 20)
    redis = FakeRedis()

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


def test_create_task_store_provider_rejects_ambiguous_configuration():
    with pytest.raises(ValueError, match="cannot be used together"):
        create_task_store_provider(backend="redis", custom_provider="my_project:Provider")


def test_a2a_app_rejects_ambiguous_task_store_configuration(monkeypatch):
    from hayhooks.server.a2a import a2a_import, create_a2a_app
    from hayhooks.settings import settings

    monkeypatch.setattr(a2a_import, "check", lambda: None)
    monkeypatch.setattr(settings, "a2a_task_store", "redis")
    monkeypatch.setattr(settings, "a2a_task_store_provider", "my_project:Provider")

    with pytest.raises(ValueError, match="cannot be used together"):
        create_a2a_app()
