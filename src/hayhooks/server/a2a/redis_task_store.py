"""Optional Redis-backed persistence for A2A tasks."""

from __future__ import annotations

import asyncio
import builtins
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from hayhooks.a2a import TaskStoreProvider
from hayhooks.server.a2a.imports import InvalidParamsError, TaskStore, a2a_import
from hayhooks.settings import settings

if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext
    from a2a.types import ListTasksRequest, ListTasksResponse, Task


OwnerResolver = Callable[[Any], str]
_STALE_VERSION = -2
_MAX_TRACKED_SNAPSHOTS_PER_TASK = 16

_LOAD_TASK_SCRIPT = """
-- load-task
local payload = redis.call("HGET", KEYS[1], ARGV[1])
if not payload then return false end
local version = tonumber(redis.call("HGET", KEYS[2], ARGV[1]) or "0")
return {payload, version}
"""

_SAVE_TASK_SCRIPT = """
-- save-task
local recorded_owner = redis.call("HGET", KEYS[2], ARGV[1])
if recorded_owner and recorded_owner ~= ARGV[2] then
    return 0
end
local current_version = tonumber(redis.call("HGET", KEYS[6], ARGV[1]) or "0")
local expected_version = tonumber(ARGV[7])
if recorded_owner and (expected_version < 0 or current_version ~= expected_version) then
    return -2
end
redis.call("HSET", KEYS[1], ARGV[1], ARGV[3])
redis.call("HSET", KEYS[2], ARGV[1], ARGV[2])
redis.call("ZADD", KEYS[3], ARGV[4], ARGV[1])
local version = redis.call("HINCRBY", KEYS[6], ARGV[1], 1)
if ARGV[5] == "1" then
    redis.call("ZREM", KEYS[4], ARGV[1])
    redis.call("ZADD", KEYS[5], ARGV[6], ARGV[1])
else
    redis.call("ZADD", KEYS[4], ARGV[4], ARGV[1])
    redis.call("ZREM", KEYS[5], ARGV[1])
end
return version
"""

_DELETE_TASK_SCRIPT = """
-- delete-task
local recorded_owner = redis.call("HGET", KEYS[2], ARGV[1])
if recorded_owner ~= ARGV[2] then
    return 0
end
redis.call("HDEL", KEYS[1], ARGV[1])
redis.call("HDEL", KEYS[2], ARGV[1])
redis.call("ZREM", KEYS[3], ARGV[1])
redis.call("ZREM", KEYS[4], ARGV[1])
redis.call("ZREM", KEYS[5], ARGV[1])
redis.call("HDEL", KEYS[6], ARGV[1])
return 1
"""

_SAVE_PROJECTION_SCRIPT = """
-- save-projection
if redis.call("GET", KEYS[7]) ~= ARGV[7] then return -1 end
local recorded_owner = redis.call("HGET", KEYS[2], ARGV[1])
if not recorded_owner or recorded_owner ~= ARGV[2] then return -1 end
local version = tonumber(redis.call("HGET", KEYS[6], ARGV[1]) or "0")
if version ~= tonumber(ARGV[8]) then return -2 end
redis.call("HSET", KEYS[1], ARGV[1], ARGV[3])
redis.call("ZADD", KEYS[3], ARGV[4], ARGV[1])
version = redis.call("HINCRBY", KEYS[6], ARGV[1], 1)
if ARGV[5] == "1" then
    redis.call("ZREM", KEYS[4], ARGV[1])
    redis.call("ZADD", KEYS[5], ARGV[6], ARGV[1])
else
    redis.call("ZADD", KEYS[4], ARGV[4], ARGV[1])
    redis.call("ZREM", KEYS[5], ARGV[1])
end
return version
"""

_RENEW_PROJECTION_SCRIPT = """
-- renew-projection
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("PEXPIRE", KEYS[1], ARGV[2])
end
return 0
"""

_RELEASE_PROJECTION_SCRIPT = """
-- release-projection
if redis.call("GET", KEYS[1]) == ARGV[1] then return redis.call("DEL", KEYS[1]) end
return 0
"""


def _default_owner_resolver(context: ServerCallContext) -> str:
    return context.user.user_name


class RedisTaskStore(TaskStore):
    """Persist A2A tasks in Redis with agent- and owner-scoped keys."""

    def __init__(
        self,
        redis: Any,
        agent_name: str,
        *,
        key_prefix: str | None = None,
        owner_resolver: OwnerResolver = _default_owner_resolver,
        terminal_ttl_seconds: int | None = None,
    ) -> None:
        self.redis = redis
        self.agent_name = agent_name
        self.key_prefix = (key_prefix or settings.a2a_redis_key_prefix).rstrip(":")
        self.owner_resolver = owner_resolver
        self.terminal_ttl_seconds = terminal_ttl_seconds or settings.a2a_terminal_task_ttl_seconds
        # Versions belong to the loaded protobuf snapshot, not merely its task
        # ID. Multiple requests can legitimately hold different snapshots.
        self._loaded_task_versions: dict[str, dict[int, tuple[Any, int]]] = {}

    def _remember_task_version(self, task: Task, version: int) -> None:
        snapshots = self._loaded_task_versions.setdefault(task.id, {})
        snapshots[id(task)] = (task, version)
        while len(snapshots) > _MAX_TRACKED_SNAPSHOTS_PER_TASK:
            snapshots.pop(next(iter(snapshots)))

    def _loaded_task_version(self, task: Task) -> int:
        snapshot = self._loaded_task_versions.get(task.id, {}).get(id(task))
        return snapshot[1] if snapshot is not None and snapshot[0] is task else -1

    def _forget_task_versions(self, task_id: str) -> None:
        self._loaded_task_versions.pop(task_id, None)

    def _agent_key(self) -> str:
        return quote(self.agent_name, safe="-_.")

    def _owner_key(self, owner: str) -> str:
        return quote(owner, safe="-_.@")

    def _tasks_key_for_owner(self, owner: str) -> str:
        return f"{self.key_prefix}:agent:{self._agent_key()}:owner:{self._owner_key(owner)}:tasks"

    def _task_index_key(self) -> str:
        return f"{self.key_prefix}:agent:{self._agent_key()}:task-owners"

    def _task_updates_key_for_owner(self, owner: str) -> str:
        return f"{self._tasks_key_for_owner(owner)}:updates"

    def _active_tasks_key(self) -> str:
        return f"{self.key_prefix}:agent:{self._agent_key()}:active"

    def _terminal_tasks_key(self) -> str:
        return f"{self.key_prefix}:agent:{self._agent_key()}:terminal-expiry"

    def _versions_key(self) -> str:
        return f"{self.key_prefix}:agent:{self._agent_key()}:versions"

    def _projection_lease_key(self, task_id: str) -> str:
        return f"{self.key_prefix}:agent:{self._agent_key()}:projection:{quote(task_id, safe='-_.')}:lease"

    def _tasks_key(self, context: ServerCallContext) -> str:
        return self._tasks_key_for_owner(self.owner_resolver(context))

    def _task_updates_key(self, context: ServerCallContext) -> str:
        return self._task_updates_key_for_owner(self.owner_resolver(context))

    @staticmethod
    def _deserialize(payload: bytes | str) -> Task:
        a2a_import.check()
        from a2a.types import Task

        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        task = Task()
        task.ParseFromString(payload)
        return task

    @staticmethod
    def _serialize(task: Task) -> bytes:
        return task.SerializeToString()

    @staticmethod
    def _decode_value(value: bytes | str) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    @staticmethod
    def _task_score(task: Task) -> float:
        if task.HasField("status") and task.status.HasField("timestamp"):
            return task.status.timestamp.ToNanoseconds() / 1_000_000_000
        return -1.0

    @staticmethod
    def _terminal(task: Task) -> bool:
        from a2a.types import TaskState

        return task.status.state in {
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_CANCELED,
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_REJECTED,
        }

    async def _save_payload(self, task: Task, owner: str) -> int:
        """Atomically save a task without allowing its owner to change."""
        await self.cleanup_expired_tasks(limit=10)
        result = await self.redis.eval(
            _SAVE_TASK_SCRIPT,
            6,
            self._tasks_key_for_owner(owner),
            self._task_index_key(),
            self._task_updates_key_for_owner(owner),
            self._active_tasks_key(),
            self._terminal_tasks_key(),
            self._versions_key(),
            task.id,
            owner,
            self._serialize(task),
            str(self._task_score(task)),
            "1" if self._terminal(task) else "0",
            str(time.time() + self.terminal_ttl_seconds),
            self._loaded_task_version(task),
        )
        version = int(result)
        if version > 0:
            self._remember_task_version(task, version)
        return version

    async def save(self, task: Task, context: ServerCallContext) -> None:
        owner = self.owner_resolver(context)
        saved = await self._save_payload(task, owner)
        if saved == _STALE_VERSION:
            msg = f"Task '{task.id}' has a stale projection version"
            raise InvalidParamsError(msg)
        if saved < 1:
            msg = f"Task '{task.id}' belongs to another owner"
            raise InvalidParamsError(msg)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        loaded = await self.redis.eval(
            _LOAD_TASK_SCRIPT,
            2,
            self._tasks_key(context),
            self._versions_key(),
            task_id,
        )
        if not loaded:
            return None
        payload, version = loaded
        task = self._deserialize(payload)
        self._remember_task_version(task, int(version))
        return task

    async def get_for_execution(self, task_id: str) -> Task | None:
        """Load a task for a recovered durable execution without request context."""
        owner = await self.redis.hget(self._task_index_key(), task_id)
        if owner is None:
            return None
        owner = self._decode_value(owner)
        loaded = await self.redis.eval(
            _LOAD_TASK_SCRIPT,
            2,
            self._tasks_key_for_owner(owner),
            self._versions_key(),
            task_id,
        )
        if not loaded:
            return None
        payload, version = loaded
        task = self._deserialize(payload)
        self._remember_task_version(task, int(version))
        return task

    async def save_for_execution(self, task: Task) -> None:
        """Persist a durable projection using the owner recorded at submission."""
        owner = await self.redis.hget(self._task_index_key(), task.id)
        if owner is None:
            return
        owner = self._decode_value(owner)
        saved = await self._save_payload(task, owner)
        if saved == _STALE_VERSION:
            msg = f"Task '{task.id}' has a stale projection version"
            raise RuntimeError(msg)
        if saved < 1:
            msg = f"Task '{task.id}' could not be saved for durable execution"
            raise RuntimeError(msg)

    async def projection_version(self, task_id: str) -> int:
        value = await self.redis.hget(self._versions_key(), task_id)
        return int(value or 0)

    async def acquire_projection(self, task_id: str, *, lease_ms: int) -> str | None:
        token = uuid.uuid4().hex
        acquired = await self.redis.set(self._projection_lease_key(task_id), token, nx=True, px=lease_ms)
        return token if acquired else None

    async def renew_projection(self, task_id: str, token: str, *, lease_ms: int) -> bool:
        return bool(
            await self.redis.eval(
                _RENEW_PROJECTION_SCRIPT,
                1,
                self._projection_lease_key(task_id),
                token,
                lease_ms,
            )
        )

    async def release_projection(self, task_id: str, token: str) -> None:
        await self.redis.eval(
            _RELEASE_PROJECTION_SCRIPT,
            1,
            self._projection_lease_key(task_id),
            token,
        )

    async def save_projection(self, task: Task, token: str, expected_version: int) -> int:
        owner = await self.redis.hget(self._task_index_key(), task.id)
        if owner is None:
            return -1
        owner = self._decode_value(owner)
        version = int(
            await self.redis.eval(
                _SAVE_PROJECTION_SCRIPT,
                7,
                self._tasks_key_for_owner(owner),
                self._task_index_key(),
                self._task_updates_key_for_owner(owner),
                self._active_tasks_key(),
                self._terminal_tasks_key(),
                self._versions_key(),
                self._projection_lease_key(task.id),
                task.id,
                owner,
                self._serialize(task),
                str(self._task_score(task)),
                "1" if self._terminal(task) else "0",
                str(time.time() + self.terminal_ttl_seconds),
                token,
                expected_version,
            )
        )
        if version > 0:
            self._remember_task_version(task, version)
        return version

    async def recoverable_task_batch(self, cursor: int, limit: int) -> tuple[builtins.list[Task], int | None]:
        """Return one active-task page with owner payloads loaded in batches."""
        await self.cleanup_expired_tasks(limit=limit)
        next_cursor, entries = await self.redis.zscan(self._active_tasks_key(), cursor=cursor, count=limit)
        task_ids = [self._decode_value(raw_task_id) for raw_task_id, _score in entries]
        if not task_ids:
            return [], int(next_cursor) or None

        owners = await self.redis.hmget(self._task_index_key(), task_ids)
        owner_groups: dict[str, builtins.list[str]] = {}
        for task_id, raw_owner in zip(task_ids, owners, strict=True):
            if raw_owner is not None:
                owner_groups.setdefault(self._decode_value(raw_owner), []).append(task_id)

        grouped = list(owner_groups.items())
        results = await asyncio.gather(
            *(self.redis.hmget(self._tasks_key_for_owner(owner), owner_task_ids) for owner, owner_task_ids in grouped),
            self.redis.hmget(self._versions_key(), task_ids),
        )

        versions = dict(zip(task_ids, results[-1], strict=True))
        payloads: dict[str, bytes | str] = {}
        for (_owner, owner_task_ids), owner_payloads in zip(grouped, results[:-1], strict=True):
            payloads.update(
                {
                    task_id: payload
                    for task_id, payload in zip(owner_task_ids, owner_payloads, strict=True)
                    if payload is not None
                }
            )

        tasks: builtins.list[Task] = []
        for task_id in task_ids:
            payload = payloads.get(task_id)
            if payload is None:
                continue
            task = self._deserialize(payload)
            self._remember_task_version(task, int(versions.get(task_id) or 0))
            tasks.append(task)
        return tasks, int(next_cursor) or None

    async def cleanup_expired_tasks(self, *, limit: int = 100) -> int:
        expired = await self.redis.zrangebyscore(
            self._terminal_tasks_key(),
            "-inf",
            time.time(),
            start=0,
            num=limit,
        )
        removed = 0
        for raw_task_id in expired:
            task_id = self._decode_value(raw_task_id)
            owner = await self.redis.hget(self._task_index_key(), task_id)
            if owner is None:
                await self.redis.zrem(self._terminal_tasks_key(), task_id)
                continue
            owner = self._decode_value(owner)
            removed += int(
                await self.redis.eval(
                    _DELETE_TASK_SCRIPT,
                    6,
                    self._tasks_key_for_owner(owner),
                    self._task_index_key(),
                    self._task_updates_key_for_owner(owner),
                    self._active_tasks_key(),
                    self._terminal_tasks_key(),
                    self._versions_key(),
                    task_id,
                    owner,
                )
            )
            self._forget_task_versions(task_id)
        return removed

    async def recoverable_tasks(self) -> builtins.list[Task]:
        """Load nonterminal tasks so execution projections resume after restart."""
        tasks: builtins.list[Task] = []
        offset: int | None = 0
        while offset is not None:
            batch, offset = await self.recoverable_task_batch(offset, settings.a2a_projection_batch_size)
            tasks.extend(batch)
        return tasks

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        a2a_import.check()
        await self.cleanup_expired_tasks(limit=100)
        from a2a.types import ListTasksResponse
        from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
        from a2a.utils.task import decode_page_token, encode_page_token

        page_size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
        if self._has_list_filters(params):
            page, total_size, next_page_token = await self._list_filtered(
                params,
                context,
                page_size,
                decode_page_token,
                encode_page_token,
            )
        else:
            page, total_size, next_page_token = await self._list_by_recent_update(
                params,
                context,
                page_size,
                decode_page_token,
                encode_page_token,
            )
        return ListTasksResponse(
            tasks=page,
            next_page_token=next_page_token,
            page_size=page_size,
            total_size=total_size,
        )

    @staticmethod
    def _has_list_filters(params: ListTasksRequest) -> bool:
        return bool(params.context_id or params.status or params.HasField("status_timestamp_after"))

    async def _list_by_recent_update(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
        page_size: int,
        decode_page_token: Callable[[str], str],
        encode_page_token: Callable[[str], str],
    ) -> tuple[builtins.list[Task], int, str | None]:
        updates_key = self._task_updates_key(context)
        total_size = await self.redis.zcard(updates_key)
        start_index = 0
        if params.page_token:
            start_task_id = decode_page_token(params.page_token)
            rank = await self.redis.zrevrank(updates_key, start_task_id)
            if rank is None:
                msg = f"Invalid page token: {params.page_token}"
                raise InvalidParamsError(msg)
            start_index = rank + 1

        task_ids = await self.redis.zrevrange(updates_key, start_index, start_index + page_size - 1)
        task_ids = [self._decode_value(task_id) for task_id in task_ids]
        payloads = await self.redis.hmget(self._tasks_key(context), task_ids) if task_ids else []
        page = [self._deserialize(payload) for payload in payloads if payload is not None]
        has_next_page = start_index + len(task_ids) < total_size
        next_page_token = encode_page_token(task_ids[-1]) if task_ids and has_next_page else None
        return page, total_size, next_page_token

    async def _list_filtered(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
        page_size: int,
        decode_page_token: Callable[[str], str],
        encode_page_token: Callable[[str], str],
    ) -> tuple[builtins.list[Task], int, str | None]:
        """
        Apply SDK filters by scanning this owner's tasks.

        The common unfiltered path is indexed by update time. Adding mutable
        secondary indexes for every A2A filter would make every task update
        more expensive and complicate protobuf ownership writes.
        """
        payloads = await self.redis.hvals(self._tasks_key(context))
        tasks = [self._deserialize(payload) for payload in payloads]
        if params.context_id:
            tasks = [task for task in tasks if task.context_id == params.context_id]
        if params.status:
            tasks = [task for task in tasks if task.status.state == params.status]
        if params.HasField("status_timestamp_after"):
            timestamp_after = params.status_timestamp_after.ToJsonString()
            tasks = [
                task
                for task in tasks
                if task.HasField("status")
                and task.status.HasField("timestamp")
                and task.status.timestamp.ToJsonString() >= timestamp_after
            ]

        tasks.sort(key=self._task_sort_key, reverse=True)
        total_size = len(tasks)
        start_index = 0
        if params.page_token:
            start_task_id = decode_page_token(params.page_token)
            try:
                start_index = next(index + 1 for index, task in enumerate(tasks) if task.id == start_task_id)
            except StopIteration as error:
                msg = f"Invalid page token: {params.page_token}"
                raise InvalidParamsError(msg) from error

        page = tasks[start_index : start_index + page_size]
        has_next_page = start_index + page_size < total_size
        next_page_token = encode_page_token(page[-1].id) if page and has_next_page else None
        return page, total_size, next_page_token

    @staticmethod
    def _task_sort_key(task: Task) -> tuple[bool, str, str]:
        has_timestamp = task.HasField("status") and task.status.HasField("timestamp")
        timestamp = task.status.timestamp.ToJsonString() if has_timestamp else ""
        return has_timestamp, timestamp, task.id

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        owner = self.owner_resolver(context)
        await self.redis.eval(
            _DELETE_TASK_SCRIPT,
            6,
            self._tasks_key_for_owner(owner),
            self._task_index_key(),
            self._task_updates_key_for_owner(owner),
            self._active_tasks_key(),
            self._terminal_tasks_key(),
            self._versions_key(),
            task_id,
            owner,
        )
        self._forget_task_versions(task_id)


class RedisTaskStoreProvider(TaskStoreProvider):
    """
    Create one Redis task store per exposed agent.

    The provider intentionally does not import ``redis`` until it needs to
    create a client, so importing Hayhooks remains possible without the A2A
    optional dependencies installed.
    """

    def __init__(  # noqa: PLR0913 - provider options mirror Redis connection settings
        self,
        redis_url: str | None = None,
        *,
        key_prefix: str | None = None,
        redis: Any | None = None,
        owner_resolver: OwnerResolver = _default_owner_resolver,
        close_redis: bool = True,
        terminal_ttl_seconds: int | None = None,
        socket_timeout: float | None = None,
        socket_connect_timeout: float | None = None,
        health_check_interval: int | None = None,
    ) -> None:
        redis_url = redis_url or settings.a2a_redis_url
        key_prefix = key_prefix or settings.a2a_redis_key_prefix
        self.key_prefix = key_prefix.rstrip(":")
        self.owner_resolver = owner_resolver
        self.stores: dict[str, RedisTaskStore] = {}
        self._close_redis = close_redis
        self.terminal_ttl_seconds = terminal_ttl_seconds or settings.a2a_terminal_task_ttl_seconds
        self.socket_timeout = socket_timeout if socket_timeout is not None else settings.a2a_redis_socket_timeout
        self.socket_connect_timeout = (
            socket_connect_timeout if socket_connect_timeout is not None else settings.a2a_redis_socket_connect_timeout
        )
        self.health_check_interval = (
            health_check_interval if health_check_interval is not None else settings.a2a_redis_health_check_interval
        )
        if redis is None:
            try:
                from redis.asyncio import Redis
            except ImportError as error:  # pragma: no cover - depends on optional extras
                msg = 'Redis task storage requires the A2A extra. Install with `pip install "hayhooks[a2a]"`.'
                raise ImportError(msg) from error
            redis = Redis.from_url(
                redis_url,
                decode_responses=False,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                health_check_interval=self.health_check_interval,
            )
        self.redis = redis

    def create_task_store(self, agent_name: str) -> RedisTaskStore:
        if agent_name not in self.stores:
            self.stores[agent_name] = RedisTaskStore(
                self.redis,
                agent_name,
                key_prefix=self.key_prefix,
                owner_resolver=self.owner_resolver,
                terminal_ttl_seconds=self.terminal_ttl_seconds,
            )
        return self.stores[agent_name]

    async def close(self) -> None:
        if self._close_redis:
            await self.redis.aclose()


__all__ = ["RedisTaskStore", "RedisTaskStoreProvider"]
