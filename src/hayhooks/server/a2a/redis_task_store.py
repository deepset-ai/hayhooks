"""Optional Redis-backed persistence for A2A tasks."""

from __future__ import annotations

import builtins
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from google.protobuf.message import DecodeError

from hayhooks.a2a import TaskStoreProvider, default_a2a_owner, validate_a2a_owner
from hayhooks.durable.backend import (
    DEFAULT_TRANSACTION_MAX_RETRIES,
    ExecutionContentionError,
    ExecutionStoreCorruptionError,
)
from hayhooks.durable.redis import digest, redis_time_ms, redis_transaction_backoff, redis_watch_error
from hayhooks.server.a2a.imports import (
    DEFAULT_LIST_TASKS_PAGE_SIZE,
    InvalidParamsError,
    ListTasksRequest,
    ListTasksResponse,
    Task,
    TaskStore,
    decode_page_token,
    encode_page_token,
)
from hayhooks.server.a2a.messages import task_is_terminal, task_matches_filters
from hayhooks.settings import settings

if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext


OwnerResolver = Callable[[Any], str]
_STALE_VERSION = -2
_OWNER_MISMATCH = 0


_default_owner_resolver = default_a2a_owner


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
        self.key_prefix = (key_prefix or settings.a2a_redis_key_prefix).rstrip(":")
        self._base_key = f"{self.key_prefix}:v2:{{{digest('a2a-agent', agent_name)}}}"
        self.owner_resolver = owner_resolver
        self.terminal_ttl_seconds = terminal_ttl_seconds or settings.a2a_terminal_task_ttl_seconds
        # Versions belong to the loaded protobuf snapshot, not merely its task
        # ID. Multiple requests can legitimately hold different snapshots. A
        # global LRU prevents one long-lived process from retaining every task
        # object it has ever loaded.
        self._loaded_task_versions: OrderedDict[tuple[str, int], tuple[Any, int]] = OrderedDict()

    def _remember_task_version(self, task: Task, version: int) -> None:
        key = (task.id, id(task))
        self._loaded_task_versions[key] = (task, version)
        self._loaded_task_versions.move_to_end(key)
        while len(self._loaded_task_versions) > settings.a2a_task_snapshot_cache_size:
            self._loaded_task_versions.popitem(last=False)

    def _loaded_task_version(self, task: Task) -> int:
        key = (task.id, id(task))
        snapshot = self._loaded_task_versions.get(key)
        if snapshot is None or snapshot[0] is not task:
            return -1
        self._loaded_task_versions.move_to_end(key)
        return snapshot[1]

    def copy_task_version(self, source: Task, target: Task) -> None:
        """Preserve optimistic-write state when an adapter clones a loaded task."""
        version = self._loaded_task_version(source)
        if version >= 0:
            self._remember_task_version(target, version)

    def _forget_task_versions(self, task_id: str) -> None:
        for key in [key for key in self._loaded_task_versions if key[0] == task_id]:
            self._loaded_task_versions.pop(key, None)

    def _key(self, suffix: str) -> str:
        return f"{self._base_key}:{suffix}"

    def _task_key(self, task_id: str) -> str:
        return self._key(f"task:{digest('a2a-task', task_id)}")

    def _updates_key(self, owner: str) -> str:
        return self._key(f"owner:{digest('a2a-owner', owner)}:updates")

    def owner_id_for_context(self, context: ServerCallContext) -> str:
        """Resolve ownership once for every Redis-backed A2A operation."""
        return validate_a2a_owner(self.owner_resolver(context))

    @staticmethod
    def _deserialize(payload: bytes | str) -> Task:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        task = Task()
        task.ParseFromString(payload)
        return task

    @staticmethod
    def _serialize(task: Task) -> bytes:
        return task.SerializeToString()

    @staticmethod
    def _decode_value(value: bytes | str | int) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @classmethod
    def _decode_snapshot(cls, task_id: str, values: Mapping[Any, Any]) -> tuple[Task, str, int, int]:
        try:
            snapshot = {cls._decode_value(key): value for key, value in values.items()}
            owner = validate_a2a_owner(cls._decode_value(snapshot["owner"]))
            version = int(snapshot["version"])
            terminal_expiry_ms = int(snapshot["terminal_expiry_ms"])
            task = cls._deserialize(snapshot["payload"])
            if task.id != task_id or version < 1 or terminal_expiry_ms < 0:
                raise ValueError
        except (AttributeError, DecodeError, KeyError, TypeError, UnicodeError, ValueError) as error:
            msg = f"A2A task '{task_id}' has an invalid Redis snapshot"
            raise ExecutionStoreCorruptionError(msg) from error
        return task, owner, version, terminal_expiry_ms

    @staticmethod
    def _task_score(task: Task) -> float:
        if task.HasField("status") and task.status.HasField("timestamp"):
            return task.status.timestamp.ToNanoseconds() / 1_000_000_000
        return -1.0

    async def _save_payload(
        self,
        task: Task,
        owner: str,
        *,
        expected_version: int | None = None,
    ) -> int:
        """Persist one task and its indexes through an optimistic transaction."""
        await self.cleanup_expired_tasks(limit=10)
        expected_version = self._loaded_task_version(task) if expected_version is None else expected_version
        task_key = self._task_key(task.id)
        for attempt in range(DEFAULT_TRANSACTION_MAX_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(task_key)
                    values = await pipe.hgetall(task_key)
                    if values:
                        _, recorded_owner, current_version, _ = self._decode_snapshot(task.id, values)
                        if recorded_owner != owner:
                            return _OWNER_MISMATCH
                        if expected_version < 0 or current_version != expected_version:
                            return _STALE_VERSION
                    else:
                        if expected_version >= 0:
                            return _STALE_VERSION
                        current_version = 0

                    now_ms = await redis_time_ms(pipe)
                    version = current_version + 1
                    terminal_expiry_ms = now_ms + self.terminal_ttl_seconds * 1_000 if task_is_terminal(task) else 0
                    pipe.multi()
                    pipe.hset(
                        task_key,
                        mapping={
                            "owner": owner,
                            "payload": self._serialize(task),
                            "version": version,
                            "terminal_expiry_ms": terminal_expiry_ms,
                        },
                    )
                    pipe.zadd(self._updates_key(owner), {task.id: self._task_score(task)})
                    if terminal_expiry_ms:
                        pipe.zrem(self._key("active"), task.id)
                        pipe.zadd(self._key("terminal-expiry"), {task.id: terminal_expiry_ms})
                    else:
                        pipe.zadd(self._key("active"), {task.id: self._task_score(task)})
                        pipe.zrem(self._key("terminal-expiry"), task.id)
                    await pipe.execute()
                    self._remember_task_version(task, version)
                    return version
                except redis_watch_error():
                    await redis_transaction_backoff(attempt)
        msg = "A2A task save transaction retry budget exhausted"
        raise ExecutionContentionError(msg)

    async def _load_snapshots(self, task_ids: builtins.list[str]) -> builtins.list[tuple[Task, str, int, int] | None]:
        if not task_ids:
            return []
        async with self.redis.pipeline(transaction=False) as pipe:
            for task_id in task_ids:
                pipe.hgetall(self._task_key(task_id))
            values = await pipe.execute()
        return [
            self._decode_snapshot(task_id, value) if value else None
            for task_id, value in zip(task_ids, values, strict=True)
        ]

    async def save(self, task: Task, context: ServerCallContext) -> None:
        owner = self.owner_id_for_context(context)
        saved = await self._save_payload(task, owner)
        if saved == _STALE_VERSION:
            msg = f"Task '{task.id}' has a stale projection version"
            raise InvalidParamsError(msg)
        if saved < 1:
            msg = f"Task '{task.id}' belongs to another owner"
            raise InvalidParamsError(msg)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        values = await self.redis.hgetall(self._task_key(task_id))
        if not values:
            return None
        task, owner, version, _ = self._decode_snapshot(task_id, values)
        if owner != self.owner_id_for_context(context):
            return None
        self._remember_task_version(task, version)
        return task

    async def save_projection(self, task: Task, owner: str, expected_version: int) -> bool:
        return await self._save_payload(task, owner, expected_version=expected_version) > 0

    async def recoverable_task_batch(
        self, cursor: int, limit: int
    ) -> tuple[builtins.list[tuple[Task, str, int]], int | None]:
        """Return one active-task page for restart projection."""
        await self.cleanup_expired_tasks(limit=limit)
        next_cursor, entries = await self.redis.zscan(self._key("active"), cursor=cursor, count=limit)
        task_ids = [self._decode_value(raw_task_id) for raw_task_id, _score in entries]
        tasks: builtins.list[tuple[Task, str, int]] = []
        for snapshot in await self._load_snapshots(task_ids):
            if snapshot is None:
                continue
            task, owner, version, _ = snapshot
            self._remember_task_version(task, version)
            tasks.append((task, owner, version))
        return tasks, int(next_cursor) or None

    async def cleanup_expired_tasks(self, *, limit: int = 100) -> int:
        now_ms = await redis_time_ms(self.redis)
        expired = await self.redis.zrangebyscore(
            self._key("terminal-expiry"),
            "-inf",
            now_ms,
            start=0,
            num=limit,
        )
        removed = 0
        for raw_task_id in expired:
            task_id = self._decode_value(raw_task_id)
            deleted = await self._delete_payload(task_id, expired_before_ms=now_ms)
            removed += deleted
            if deleted:
                self._forget_task_versions(task_id)
        return removed

    async def _delete_payload(
        self,
        task_id: str,
        *,
        owner: str | None = None,
        expired_before_ms: int | None = None,
    ) -> int:
        task_key = self._task_key(task_id)
        for attempt in range(DEFAULT_TRANSACTION_MAX_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(task_key)
                    values = await pipe.hgetall(task_key)
                    if not values:
                        if expired_before_ms is None:
                            return 0
                        pipe.multi()
                        pipe.zrem(self._key("active"), task_id)
                        pipe.zrem(self._key("terminal-expiry"), task_id)
                        await pipe.execute()
                        return 0

                    _, recorded_owner, _, terminal_expiry_ms = self._decode_snapshot(task_id, values)
                    if owner is not None and recorded_owner != owner:
                        return 0
                    if expired_before_ms is not None and (
                        not terminal_expiry_ms or terminal_expiry_ms > expired_before_ms
                    ):
                        pipe.multi()
                        if terminal_expiry_ms:
                            pipe.zadd(self._key("terminal-expiry"), {task_id: terminal_expiry_ms})
                        else:
                            pipe.zrem(self._key("terminal-expiry"), task_id)
                        await pipe.execute()
                        return 0

                    pipe.multi()
                    pipe.delete(task_key)
                    pipe.zrem(self._updates_key(recorded_owner), task_id)
                    pipe.zrem(self._key("active"), task_id)
                    pipe.zrem(self._key("terminal-expiry"), task_id)
                    await pipe.execute()
                    return 1
                except redis_watch_error():
                    await redis_transaction_backoff(attempt)
        msg = "A2A task delete transaction retry budget exhausted"
        raise ExecutionContentionError(msg)

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        await self.cleanup_expired_tasks(limit=100)
        page_size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
        filtered = bool(params.context_id or params.status or params.HasField("status_timestamp_after"))
        lister = self._list_filtered if filtered else self._list_by_recent_update
        page, total_size, next_page_token = await lister(params, context, page_size)
        return ListTasksResponse(
            tasks=page,
            next_page_token=next_page_token,
            page_size=page_size,
            total_size=total_size,
        )

    async def _start_rank(self, page_token: str, updates_key: str) -> int:
        """Resolve a page token to the next index in the owner update index."""
        if not page_token:
            return 0
        rank = await self.redis.zrevrank(updates_key, decode_page_token(page_token))
        if rank is None:
            msg = f"Invalid page token: {page_token}"
            raise InvalidParamsError(msg)
        return int(rank) + 1

    async def _list_by_recent_update(
        self, params: ListTasksRequest, context: ServerCallContext, page_size: int
    ) -> tuple[builtins.list[Task], int, str | None]:
        owner = self.owner_id_for_context(context)
        updates_key = self._updates_key(owner)
        total_size = await self.redis.zcard(updates_key)
        start_index = await self._start_rank(params.page_token, updates_key)

        task_ids = await self.redis.zrevrange(updates_key, start_index, start_index + page_size - 1)
        task_ids = [self._decode_value(task_id) for task_id in task_ids]
        snapshots = await self._load_snapshots(task_ids)
        page = [snapshot[0] for snapshot in snapshots if snapshot is not None and snapshot[1] == owner]
        has_next_page = start_index + len(task_ids) < total_size
        next_page_token = encode_page_token(task_ids[-1]) if task_ids and has_next_page else None
        return page, total_size, next_page_token

    async def _list_filtered(
        self, params: ListTasksRequest, context: ServerCallContext, page_size: int
    ) -> tuple[builtins.list[Task], int, str | None]:
        """Apply filters in bounded update-index batches with exact pagination."""
        owner = self.owner_id_for_context(context)
        updates_key = self._updates_key(owner)
        indexed_size = int(await self.redis.zcard(updates_key))
        start_rank = await self._start_rank(params.page_token, updates_key)

        page: builtins.list[Task] = []
        matching_after_page = False
        total_size = 0
        batch_size = settings.a2a_list_scan_batch_size
        timestamp_after = params.status_timestamp_after if params.HasField("status_timestamp_after") else None
        for offset in range(0, indexed_size, batch_size):
            task_ids = await self.redis.zrevrange(
                updates_key,
                offset,
                min(indexed_size - 1, offset + batch_size - 1),
            )
            decoded_ids = [self._decode_value(task_id) for task_id in task_ids]
            snapshots = await self._load_snapshots(decoded_ids)
            for rank, snapshot in enumerate(snapshots, start=offset):
                if snapshot is None or snapshot[1] != owner:
                    continue
                task = snapshot[0]
                if not task_matches_filters(task, params, timestamp_after):
                    continue
                total_size += 1
                if rank < start_rank:
                    continue
                if len(page) < page_size:
                    page.append(task)
                else:
                    matching_after_page = True

        has_next_page = bool(page) and matching_after_page
        next_page_token = encode_page_token(page[-1].id) if page and has_next_page else None
        return page, total_size, next_page_token

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        owner = self.owner_id_for_context(context)
        if await self._delete_payload(task_id, owner=owner):
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

    async def initialize(self) -> None:
        """Fail A2A startup when its authoritative task store is unavailable."""
        await self.redis.ping()

    async def health(self) -> dict[str, Any]:
        try:
            await self.redis.ping()
        except Exception as error:
            return {
                "healthy": False,
                "provider": type(self).__name__,
                "error": type(error).__name__,
            }
        return {"healthy": True, "provider": type(self).__name__}

    async def close(self) -> None:
        if self._close_redis:
            await self.redis.aclose()


__all__ = ["RedisTaskStore", "RedisTaskStoreProvider"]
