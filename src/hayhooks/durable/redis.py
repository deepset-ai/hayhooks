"""Redis implementation of the private durable execution store contract."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache, wraps
from importlib.resources import files
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

from hayhooks.durable.models import (
    ExecutionError,
    ExecutionLeaseLostError,
    ExecutionRecord,
    ExecutionRecordSizeError,
    ExecutionStatus,
    ExecutionStoreError,
    JsonValue,
    utc_now,
    validate_json,
)
from hayhooks.server.logger import log
from hayhooks.settings import settings

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from redis.commands.core import AsyncScript


EXECUTION_GROUP = "hayhooks-durable-executions"
_SCRIPT_PACKAGE = "hayhooks.redis_scripts"
_SCRIPT_RESULT_CANCELED = 2
_MIN_REDIS_SERVER_VERSION = (6, 2)


@cache
def _script_source(name: str) -> str:
    """Load an immutable packaged Lua script once per process."""
    return files(_SCRIPT_PACKAGE).joinpath(f"{name}.lua").read_text(encoding="utf-8")


@dataclass(frozen=True)
class _RegisteredScripts:
    """redis-py handles that use EVALSHA and transparently recover from NOSCRIPT."""

    submit: AsyncScript
    checkpoint: AsyncScript
    complete: AsyncScript
    suspend: AsyncScript
    retry: AsyncScript
    resume: AsyncScript
    cancel: AsyncScript
    renew_lease: AsyncScript
    release_lease: AsyncScript
    promote_delayed: AsyncScript
    acknowledge_delivery: AsyncScript
    delay_delivery: AsyncScript
    retire_incompatible: AsyncScript
    cleanup_expired_counts: AsyncScript

    @classmethod
    def register(cls, redis: Redis) -> _RegisteredScripts:
        return cls(
            submit=redis.register_script(_script_source("submit")),
            checkpoint=redis.register_script(_script_source("checkpoint")),
            complete=redis.register_script(_script_source("complete")),
            suspend=redis.register_script(_script_source("suspend")),
            retry=redis.register_script(_script_source("retry")),
            resume=redis.register_script(_script_source("resume")),
            cancel=redis.register_script(_script_source("cancel")),
            renew_lease=redis.register_script(_script_source("renew_lease")),
            release_lease=redis.register_script(_script_source("release_lease")),
            promote_delayed=redis.register_script(_script_source("promote_delayed")),
            acknowledge_delivery=redis.register_script(_script_source("acknowledge_delivery")),
            delay_delivery=redis.register_script(_script_source("delay_delivery")),
            retire_incompatible=redis.register_script(_script_source("retire_incompatible")),
            cleanup_expired_counts=redis.register_script(_script_source("cleanup_expired_counts")),
        )


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _store_operation(operation: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Keep Redis client exceptions behind the transport-neutral store boundary."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return await function(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except (ExecutionLeaseLostError, ExecutionRecordSizeError, ExecutionStoreError):
                raise
            except Exception as error:
                msg = f"Durable execution store {operation} failed"
                raise ExecutionStoreError(msg) from error

        return wrapped

    return decorate


@dataclass(frozen=True)
class _Delivery:
    entry_id: str
    execution_id: str


class RedisExecutionClaim:
    """Fenced owner for one Redis Stream delivery."""

    def __init__(self, store: RedisExecutionStore, delivery: _Delivery, record: ExecutionRecord, token: str) -> None:
        self.store = store
        self.delivery = delivery
        self._record = record
        self.token = token
        self._heartbeat: asyncio.Task[None] | None = None
        self._finished = False
        self._lost = False

    @property
    def record(self) -> ExecutionRecord:
        return self._record

    async def __aenter__(self) -> RedisExecutionClaim:
        if not await self.store._renew_lease(self.record.execution_id, self.token):
            self._lost = True
            self._ensure_owned()
        self._heartbeat = asyncio.create_task(self._heartbeat_loop())
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._heartbeat:
            self._heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat
        if not self._finished:
            await self.store._release_lease(self.record.execution_id, self.token)

    async def checkpoint(self) -> None:
        self._ensure_owned()
        await self.store._checkpoint(self)

    async def cancellation_requested(self) -> bool:
        requested, reason = await self.store._cancellation_requested(self.record.execution_id)
        if requested:
            self.record.cancel_requested_at = self.record.cancel_requested_at or utc_now()
            self.record.cancel_reason = reason
        return requested

    async def complete(self) -> None:
        self._ensure_owned()
        await self.store._complete(self)
        self._finished = True

    async def suspend(self) -> None:
        self._ensure_owned()
        await self.store._suspend(self)
        self._finished = True

    async def retry(self, error: ExecutionError, *, delay: float) -> None:
        self._ensure_owned()
        await self.store._retry(self, error, delay=delay)
        self._finished = True

    def _ensure_owned(self) -> None:
        if self._lost:
            msg = f"Execution lease for '{self.record.execution_id}' was lost"
            raise ExecutionLeaseLostError(msg)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.store.lease_renewal_interval)
            try:
                if not await self.store._renew_lease(self.record.execution_id, self.token):
                    self._lost = True
                    return
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._lost = True
                log.warning("Durable Redis lease heartbeat failed for '{}': {}", self.record.execution_id, error)
                return


class RedisExecutionStore:
    """Per-record Redis/Streams durable store; Redis mechanics never escape this module."""

    def __init__(  # noqa: PLR0913
        self,
        redis: Redis,
        *,
        key_prefix: str | None = None,
        claim_idle_ms: int | None = None,
        queue_block_ms: int | None = None,
        reclaim_interval: float | None = None,
        terminal_ttl_seconds: int | None = None,
        cancellation_ttl_seconds: int | None = None,
        max_stream_length: int | None = None,
        max_progress_events: int | None = None,
        max_record_bytes: int | None = None,
        delayed_promotion_interval: float | None = None,
        delayed_promotion_batch_size: int | None = None,
        close_redis: bool = False,
    ) -> None:
        key_prefix = key_prefix or settings.durable_redis_key_prefix
        claim_idle_ms = claim_idle_ms if claim_idle_ms is not None else settings.durable_redis_claim_idle_ms
        queue_block_ms = queue_block_ms if queue_block_ms is not None else settings.durable_redis_queue_block_ms
        reclaim_interval = reclaim_interval if reclaim_interval is not None else settings.durable_redis_reclaim_interval
        terminal_ttl_seconds = (
            terminal_ttl_seconds if terminal_ttl_seconds is not None else settings.durable_terminal_ttl_seconds
        )
        cancellation_ttl_seconds = (
            cancellation_ttl_seconds
            if cancellation_ttl_seconds is not None
            else settings.durable_redis_cancellation_ttl_seconds
        )
        max_stream_length = (
            max_stream_length if max_stream_length is not None else settings.durable_redis_stream_max_length
        )
        max_progress_events = (
            max_progress_events if max_progress_events is not None else settings.durable_max_progress_events
        )
        max_record_bytes = max_record_bytes if max_record_bytes is not None else settings.durable_max_record_bytes
        delayed_promotion_interval = (
            delayed_promotion_interval
            if delayed_promotion_interval is not None
            else settings.durable_redis_delayed_promotion_interval
        )
        delayed_promotion_batch_size = (
            delayed_promotion_batch_size
            if delayed_promotion_batch_size is not None
            else settings.durable_redis_delayed_promotion_batch_size
        )
        if claim_idle_ms < 1:
            msg = "claim_idle_ms must be positive"
            raise ValueError(msg)
        if queue_block_ms < 1:
            msg = "queue_block_ms must be positive"
            raise ValueError(msg)
        if reclaim_interval < 0:
            msg = "reclaim_interval cannot be negative"
            raise ValueError(msg)
        self.redis = redis
        self.key_prefix = key_prefix.rstrip(":")
        self.claim_idle_ms = claim_idle_ms
        self.queue_block_ms = queue_block_ms
        self.reclaim_interval = reclaim_interval
        self.terminal_ttl_seconds = terminal_ttl_seconds
        self.cancellation_ttl_seconds = cancellation_ttl_seconds
        self.max_stream_length = max_stream_length
        self.max_progress_events = max(1, max_progress_events)
        self.max_record_bytes = max_record_bytes
        self.delayed_promotion_interval = delayed_promotion_interval
        self.delayed_promotion_batch_size = delayed_promotion_batch_size
        self.close_redis = close_redis
        self.stream_key = f"{self.key_prefix}:queue"
        self.delayed_key = f"{self.key_prefix}:delayed"
        self.state_counts_key = f"{self.key_prefix}:state-counts"
        self.active_revisions_key = f"{self.key_prefix}:active-revisions"
        self.record_sequences_key = f"{self.key_prefix}:record-sequences"
        self.terminal_count_index_key = f"{self.key_prefix}:terminal-count-expiry"
        self.terminal_count_states_key = f"{self.key_prefix}:terminal-count-states"
        self._scripts = _RegisteredScripts.register(redis)
        self._delayed_promotion_lock = asyncio.Lock()
        self._count_cleanup_lock = asyncio.Lock()
        self._next_delayed_promotion_at = 0.0
        self._next_count_cleanup_at = 0.0
        self._reclaim_cursors: dict[str, str] = {}
        self._next_reclaim_at: dict[str, float] = {}

    @property
    def lease_renewal_interval(self) -> float:
        return max(0.001, self.claim_idle_ms / 3_000)

    @_store_operation("initialization")
    async def initialize(self) -> None:
        # Ping first: deployment must fail clearly rather than silently select memory.
        await self.redis.ping()
        await self._check_server_version()
        await self._warn_on_unsafe_redis_configuration()
        try:
            await self.redis.xgroup_create(self.stream_key, EXECUTION_GROUP, id="0", mkstream=True)
        except Exception as error:
            if "BUSYGROUP" not in str(error):
                msg = f"Unable to initialize durable Redis consumer group: {error}"
                raise RuntimeError(msg) from error

    async def _check_server_version(self) -> None:
        try:
            server = await self.redis.info("server")
        except Exception:
            log.debug("Durable Redis server version could not be inspected; managed services may restrict INFO")
            return
        raw_version = server.get("redis_version", server.get(b"redis_version"))
        if raw_version is None:
            log.debug("Durable Redis INFO response did not include redis_version")
            return
        version = _decode(raw_version)
        try:
            major, minor = (int(part) for part in version.split(".", 2)[:2])
        except (TypeError, ValueError):
            log.debug("Durable Redis reported an unrecognized server version '{}'", version)
            return
        if (major, minor) < _MIN_REDIS_SERVER_VERSION:
            required = ".".join(str(part) for part in _MIN_REDIS_SERVER_VERSION)
            msg = (
                f"Durable execution requires Redis server {required} or newer because recovery uses XAUTOCLAIM; "
                f"found {version}"
            )
            raise ExecutionStoreError(msg)

    async def _warn_on_unsafe_redis_configuration(self) -> None:
        try:
            config = await self.redis.config_get("maxmemory-policy", "appendonly", "save")
        except Exception:
            log.debug("Durable Redis configuration could not be inspected; managed services may restrict CONFIG GET")
            return
        eviction = _decode(config.get("maxmemory-policy", config.get(b"maxmemory-policy", "unknown")))
        if eviction not in {"noeviction", "unknown"}:
            log.warning(
                "Durable Redis uses maxmemory-policy='{}'; eviction can delete execution records. "
                "Use 'noeviction' for production.",
                eviction,
            )
        appendonly = _decode(config.get("appendonly", config.get(b"appendonly", "no")))
        save = _decode(config.get("save", config.get(b"save", "")))
        if appendonly != "yes" and not save.strip():
            log.warning(
                "Durable Redis has neither AOF nor RDB persistence configured; process restart can lose executions"
            )

    @_store_operation("submission")
    async def submit(self, record: ExecutionRecord) -> bool:
        record.max_progress_events = self.max_progress_events
        record.max_record_bytes = self.max_record_bytes
        return bool(
            await self._scripts.submit(
                keys=[
                    self._record_key(record.execution_id),
                    self.stream_key,
                    self.state_counts_key,
                    self.active_revisions_key,
                    self.record_sequences_key,
                ],
                args=[
                    record.to_json(),
                    record.execution_id,
                    record.definition_revision,
                    record.sequence,
                ],
            )
        )

    @_store_operation("read")
    async def get(self, execution_id: str) -> ExecutionRecord | None:
        payload = await self.redis.get(self._record_key(execution_id))
        return self._record_from_payload(payload)

    @_store_operation("batch read")
    async def get_many(self, execution_ids: list[str]) -> dict[str, ExecutionRecord | None]:
        """Load execution records in one Redis round trip."""
        if not execution_ids:
            return {}
        payloads = await self.redis.mget([self._record_key(execution_id) for execution_id in execution_ids])
        return {
            execution_id: self._record_from_payload(payload)
            for execution_id, payload in zip(execution_ids, payloads, strict=True)
        }

    @_store_operation("changed-record read")
    async def get_changed(self, known_sequences: Mapping[str, int]) -> dict[str, ExecutionRecord | None]:
        """Fetch full records only when their lightweight persisted sequence changed."""
        if not known_sequences:
            return {}
        execution_ids = list(known_sequences)
        sequences = await self.redis.hmget(self.record_sequences_key, execution_ids)
        changed_ids = [
            execution_id
            for execution_id, sequence in zip(execution_ids, sequences, strict=True)
            if sequence is None or int(sequence) != known_sequences[execution_id]
        ]
        return await self.get_many(changed_ids)

    @_store_operation("claim")
    async def claim_next(self, worker_name: str) -> RedisExecutionClaim | None:
        await self._cleanup_expired_counts()
        await self._promote_delayed()
        delivery = await self._next_delivery(worker_name)
        if delivery is None:
            return None
        token = await self._acquire_lease(delivery.execution_id)
        if token is None:
            return None
        record = await self.get(delivery.execution_id)
        if record is None or record.terminal or record.status is ExecutionStatus.WAITING:
            await self._acknowledge_delivery(delivery.entry_id)
            await self._release_lease(delivery.execution_id, token)
            return None
        if record.retry_at and record.retry_at > utc_now():
            await self._delay_delivery(delivery, record.retry_at.timestamp())
            await self._release_lease(delivery.execution_id, token)
            return None
        record.status = ExecutionStatus.RUNNING
        record.attempt += 1
        if record.error is not None:
            record.last_retry_error = record.error
            record.error = None
        record.retry_at = None
        record.touch()
        try:
            await self._save_owned(record, token)
        except Exception:
            await self._release_lease(delivery.execution_id, token)
            raise
        return RedisExecutionClaim(self, delivery, record, token)

    @_store_operation("cancellation")
    async def request_cancel(self, execution_id: str, reason: str | None = None) -> bool:
        return bool(
            await self._scripts.cancel(
                keys=[
                    self._record_key(execution_id),
                    self.state_counts_key,
                    self.terminal_count_index_key,
                    self.terminal_count_states_key,
                    self.active_revisions_key,
                    self.record_sequences_key,
                ],
                args=[
                    utc_now().isoformat(),
                    reason or "",
                    self.terminal_ttl_seconds,
                    self.max_progress_events,
                    execution_id,
                    time.time() + self.terminal_ttl_seconds,
                ],
            )
        )

    @_store_operation("resume")
    async def resume(self, execution_id: str, update: JsonValue | None = None) -> bool:
        record = await self.get(execution_id)
        if record is None or record.status is not ExecutionStatus.WAITING:
            return False
        if update is not None:
            record.application_state["__hayhooks_resume_input"] = validate_json(
                update, limit=record.max_record_bytes, label="resume update"
            )
        record.status = ExecutionStatus.QUEUED
        record.wait = None
        record.append_progress("Execution resumed", kind="resumed")
        return bool(
            await self._scripts.resume(
                keys=[
                    self._record_key(execution_id),
                    self.stream_key,
                    self.state_counts_key,
                    self.active_revisions_key,
                    self.record_sequences_key,
                ],
                args=[record.to_json(), execution_id],
            )
        )

    @_store_operation("revision retirement")
    async def retire_incompatible(self, definition_revision: str) -> int:
        """Atomically retire queued/waiting records from replaced definitions."""
        retired = 0
        cursor: int | bytes = 0
        candidates: list[str] = []
        while True:
            cursor, revisions = await self.redis.hscan(self.active_revisions_key, cursor=cursor, count=100)
            revisions = cast(dict[Any, Any], revisions)
            for raw_execution_id, raw_revision in revisions.items():
                if _decode(raw_revision) != definition_revision:
                    candidates.append(_decode(raw_execution_id))
            if int(cursor) == 0:
                break

        # Finish the stable index scan before terminal transitions remove fields
        # from that same hash, then process candidates in bounded network batches.
        for offset in range(0, len(candidates), 100):
            now = utc_now().isoformat()
            retirements = [
                self._scripts.retire_incompatible(
                    keys=[
                        self._record_key(execution_id),
                        self.delayed_key,
                        self.state_counts_key,
                        self.terminal_count_index_key,
                        self.terminal_count_states_key,
                        self.active_revisions_key,
                        self.record_sequences_key,
                    ],
                    args=[
                        definition_revision,
                        now,
                        self.terminal_ttl_seconds,
                        self.max_progress_events,
                        time.time() + self.terminal_ttl_seconds,
                        execution_id,
                    ],
                )
                for execution_id in candidates[offset : offset + 100]
            ]
            retired += sum(int(result) for result in await asyncio.gather(*retirements))
        return retired

    @_store_operation("operational counts")
    async def operational_counts(self) -> dict[str, int]:
        """Build a constant-time, payload-free state snapshot for health checks."""
        await self._cleanup_expired_counts(force=True)
        stream_length, pending_summary, delayed, state_counts = await asyncio.gather(
            self.redis.xlen(self.stream_key),
            self.redis.xpending(self.stream_key, EXECUTION_GROUP),
            self.redis.zcard(self.delayed_key),
            self.redis.hmget(self.state_counts_key, [status.value for status in ExecutionStatus]),
        )
        counts = {status.value: int(value or 0) for status, value in zip(ExecutionStatus, state_counts, strict=True)}
        if isinstance(pending_summary, dict):
            pending = int(pending_summary.get("pending", pending_summary.get(b"pending", 0)))
        else:
            pending = int(pending_summary[0]) if pending_summary else 0
        counts["queue_deliveries"] = int(stream_length)
        counts["pending_deliveries"] = pending
        counts["delayed"] = int(delayed)
        return counts

    @_store_operation("close")
    async def close(self) -> None:
        if self.close_redis:
            await self.redis.aclose()

    @_store_operation("checkpoint")
    async def _checkpoint(self, claim: RedisExecutionClaim) -> None:
        await self._save_owned(claim.record, claim.token)

    @_store_operation("completion")
    async def _complete(self, claim: RedisExecutionClaim) -> None:
        if not claim.record.terminal:
            msg = f"Execution '{claim.record.execution_id}' is not terminal"
            raise RuntimeError(msg)
        regular = claim.record.to_json()
        canceled = deepcopy(claim.record)
        # The Lua script selects this payload when it observes a cancellation
        # at the same atomic terminal-write boundary.
        canceled.mark_canceled(force=True)
        result = await self._scripts.complete(
            keys=[
                self._lease_key(claim.record.execution_id),
                self._record_key(claim.record.execution_id),
                self.stream_key,
                self.state_counts_key,
                self.terminal_count_index_key,
                self.terminal_count_states_key,
                self.active_revisions_key,
                self.record_sequences_key,
            ],
            args=[
                claim.token,
                regular,
                canceled.to_json(),
                self.terminal_ttl_seconds,
                EXECUTION_GROUP,
                claim.delivery.entry_id,
                claim.record.execution_id,
                time.time() + self.terminal_ttl_seconds,
            ],
        )
        if int(result) == -1:
            msg = f"Execution lease for '{claim.record.execution_id}' was lost"
            raise ExecutionLeaseLostError(msg)
        if int(result) == _SCRIPT_RESULT_CANCELED:
            claim.record.mark_canceled()

    @_store_operation("suspension")
    async def _suspend(self, claim: RedisExecutionClaim) -> None:
        if claim.record.status is not ExecutionStatus.WAITING:
            msg = "Only waiting executions can be suspended"
            raise RuntimeError(msg)
        result = await self._scripts.suspend(
            keys=[
                self._lease_key(claim.record.execution_id),
                self._record_key(claim.record.execution_id),
                self.stream_key,
                self.state_counts_key,
                self.terminal_count_index_key,
                self.terminal_count_states_key,
                self.active_revisions_key,
                self.record_sequences_key,
            ],
            args=[
                claim.token,
                claim.record.to_json(),
                self._canceled_payload(claim.record),
                self.terminal_ttl_seconds,
                EXECUTION_GROUP,
                claim.delivery.entry_id,
                claim.record.execution_id,
                time.time() + self.terminal_ttl_seconds,
            ],
        )
        if not result:
            msg = f"Execution lease for '{claim.record.execution_id}' was lost"
            raise ExecutionLeaseLostError(msg)
        if int(result) == _SCRIPT_RESULT_CANCELED:
            claim.record.mark_canceled(force=True)

    @_store_operation("retry")
    async def _retry(self, claim: RedisExecutionClaim, error: ExecutionError, *, delay: float) -> None:
        record = claim.record
        record.status = ExecutionStatus.QUEUED
        record.error = error
        retry_at = utc_now().timestamp() + delay
        record.retry_at = datetime.fromtimestamp(retry_at, tz=timezone.utc)
        record.append_progress("Execution retry scheduled", kind="retry")
        result = await self._scripts.retry(
            keys=[
                self._lease_key(record.execution_id),
                self._record_key(record.execution_id),
                self.delayed_key,
                self.stream_key,
                self.state_counts_key,
                self.terminal_count_index_key,
                self.terminal_count_states_key,
                self.active_revisions_key,
                self.record_sequences_key,
            ],
            args=[
                claim.token,
                record.to_json(),
                self._canceled_payload(record),
                self.terminal_ttl_seconds,
                retry_at,
                record.execution_id,
                EXECUTION_GROUP,
                claim.delivery.entry_id,
                time.time() + self.terminal_ttl_seconds,
            ],
        )
        if not result:
            msg = f"Execution lease for '{record.execution_id}' was lost"
            raise ExecutionLeaseLostError(msg)
        if int(result) == _SCRIPT_RESULT_CANCELED:
            record.mark_canceled(force=True)

    @_store_operation("checkpoint")
    async def _save_owned(self, record: ExecutionRecord, token: str) -> None:
        record.touch()
        saved = await self._scripts.checkpoint(
            keys=[
                self._lease_key(record.execution_id),
                self._record_key(record.execution_id),
                self.state_counts_key,
                self.active_revisions_key,
                self.record_sequences_key,
            ],
            args=[token, record.to_json(), record.execution_id],
        )
        if not saved:
            msg = f"Execution lease for '{record.execution_id}' was lost"
            raise ExecutionLeaseLostError(msg)
        if int(saved) == _SCRIPT_RESULT_CANCELED:
            persisted = await self.get(record.execution_id)
            if persisted is not None:
                record.cancel_requested_at = persisted.cancel_requested_at
                record.cancel_reason = persisted.cancel_reason

    @_store_operation("delivery")
    async def _next_delivery(self, worker_name: str) -> _Delivery | None:
        reclaimed = await self._reclaim_delivery(worker_name)
        if reclaimed is not None:
            return reclaimed
        batches = cast(
            list[tuple[Any, list[Any]]],
            await self.redis.xreadgroup(
                EXECUTION_GROUP,
                worker_name,
                {self.stream_key: ">"},
                count=1,
                block=self.queue_block_ms,
            ),
        )
        return self._delivery(batches[0][1][0]) if batches else None

    async def _reclaim_delivery(self, worker_name: str) -> _Delivery | None:
        now = time.monotonic()
        if now < self._next_reclaim_at.get(worker_name, 0.0):
            return None
        self._next_reclaim_at[worker_name] = now + self.reclaim_interval
        cursor = self._reclaim_cursors.get(worker_name, "0-0")
        try:
            claimed = await self.redis.xautoclaim(
                self.stream_key, EXECUTION_GROUP, worker_name, self.claim_idle_ms, cursor, count=1
            )
        except BaseException:
            # Store failures already back off at the manager; make the next healthy
            # attempt eligible to recover pending work immediately.
            self._next_reclaim_at.pop(worker_name, None)
            raise
        if claimed:
            self._reclaim_cursors[worker_name] = _decode(claimed[0])
            entries = claimed[1] if len(claimed) > 1 else []
            if entries:
                return self._delivery(entries[0])
        return None

    @_store_operation("delayed promotion")
    async def _promote_delayed(self) -> None:
        now = time.monotonic()
        if now < self._next_delayed_promotion_at:
            return
        async with self._delayed_promotion_lock:
            now = time.monotonic()
            if now < self._next_delayed_promotion_at:
                return
            self._next_delayed_promotion_at = now + self.delayed_promotion_interval
            try:
                await self._scripts.promote_delayed(
                    keys=[self.delayed_key, self.stream_key],
                    args=[utc_now().timestamp(), self.delayed_promotion_batch_size],
                )
            except BaseException:
                # Do not defer recovery after a failed Redis call.
                self._next_delayed_promotion_at = 0.0
                raise

    async def _cleanup_expired_counts(self, *, force: bool = False) -> int:
        """Clean one bounded terminal-retention batch outside the health critical path."""
        now = time.monotonic()
        if not force and now < self._next_count_cleanup_at:
            return 0
        async with self._count_cleanup_lock:
            now = time.monotonic()
            if not force and now < self._next_count_cleanup_at:
                return 0
            batch_size = 1_000
            removed = int(
                await self._scripts.cleanup_expired_counts(
                    keys=[
                        self.state_counts_key,
                        self.terminal_count_index_key,
                        self.terminal_count_states_key,
                        self.record_sequences_key,
                    ],
                    args=[time.time(), batch_size],
                )
            )
            # Drain a backlog one worker iteration at a time; otherwise stay idle
            # for a minute. Health checks may force one bounded pass for freshness.
            self._next_count_cleanup_at = 0.0 if removed == batch_size else now + 60.0
            return removed

    @_store_operation("delivery acknowledgement")
    async def _acknowledge_delivery(self, entry_id: str) -> None:
        await self._scripts.acknowledge_delivery(
            keys=[self.stream_key],
            args=[EXECUTION_GROUP, entry_id],
        )

    @_store_operation("delivery delay")
    async def _delay_delivery(self, delivery: _Delivery, retry_at: float) -> None:
        await self._scripts.delay_delivery(
            keys=[self.delayed_key, self.stream_key],
            args=[retry_at, delivery.execution_id, EXECUTION_GROUP, delivery.entry_id],
        )

    @staticmethod
    def _delivery(entry: Any) -> _Delivery:
        entry_id, fields = entry
        execution_id = fields.get("execution_id") or fields.get(b"execution_id")
        if execution_id is None:
            msg = f"Durable Redis delivery {entry_id!r} has no execution_id"
            raise ValueError(msg)
        return _Delivery(_decode(entry_id), _decode(execution_id))

    def _record_key(self, execution_id: str) -> str:
        return f"{self.key_prefix}:execution:{quote(execution_id, safe='-_.')}:record"

    def _lease_key(self, execution_id: str) -> str:
        return f"{self.key_prefix}:execution:{quote(execution_id, safe='-_.')}:lease"

    def _record_from_payload(self, payload: Any) -> ExecutionRecord | None:
        return (
            ExecutionRecord.from_json(
                payload,
                max_progress_events=self.max_progress_events,
                max_record_bytes=self.max_record_bytes,
            )
            if payload is not None
            else None
        )

    @_store_operation("lease acquisition")
    async def _acquire_lease(self, execution_id: str) -> str | None:
        token = uuid.uuid4().hex
        acquired = await self.redis.set(self._lease_key(execution_id), token, nx=True, px=self.claim_idle_ms)
        return token if acquired else None

    @_store_operation("lease renewal")
    async def _renew_lease(self, execution_id: str, token: str) -> bool:
        return bool(
            await self._scripts.renew_lease(keys=[self._lease_key(execution_id)], args=[token, self.claim_idle_ms])
        )

    @_store_operation("lease release")
    async def _release_lease(self, execution_id: str, token: str) -> None:
        await self._scripts.release_lease(keys=[self._lease_key(execution_id)], args=[token])

    @_store_operation("cancellation read")
    async def _cancellation_requested(self, execution_id: str) -> tuple[bool, str | None]:
        record = await self.get(execution_id)
        if record is None or record.cancel_requested_at is None:
            return False, None
        return True, record.cancel_reason

    @staticmethod
    def _canceled_payload(record: ExecutionRecord) -> str:
        canceled = deepcopy(record)
        canceled.mark_canceled(force=True)
        return canceled.to_json()


class RedisExecutionStoreProvider:
    """One Redis client with isolated deployment namespaces."""

    def __init__(  # noqa: PLR0913
        self,
        redis_url: str | None = None,
        *,
        key_prefix: str | None = None,
        redis: Redis | None = None,
        claim_idle_ms: int | None = None,
        queue_block_ms: int | None = None,
        reclaim_interval: float | None = None,
        terminal_ttl_seconds: int | None = None,
        cancellation_ttl_seconds: int | None = None,
        max_stream_length: int | None = None,
        max_progress_events: int | None = None,
        max_record_bytes: int | None = None,
        delayed_promotion_interval: float | None = None,
        delayed_promotion_batch_size: int | None = None,
        close_redis: bool = True,
    ) -> None:
        self.key_prefix = (key_prefix or settings.durable_redis_key_prefix).rstrip(":")
        self.claim_idle_ms = claim_idle_ms if claim_idle_ms is not None else settings.durable_redis_claim_idle_ms
        self.queue_block_ms = queue_block_ms if queue_block_ms is not None else settings.durable_redis_queue_block_ms
        self.reclaim_interval = (
            reclaim_interval if reclaim_interval is not None else settings.durable_redis_reclaim_interval
        )
        self.terminal_ttl_seconds = (
            terminal_ttl_seconds if terminal_ttl_seconds is not None else settings.durable_terminal_ttl_seconds
        )
        self.cancellation_ttl_seconds = (
            cancellation_ttl_seconds
            if cancellation_ttl_seconds is not None
            else settings.durable_redis_cancellation_ttl_seconds
        )
        self.max_stream_length = (
            max_stream_length if max_stream_length is not None else settings.durable_redis_stream_max_length
        )
        self.max_progress_events = (
            max_progress_events if max_progress_events is not None else settings.durable_max_progress_events
        )
        self.max_record_bytes = max_record_bytes if max_record_bytes is not None else settings.durable_max_record_bytes
        self.delayed_promotion_interval = (
            delayed_promotion_interval
            if delayed_promotion_interval is not None
            else settings.durable_redis_delayed_promotion_interval
        )
        self.delayed_promotion_batch_size = (
            delayed_promotion_batch_size
            if delayed_promotion_batch_size is not None
            else settings.durable_redis_delayed_promotion_batch_size
        )
        self.stores: dict[str, RedisExecutionStore] = {}
        self.close_redis = close_redis
        self.redis_url = redis_url or settings.durable_redis_url
        if redis is None:
            try:
                from redis.asyncio import Redis as AsyncRedis
            except ImportError as error:  # pragma: no cover - optional feature error
                msg = 'Durable Redis storage requires `pip install "hayhooks[durable]`.'
                raise ImportError(msg) from error
            redis = AsyncRedis.from_url(self.redis_url, decode_responses=False)
        self.redis = redis

    def create_execution_store(self, deployment_name: str) -> RedisExecutionStore:
        if deployment_name not in self.stores:
            namespace = quote(deployment_name, safe="-_.")
            self.stores[deployment_name] = RedisExecutionStore(
                self.redis,
                key_prefix=f"{self.key_prefix}:deployment:{namespace}",
                claim_idle_ms=self.claim_idle_ms,
                queue_block_ms=self.queue_block_ms,
                reclaim_interval=self.reclaim_interval,
                terminal_ttl_seconds=self.terminal_ttl_seconds,
                cancellation_ttl_seconds=self.cancellation_ttl_seconds,
                max_stream_length=self.max_stream_length,
                max_progress_events=self.max_progress_events,
                max_record_bytes=self.max_record_bytes,
                delayed_promotion_interval=self.delayed_promotion_interval,
                delayed_promotion_batch_size=self.delayed_promotion_batch_size,
            )
        return self.stores[deployment_name]

    async def close(self) -> None:
        if self.close_redis:
            await self.redis.aclose()


__all__ = [
    "RedisExecutionStore",
    "RedisExecutionStoreProvider",
]
