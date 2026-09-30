"""Redis implementation of the durable execution store."""
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0913

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import fields, replace
from itertools import chain
from typing import Any, cast

from loguru import logger as log

try:
    from redis.exceptions import RedisError, WatchError
except ImportError as error:  # pragma: no cover - exercised by packaging checks
    raise RuntimeError("Redis durable storage requires `hayhooks[durable]`") from error

from hayhooks.durable.engine import (
    MAX_CONTROL_SCALAR_BYTES,
    STORAGE_SCHEMA_VERSION,
    Claim,
    ExecutionCommand,
    ExecutionControl,
    ExecutionLeaseLostError,
    ExecutionNotFoundError,
    ExecutionStatus,
    Heartbeat,
    InvalidExecutionTransitionError,
    LeaseIndexUpdate,
    PayloadKind,
    ProgressEvent,
    RecoverExpiredLease,
    TransitionPlan,
    decide,
    submission_plan,
    validate_run_id,
)
from hayhooks.durable.store import (
    CHUNK_CURSOR_START,
    LEASE_COMMANDS,
    MAINTENANCE_BATCH_SIZE,
    PUBLIC_PAYLOAD_KINDS,
    ChunkCursorExpiredError,
    ExecutionAdmissionError,
    ExecutionContentionError,
    ExecutionIdempotencyConflictError,
    ExecutionStoreCorruptionError,
    ExecutionStoreError,
    StoreConfig,
    StoredExecution,
    StreamChunk,
    SubmissionResult,
    bind_store_command,
    chunk_read_count,
    parse_chunk_cursor,
    runnable_score,
    validate_payload_size,
    validate_stored_execution,
    validate_transition_plan,
)

_KEY_PREFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_OPTIONAL_CONTROL_FIELDS = {
    "owner_id",
    "available_at_ms",
    "lease_owner",
    "lease_expires_at_ms",
    "cancel_requested_at_ms",
    "cancel_reason",
}
_INTEGER_CONTROL_FIELDS = {
    "schema_version",
    "version",
    "fence",
    "run_attempt",
    "application_retry_count",
    "progress_sequence",
    "created_at_ms",
    "updated_at_ms",
    "available_at_ms",
    "lease_expires_at_ms",
    "cancel_requested_at_ms",
}
_CONTROL_FIELDS = {field.name for field in fields(ExecutionControl)}
_MAX_SAFE_INTEGER = 2**53 - 1
_DEFAULT_TRANSACTION_RETRIES = 8
_DEFAULT_TRANSACTION_BACKOFF_MS = 25
_PROGRESS_SEQUENCE_BYTES = 8
_MAX_COMMAND_VALUES = 1_000

# Refusals of the guarded apply script, which returns 1 once it commits.
_STALE_SNAPSHOT, _LEASE_LOST, _CAPACITY_UNDERFLOW = 0, -1, -2

# A Redis command as (name, key, *arguments); scripts receive the key through KEYS.
_Command = tuple[Any, ...]

# Returns Redis TIME in milliseconds while the worker still owns the lease with room for the safety margin.
_OWNED_LUA = """
local function owned_now(control, worker, fence, margin)
  local lease = redis.call('HMGET', control, 'status', 'lease_owner', 'fence', 'lease_expires_at_ms')
  if lease[1] ~= 'running' or lease[2] ~= worker or lease[3] ~= fence then
    return nil
  end
  local time = redis.call('TIME')
  local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
  if now < tonumber(lease[4]) - tonumber(margin) then
    return now
  end
  return nil
end
"""

# KEYS: control, chunks. ARGV: worker, fence, attempt, safety margin, maxlen, chunk...
_APPEND_CHUNKS_LUA = """
if not owned_now(KEYS[1], ARGV[1], ARGV[2], ARGV[4]) or redis.call('HGET', KEYS[1], 'run_attempt') ~= ARGV[3] then
  return 0
end
for index = 6, #ARGV do
  redis.call('XADD', KEYS[2], 'MAXLEN', ARGV[5], '*', 'attempt', ARGV[3], 'data', ARGV[index])
end
return 1
"""

# KEYS: control, lease expiry. ARGV: worker, fence, safety margin, lease duration, lease member.
_HEARTBEAT_LUA = """
local now = owned_now(KEYS[1], ARGV[1], ARGV[2], ARGV[3])
if not now then
  return false
end
local deadline = string.format('%d', now + tonumber(ARGV[4]))
-- A script error does not roll back writes, so validate the index before renewing control.
redis.call('ZCARD', KEYS[2])
redis.call('HSET', KEYS[1], 'lease_expires_at_ms', deadline)
redis.call('ZADD', KEYS[2], deadline, ARGV[5])
return redis.call('HGETALL', KEYS[1])
"""

# KEYS: control, capacity, then every key the commands touch.
# ARGV: worker ('' when unowned), fence, safety margin, releases capacity (0/1), snapshot field count,
# the snapshot field/value pairs, then each command as argument count, name, key index, arguments.
_APPLY_LUA = """
local stored = redis.call('HGETALL', KEYS[1])
local size = tonumber(ARGV[5])
if #stored ~= 2 * size then
  return 0
end
local fields = {}
for index = 1, #stored, 2 do
  fields[stored[index]] = stored[index + 1]
end
local cursor = 6
for _ = 1, size do
  if fields[ARGV[cursor]] ~= ARGV[cursor + 1] then
    return 0
  end
  cursor = cursor + 2
end
if ARGV[1] ~= '' and not owned_now(KEYS[1], ARGV[1], ARGV[2], ARGV[3]) then
  return -1
end
if ARGV[4] == '1' then
  local capacity = redis.call('HGET', KEYS[2], 'nonterminal')
  if not capacity or not string.match(capacity, '^[1-9]%d*$') or tonumber(capacity) > 2^53 - 1 then
    return -2
  end
end
-- Check each type-sensitive target before any mutation: Redis scripts have no rollback.
-- HGETALL and HGET above already validate the control and capacity hashes.
local checks = {RPUSH = 'LLEN', ZADD = 'ZCARD', ZREM = 'ZCARD', XADD = 'XLEN'}
local checked = {}
local probe = cursor
while probe <= #ARGV do
  local check = checks[ARGV[probe + 1]]
  local key = KEYS[tonumber(ARGV[probe + 2])]
  if check and not checked[key] then
    redis.call(check, key)
    checked[key] = true
  end
  probe = probe + 3 + tonumber(ARGV[probe])
end
while cursor <= #ARGV do
  local count = tonumber(ARGV[cursor])
  local command = {ARGV[cursor + 1], KEYS[tonumber(ARGV[cursor + 2])]}
  for index = cursor + 3, cursor + 2 + count do
    command[#command + 1] = ARGV[index]
  end
  redis.call(unpack(command))
  cursor = cursor + 3 + count
end
return 1
"""


class RedisKeys:
    """Build private, cluster-safe keys for one deployment."""

    def __init__(self, key_prefix: str, deployment: str) -> None:
        prefix = key_prefix.rstrip(":")
        if not _KEY_PREFIX.fullmatch(prefix):
            raise ValueError("key_prefix must contain only letters, numbers, '.', '_', ':', or '-'")
        try:
            deployment_bytes = deployment.encode()
        except (AttributeError, UnicodeError) as error:
            raise ValueError("deployment must be valid UTF-8 text") from error
        if not deployment_bytes or len(deployment_bytes) > MAX_CONTROL_SCALAR_BYTES:
            raise ValueError(f"deployment must be between 1 and {MAX_CONTROL_SCALAR_BYTES} UTF-8 bytes")
        deployment_digest = hashlib.sha256(b"hayhooks-durable:deployment:" + deployment_bytes).hexdigest()
        self.base = f"{prefix}:{{{deployment_digest}}}"

    @property
    def runnable(self) -> str:
        return f"{self.base}:runnable"

    def runnable_revision(self, revision: str) -> str:
        digest = hashlib.sha256(b"hayhooks-durable:revision:" + revision.encode()).hexdigest()
        return f"{self.base}:runnable:{digest}"

    @property
    def lease_expiry(self) -> str:
        return f"{self.base}:lease-expiry"

    @property
    def capacity(self) -> str:
        return f"{self.base}:capacity"

    def control(self, run_id: str) -> str:
        return f"{self._execution(run_id)}:control"

    def progress(self, run_id: str) -> str:
        return f"{self._execution(run_id)}:progress"

    def chunks(self, run_id: str) -> str:
        return f"{self._execution(run_id)}:chunks"

    def payload(self, run_id: str, kind: PayloadKind) -> str:
        return f"{self._execution(run_id)}:{kind.value}"

    def idempotency(self, digest: str) -> str:
        private_digest = hashlib.sha256(b"hayhooks-durable:idempotency:" + digest.encode()).hexdigest()
        return f"{self.base}:idem:{private_digest}"

    @staticmethod
    def lease_member(run_id: str, fence: int) -> str:
        validate_run_id(run_id)
        if not 0 <= fence <= _MAX_SAFE_INTEGER:
            raise ValueError("fence must be a non-negative safe integer")
        return f"{run_id}|{fence}"

    def _execution(self, run_id: str) -> str:
        validate_run_id(run_id)
        return f"{self.base}:exec:{run_id}"


def encode_control(control: ExecutionControl) -> dict[str, str]:
    """Encode every present control field for one Redis Hash."""
    encoded: dict[str, str] = {}
    for field in fields(control):
        value = getattr(control, field.name)
        if value is not None:
            encoded[field.name] = value.value if isinstance(value, ExecutionStatus) else str(value)
    return encoded


def decode_control(values: Mapping[str | bytes, str | bytes | int], *, expected_run_id: str) -> ExecutionControl:
    """Decode one strict Redis control Hash or report backend corruption."""
    try:
        decoded = {_text(key): _text(value) for key, value in values.items()}
    except (TypeError, UnicodeError) as error:
        raise ExecutionStoreCorruptionError("control Hash contains invalid UTF-8") from error
    missing = _CONTROL_FIELDS.difference(_OPTIONAL_CONTROL_FIELDS, decoded)
    unknown = decoded.keys() - _CONTROL_FIELDS
    if missing or unknown:
        details = ", ".join(sorted(missing or unknown))
        raise ExecutionStoreCorruptionError(f"control Hash has missing or unknown fields: {details}")
    if any(len(value.encode()) > MAX_CONTROL_SCALAR_BYTES for value in decoded.values()):
        raise ExecutionStoreCorruptionError("control Hash contains an oversized value")
    try:
        schema_version = _nonnegative_int(decoded["schema_version"], "schema version")
    except ExecutionStoreCorruptionError as error:
        raise ExecutionStoreCorruptionError("control Hash has an invalid schema version") from error
    if schema_version != STORAGE_SCHEMA_VERSION:
        raise ExecutionStoreCorruptionError(f"control Hash has unsupported schema version {schema_version}")
    try:
        status = ExecutionStatus(decoded["status"])
    except ValueError as error:
        raise ExecutionStoreCorruptionError("control Hash has an unknown status") from error

    numeric = {
        name: (_nonnegative_int(decoded[name], name) if name in decoded else None) for name in _INTEGER_CONTROL_FIELDS
    }
    try:
        control = ExecutionControl(
            run_id=decoded["run_id"],
            idempotency_digest=decoded["idempotency_digest"],
            idempotency_binding_digest=decoded["idempotency_binding_digest"],
            deployment=decoded["deployment"],
            definition_revision=decoded["definition_revision"],
            owner_id=decoded.get("owner_id"),
            kind=decoded["kind"],
            status=status,
            lease_owner=decoded.get("lease_owner"),
            cancel_reason=decoded.get("cancel_reason"),
            **cast(Any, numeric),
        )
    except (TypeError, ValueError) as error:
        raise ExecutionStoreCorruptionError("control Hash violates durable invariants") from error
    if control.run_id != expected_run_id:
        raise ExecutionStoreCorruptionError("control Hash belongs to another execution")
    if (
        control.version < 1
        or control.created_at_ms > control.updated_at_ms
        or (control.available_at_ms is not None and control.status is not ExecutionStatus.QUEUED)
        or (control.cancel_requested_at_ms is not None and control.cancel_requested_at_ms > control.updated_at_ms)
        or (control.lease_expires_at_ms is not None and control.lease_expires_at_ms <= control.updated_at_ms)
    ):
        raise ExecutionStoreCorruptionError("control Hash contains contradictory values")
    return control


class RedisExecutionStore:
    """
    Cross-process durable storage using Lua-guarded Redis commits.

    Blocking stream reads hold a connection for up to their timeout, so hosts
    that serve more than one viewer pass a separate ``viewer_client`` to keep
    viewers from starving worker heartbeats of connections.
    """

    def __init__(
        self,
        redis: Any,
        deployment: str,
        *,
        viewer_client: Any | None = None,
        config: StoreConfig | None = None,
        key_prefix: str = "hayhooks:durable",
        transaction_retries: int = _DEFAULT_TRANSACTION_RETRIES,
        transaction_backoff_ms: int = _DEFAULT_TRANSACTION_BACKOFF_MS,
    ) -> None:
        if transaction_retries < 1 or transaction_backoff_ms < 0:
            raise ValueError("transaction retries must be positive and backoff cannot be negative")
        viewer = redis if viewer_client is None else viewer_client
        for client in (redis, viewer):
            pool = getattr(client, "connection_pool", None)
            encoder = pool.get_encoder() if pool is not None and hasattr(pool, "get_encoder") else None
            if getattr(encoder, "decode_responses", False) is True:
                raise ValueError("Redis durable storage requires decode_responses=False")
        self.redis = redis
        self.viewer = viewer
        self.deployment = deployment
        self.config = config or StoreConfig()
        self.keys = RedisKeys(key_prefix, deployment)
        self._transaction_retries = transaction_retries
        self._transaction_backoff_ms = transaction_backoff_ms
        self._apply_script = redis.register_script(_OWNED_LUA + _APPLY_LUA)
        self._heartbeat_script = redis.register_script(_OWNED_LUA + _HEARTBEAT_LUA)
        self._append_chunks_script = redis.register_script(_OWNED_LUA + _APPEND_CHUNKS_LUA)

    async def initialize(self) -> None:
        with _redis_errors():
            info = await self.redis.info("server")
        try:
            raw_version = info.get("redis_version", info.get(b"redis_version"))
            version = tuple(int(piece) for piece in _text(raw_version).split(".")[:2])
        except (AttributeError, TypeError, ValueError) as error:
            raise ExecutionStoreError("unable to validate Redis server capabilities") from error
        # Valkey reports a compatible redis_version (7.2.4) next to its own valkey_version.
        if version < (6, 2):
            raise ExecutionStoreError("durable Redis requires Redis 6.2 or newer, or Valkey 7.2 or newer")

    async def submit(self, control: ExecutionControl, input_payload: bytes) -> SubmissionResult:
        if control.deployment != self.deployment:
            raise ValueError("control deployment does not match this store")
        validate_payload_size("input", input_payload, self.config.max_payload_bytes)
        idempotency_key = self.keys.idempotency(control.idempotency_digest)
        control_key = self.keys.control(control.run_id)
        with _redis_errors():
            for attempt in range(self._transaction_retries):
                async with self.redis.pipeline(transaction=True) as pipe:
                    try:
                        watch_keys = [idempotency_key, control_key]
                        if self.config.max_nonterminal_executions:
                            watch_keys.append(self.keys.capacity)
                        await pipe.watch(*watch_keys)
                        binding_values = await pipe.hgetall(idempotency_key)
                        if binding_values:
                            try:
                                binding = {_text(key): _text(value) for key, value in binding_values.items()}
                            except (TypeError, UnicodeError) as error:
                                raise ExecutionStoreCorruptionError(
                                    "idempotency binding contains invalid UTF-8"
                                ) from error
                            if binding.keys() != {"run_id", "binding"}:
                                raise ExecutionStoreCorruptionError("idempotency binding has invalid fields")
                            if not binding["binding"] or len(binding["binding"].encode()) > MAX_CONTROL_SCALAR_BYTES:
                                raise ExecutionStoreCorruptionError("idempotency binding has an invalid digest")
                            try:
                                mapped_key = self.keys.control(binding["run_id"])
                            except ValueError as error:
                                raise ExecutionStoreCorruptionError(
                                    "idempotency binding has an invalid execution ID"
                                ) from error
                            await pipe.watch(mapped_key)
                            current_values = await pipe.hgetall(mapped_key)
                            if binding["binding"] != control.idempotency_binding_digest:
                                raise ExecutionIdempotencyConflictError("idempotency key is bound to different work")
                            if not current_values:
                                raise ExecutionStoreCorruptionError("idempotency binding points to a missing execution")
                            return SubmissionResult(
                                created=False, control=self._decode(current_values, binding["run_id"])
                            )
                        if await pipe.exists(control_key):
                            raise ExecutionIdempotencyConflictError("run ID is bound to a different idempotency key")
                        if self.config.max_nonterminal_executions:
                            raw_count = await pipe.hget(self.keys.capacity, "nonterminal")
                            count = 0 if raw_count is None else _nonnegative_int(raw_count, "nonterminal")
                            if count >= self.config.max_nonterminal_executions:
                                raise ExecutionAdmissionError("nonterminal execution limit reached")

                        now_ms = _milliseconds(await pipe.time())
                        candidate = replace(control, created_at_ms=now_ms, updated_at_ms=now_ms)
                        plan = submission_plan(candidate, input_payload)
                        pipe.multi()
                        pipe.hset(
                            idempotency_key,
                            mapping={"run_id": candidate.run_id, "binding": candidate.idempotency_binding_digest},
                        )
                        for command in self._plan_commands(candidate, plan, new_submission=True):
                            pipe.execute_command(*command)
                        await pipe.execute()
                        log.bind(run_id=candidate.run_id, deployment=candidate.deployment).debug(
                            "Submitted durable execution"
                        )
                        return SubmissionResult(created=True, control=candidate)
                    except WatchError:
                        await self._backoff(attempt)
        raise ExecutionContentionError("submission transaction retry budget exhausted")

    async def read(self, run_id: str) -> StoredExecution | None:
        return await self._read(run_id, private=True)

    async def read_public(self, run_id: str) -> StoredExecution | None:
        return await self._read(run_id, private=False)

    async def transition(self, run_id: str, command: ExecutionCommand) -> TransitionPlan:
        with _redis_errors():
            plan = await self._transition(run_id, command)
        assert plan is not None
        return plan

    async def claim(self, command: Claim) -> TransitionPlan | None:
        if command.lease_duration_ms <= self.config.lease_commit_safety_ms:
            raise ValueError("lease duration must exceed the commit safety margin")
        candidate_index = self.keys.runnable_revision(command.worker_revision)
        with _redis_errors():
            # A head that another worker claimed first is dropped from the index, so try the next one.
            for _ in range(self._transaction_retries):
                entries = await self.redis.zrange(candidate_index, 0, 0, withscores=True)
                if not entries:
                    return None
                try:
                    member, raw_score = entries[0]
                    run_id = _text(member)
                    validate_run_id(run_id)
                    available_at_ms = _index_score_ms(raw_score, "runnable score")
                except (TypeError, UnicodeError, ValueError, ExecutionStoreCorruptionError) as error:
                    raise ExecutionStoreCorruptionError("runnable index contains an invalid member or score") from error
                now_ms = _milliseconds(await self.redis.time())
                if available_at_ms > now_ms:
                    return None
                plan = await self._transition(run_id, command, candidate_index=candidate_index)
                if plan is not None:
                    return plan
            return None

    async def maintain(
        self,
        *,
        max_run_attempts: int,
        attempts_error: bytes,
    ) -> int:
        requeued = 0
        with _redis_errors():
            entries = await self.redis.zrange(
                self.keys.lease_expiry,
                0,
                MAINTENANCE_BATCH_SIZE - 1,
                withscores=True,
            )
            if not entries:
                return requeued
            valid_entries: list[tuple[str | bytes | int, str, int, int]] = []
            for member, raw_deadline in entries:
                try:
                    run_id, separator, raw_fence = _text(member).rpartition("|")
                    validate_run_id(run_id)
                    if not separator:
                        raise ValueError
                    fence = _nonnegative_int(raw_fence, "fence")
                    deadline = _index_score_ms(raw_deadline, "lease deadline")
                except (TypeError, UnicodeError, ValueError, ExecutionStoreCorruptionError):
                    await self.redis.zrem(self.keys.lease_expiry, member)
                    continue
                valid_entries.append((member, run_id, fence, deadline))
            if not valid_entries:
                return requeued
            now_ms = _milliseconds(await self.redis.time())
            for member, run_id, fence, deadline in valid_entries:
                if deadline > now_ms:
                    break
                try:
                    plan = await self.transition(
                        run_id,
                        RecoverExpiredLease(
                            0,
                            fence,
                            deadline,
                            max_run_attempts,
                            attempts_error,
                        ),
                    )
                except ExecutionNotFoundError:
                    await self.redis.zrem(self.keys.lease_expiry, member)
                except InvalidExecutionTransitionError:
                    continue
                else:
                    requeued += plan.next_control.status is ExecutionStatus.QUEUED
        return requeued

    async def append_chunks(
        self, run_id: str, attempt: int, fence: int, worker_id: str, chunks: Sequence[bytes]
    ) -> None:
        if not self.config.max_stream_chunks:
            return
        if not 0 <= attempt <= _MAX_SAFE_INTEGER:
            raise ValueError("stream chunk attempt must be a non-negative safe integer")
        for data in chunks:
            validate_payload_size("stream chunk", data, self.config.max_stream_chunk_bytes)
        with _redis_errors():
            appended = await self._append_chunks_script(
                keys=[self.keys.control(run_id), self.keys.chunks(run_id)],
                args=[
                    worker_id,
                    fence,
                    attempt,
                    self.config.lease_commit_safety_ms,
                    self.config.max_stream_chunks,
                    *chunks,
                ],
            )
        if not appended:
            raise ExecutionLeaseLostError("execution is no longer owned by this worker fence")

    async def read_chunks(self, run_id: str, after: str) -> tuple[StreamChunk, ...]:
        validate_run_id(run_id)
        parse_chunk_cursor(after)
        count = chunk_read_count(self.config)
        with _redis_errors():
            if after == CHUNK_CURSOR_START:
                entries = await self.redis.xrange(self.keys.chunks(run_id), min="-", max="+", count=count)
            else:
                entries = await self.redis.xrange(
                    self.keys.chunks(run_id),
                    min=after,
                    max="+",
                    count=count + 1,
                )
                if not entries or _text(entries[0][0]) != after:
                    raise ChunkCursorExpiredError(after)
                entries = entries[1:]
        return self._decode_chunks(entries)

    async def wait_chunks(self, run_id: str, after: str, timeout: float) -> tuple[StreamChunk, ...]:
        validate_run_id(run_id)
        parse_chunk_cursor(after)
        if not timeout > 0:
            raise ValueError("chunk wait timeout must be positive")
        key = self.keys.chunks(run_id)
        with _redis_errors():
            async with self.viewer.pipeline(transaction=False) as pipe:
                pipe.xread({key: after}, count=chunk_read_count(self.config), block=math.ceil(timeout * 1_000))
                # Commands after a blocked XREAD run when it wakes, covering trimming during the wait.
                if after != CHUNK_CURSOR_START:
                    pipe.xrange(key, min=after, max=after)
                streams, *cursor_check = await pipe.execute()
        if cursor_check and not cursor_check[0]:
            raise ChunkCursorExpiredError(after)
        return self._decode_chunks(streams[0][1] if streams else ())

    async def operational_counts(self) -> dict[str, int]:
        with _redis_errors():
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.hget(self.keys.capacity, "nonterminal")
                pipe.zcard(self.keys.runnable)
                pipe.zcard(self.keys.lease_expiry)
                nonterminal, runnable, lease_expiry = await pipe.execute()
        return {
            "nonterminal": 0 if nonterminal is None else _nonnegative_int(nonterminal, "nonterminal"),
            "runnable": _nonnegative_int(runnable, "runnable"),
            "lease_expiry": _nonnegative_int(lease_expiry, "lease_expiry"),
        }

    async def _read(self, run_id: str, *, private: bool) -> StoredExecution | None:
        kinds = tuple(PayloadKind) if private else PUBLIC_PAYLOAD_KINDS
        with _redis_errors():
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.hgetall(self.keys.control(run_id))
                for kind in kinds:
                    pipe.get(self.keys.payload(run_id, kind))
                pipe.lrange(self.keys.progress(run_id), 0, -1)
                # Collect per-command errors so a missing control wins over wrong-type orphan keys.
                values, *raw_payloads, raw_progress = await pipe.execute(raise_on_error=False)
        if values == {}:
            return None
        if any(isinstance(reply, Exception) for reply in (values, *raw_payloads, raw_progress)):
            raise ExecutionStoreCorruptionError("stored execution keys have invalid types")
        control = self._decode(values, run_id)
        payloads: dict[PayloadKind, bytes] = {}
        for kind, payload in zip(kinds, raw_payloads, strict=True):
            if payload is None:
                continue
            if not isinstance(payload, bytes) or len(payload) > self.config.max_payload_bytes:
                raise ExecutionStoreCorruptionError(f"stored {kind.value} payload is invalid")
            payloads[kind] = payload
        progress = []
        for entry in raw_progress:
            if not isinstance(entry, bytes) or len(entry) < _PROGRESS_SEQUENCE_BYTES:
                raise ExecutionStoreCorruptionError("stored progress event is invalid")
            event = ProgressEvent(
                int.from_bytes(entry[:_PROGRESS_SEQUENCE_BYTES], "big"),
                entry[_PROGRESS_SEQUENCE_BYTES:],
            )
            if event.sequence < 1 or len(event.data) > self.config.max_progress_event_bytes:
                raise ExecutionStoreCorruptionError("stored progress event is invalid")
            progress.append(event)
        sequences = [event.sequence for event in progress]
        if (control.progress_sequence and not progress) or sequences != list(
            range(
                control.progress_sequence - len(progress) + 1,
                control.progress_sequence + 1,
            )
        ):
            raise ExecutionStoreCorruptionError("progress sequence contradicts control state")
        stored = StoredExecution(control, payloads, tuple(progress))
        validate_stored_execution(stored, private=private)
        return stored

    async def _transition(
        self,
        run_id: str,
        command: ExecutionCommand,
        *,
        candidate_index: str | None = None,
    ) -> TransitionPlan | None:
        """
        Reduce a fresh control snapshot and commit it with one guarded script.

        A claim candidate that cannot be claimed has its runnable indexes repaired
        under the same snapshot guard and yields ``None``.
        """
        if isinstance(command, Heartbeat):
            return await self._heartbeat(run_id, command)
        owner = (command.worker_id, command.fence) if isinstance(command, LEASE_COMMANDS) else None
        for attempt in range(self._transaction_retries):
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.hgetall(self.keys.control(run_id))
                pipe.time()
                values, now = await pipe.execute()
            current = self._decode(values, run_id) if values else None
            plan = None
            releases_capacity = False
            try:
                if current is None:
                    raise ExecutionNotFoundError(f"execution '{run_id}' was not found")
                plan = decide(current, bind_store_command(command, _milliseconds(now), self.config))
            except (ExecutionNotFoundError, InvalidExecutionTransitionError):
                if candidate_index is None:
                    raise
                commands = self._runnable_commands(run_id, candidate_index, current)
            else:
                validate_transition_plan(plan, self.config)
                commands = self._plan_commands(current, plan)
                releases_capacity = not current.terminal and plan.next_control.terminal
            outcome = await self._commit(run_id, values, commands, owner=owner, releases_capacity=releases_capacity)
            if outcome == _STALE_SNAPSHOT:
                await self._backoff(attempt)
                continue
            if outcome == _LEASE_LOST:
                raise ExecutionLeaseLostError("execution is no longer owned by this worker fence")
            if outcome == _CAPACITY_UNDERFLOW:
                raise ExecutionStoreCorruptionError("nonterminal execution counter is invalid or would underflow")
            if (
                plan is not None
                and current is not None
                and (
                    plan.next_control != current
                    or plan.payload_writes
                    or plan.payload_deletes
                    or plan.progress_events
                    or plan.lease_index_update
                )
            ):
                log.bind(
                    run_id=run_id,
                    command=type(command).__name__,
                    from_status=current.status.value,
                    to_status=plan.next_control.status.value,
                    version=plan.next_control.version,
                    fence=plan.next_control.fence,
                ).debug("Committed durable execution transition")
            return plan
        raise ExecutionContentionError("execution transaction retry budget exhausted")

    async def _heartbeat(self, run_id: str, command: Heartbeat) -> TransitionPlan:
        """Renew a lease inside the owned guard; only the deadline and its index change."""
        bind_store_command(command, 0, self.config)
        values = await self._heartbeat_script(
            keys=[self.keys.control(run_id), self.keys.lease_expiry],
            args=[
                command.worker_id,
                command.fence,
                self.config.lease_commit_safety_ms,
                command.lease_duration_ms,
                RedisKeys.lease_member(run_id, command.fence),
            ],
        )
        if values is None:
            raise ExecutionLeaseLostError("execution is no longer owned by this worker fence")
        control = self._decode(dict(zip(values[::2], values[1::2], strict=True)), run_id)
        return TransitionPlan(control, lease_index_update=LeaseIndexUpdate(control.lease_expires_at_ms, control.fence))

    async def _commit(
        self,
        run_id: str,
        snapshot: Mapping[bytes, bytes],
        commands: Sequence[_Command],
        *,
        owner: tuple[str, int] | None,
        releases_capacity: bool,
    ) -> int:
        """Run ``commands`` only if control still equals ``snapshot`` and every guard holds."""
        keys = [self.keys.control(run_id), self.keys.capacity]
        worker_id, fence = owner or ("", 0)
        args: list[Any] = [
            worker_id,
            fence,
            self.config.lease_commit_safety_ms,
            int(releases_capacity),
            len(snapshot),
            *chain.from_iterable(snapshot.items()),
        ]
        for name, key, *arguments in commands:
            if key not in keys:
                keys.append(key)
            args.extend((len(arguments), name, keys.index(key) + 1, *arguments))
        return int(await self._apply_script(keys=keys, args=args))

    def _plan_commands(
        self,
        current: ExecutionControl,
        plan: TransitionPlan,
        *,
        new_submission: bool = False,
    ) -> list[_Command]:
        """Translate one reducer plan into the Redis commands that persist it."""
        control = plan.next_control
        run_id = control.run_id
        control_key = self.keys.control(run_id)
        progress_key = self.keys.progress(run_id)
        next_fields = encode_control(control)
        commands: list[_Command] = [("HSET", control_key, *chain.from_iterable(next_fields.items()))]
        if removed_fields := encode_control(current).keys() - next_fields.keys():
            commands.append(("HDEL", control_key, *removed_fields))
        commands.extend(("SET", self.keys.payload(run_id, write.kind), write.data) for write in plan.payload_writes)
        commands.extend(("DEL", self.keys.payload(run_id, kind)) for kind in plan.payload_deletes)
        if plan.progress_events:
            entries = [
                event.sequence.to_bytes(_PROGRESS_SEQUENCE_BYTES, "big") + event.data for event in plan.progress_events
            ]
            # Lua's unpack() caps a script command at about 8,000 arguments.
            commands.extend(
                ("RPUSH", progress_key, *entries[start : start + _MAX_COMMAND_VALUES])
                for start in range(0, len(entries), _MAX_COMMAND_VALUES)
            )
            commands.append(("LTRIM", progress_key, -self.config.max_progress_events, -1))
        commands.extend(
            self._runnable_commands(run_id, self.keys.runnable_revision(current.definition_revision), control)
        )
        if (lease := plan.lease_index_update) is not None:
            member = RedisKeys.lease_member(run_id, lease.fence)
            commands.append(
                ("ZREM", self.keys.lease_expiry, member)
                if lease.deadline_ms is None
                else ("ZADD", self.keys.lease_expiry, lease.deadline_ms, member)
            )

        if new_submission:
            commands.append(("HINCRBY", self.keys.capacity, "nonterminal", 1))
        elif not current.terminal and control.terminal:
            chunks_key = self.keys.chunks(run_id)
            commands.append(("HINCRBY", self.keys.capacity, "nonterminal", -1))
            # The marker wakes blocked stream viewers; it is written even when chunk persistence is disabled.
            commands.append(
                (
                    "XADD",
                    chunks_key,
                    "MAXLEN",
                    max(self.config.max_stream_chunks, 1),
                    "*",
                    "attempt",
                    control.run_attempt,
                    "terminal",
                    control.status.value,
                )
            )
            commands.extend(
                ("EXPIRE", key, self.config.terminal_ttl_seconds)
                for key in (
                    control_key,
                    progress_key,
                    chunks_key,
                    *(self.keys.payload(run_id, kind) for kind in PayloadKind),
                    self.keys.idempotency(control.idempotency_digest),
                )
            )
        return commands

    def _runnable_commands(
        self,
        run_id: str,
        indexed_revision_key: str,
        control: ExecutionControl | None,
    ) -> list[_Command]:
        """Drop ``run_id`` from its runnable indexes and re-add it when ``control`` is queued."""
        commands: list[_Command] = [("ZREM", self.keys.runnable, run_id), ("ZREM", indexed_revision_key, run_id)]
        if control is not None and control.status is ExecutionStatus.QUEUED:
            score = runnable_score(control)
            commands.append(("ZADD", self.keys.runnable, score, run_id))
            commands.append(("ZADD", self.keys.runnable_revision(control.definition_revision), score, run_id))
        return commands

    def _decode(self, values: Mapping[str | bytes, str | bytes | int], run_id: str) -> ExecutionControl:
        control = decode_control(values, expected_run_id=run_id)
        if control.deployment != self.deployment:
            raise ExecutionStoreCorruptionError("control belongs to another deployment")
        return control

    def _decode_chunks(self, entries: Iterable[tuple[Any, Mapping[Any, Any]]]) -> tuple[StreamChunk, ...]:
        chunks = []
        for entry_id, raw_fields in entries:
            try:
                values = {_text(key): value for key, value in raw_fields.items()}
                attempt = _nonnegative_int(values.pop("attempt"), "stream chunk attempt")
                if values.keys() == {"terminal"}:
                    chunks.append(StreamChunk(_text(entry_id), attempt, b"", terminal=True))
                    continue
                if (
                    values.keys() != {"data"}
                    or not isinstance(values["data"], bytes)
                    or len(values["data"]) > self.config.max_stream_chunk_bytes
                ):
                    raise ValueError
                chunks.append(StreamChunk(_text(entry_id), attempt, values["data"]))
            except (KeyError, TypeError, UnicodeError, ValueError) as error:
                raise ExecutionStoreCorruptionError("stream chunk entry is invalid") from error
        return tuple(chunks)

    async def _backoff(self, attempt: int) -> None:
        if attempt + 1 < self._transaction_retries and self._transaction_backoff_ms:
            await asyncio.sleep(random.uniform(0, self._transaction_backoff_ms) / 1_000)  # noqa: S311


def _milliseconds(redis_time: tuple[int, int]) -> int:
    seconds, microseconds = redis_time
    return int(seconds) * 1_000 + int(microseconds) // 1_000


def _text(value: str | bytes | int | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str | int):
        return str(value)
    raise TypeError("Redis value is not text")


def _nonnegative_int(value: str | bytes | int, name: str) -> int:
    try:
        raw = _text(value)
        if not raw.isascii() or not raw.isdecimal():
            raise ValueError
        parsed = int(raw)
    except (TypeError, UnicodeError, ValueError) as error:
        raise ExecutionStoreCorruptionError(f"{name} is not a non-negative integer") from error
    if parsed > _MAX_SAFE_INTEGER:
        raise ExecutionStoreCorruptionError(f"{name} exceeds Redis's safe integer range")
    return parsed


def _index_score_ms(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or int(value) != value:
        raise ExecutionStoreCorruptionError(f"{name} is not an integer millisecond timestamp")
    return _nonnegative_int(str(int(value)), name)


@contextmanager
def _redis_errors() -> Iterator[None]:
    try:
        yield
    except RedisError as error:
        raise ExecutionStoreError("Redis durable store operation failed") from error


__all__ = ["RedisExecutionStore", "decode_control", "encode_control"]
