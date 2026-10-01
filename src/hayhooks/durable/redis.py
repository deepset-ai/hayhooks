"""Redis implementation of the durable execution store."""
# ruff: noqa: C901, EM101, EM102, PLR0913

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
    from redis.exceptions import RedisError, ResponseError
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
    MAINTENANCE_MAX_BATCHES,
    PUBLIC_PAYLOAD_KINDS,
    ChunkCursorExpiredError,
    ExecutionAdmissionError,
    ExecutionContentionError,
    ExecutionIdempotencyConflictError,
    ExecutionProgressCorruptionError,
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
    "lease_recoveries",
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
_CLAIM_CANDIDATES = 8

# Refusals of the guarded apply script, which returns 1 once it commits.
_STALE_SNAPSHOT, _LEASE_LOST, _CAPACITY_UNDERFLOW = 0, -1, -2
_SUBMITTED, _REPLAYED, _RUN_ID_TAKEN, _ADMISSION_FULL, _CAPACITY_INVALID = 1, 2, 3, 4, 5

# A Redis command as (name, key, *arguments); scripts receive the key through KEYS.
_Command = tuple[Any, ...]


class _UndecodableControlError(ExecutionStoreCorruptionError):
    """An execution control that scheduling can isolate from the shared indexes."""


# Returns Redis TIME in milliseconds while the worker still owns the lease with room for the safety margin.
_OWNED_LUA = """
local function owned_now(control, worker, fence, margin)
  local lease = redis.call('HMGET', control, 'status', 'lease_owner', 'fence', 'lease_expires_at_ms', 'updated_at_ms')
  if lease[1] ~= 'running' or lease[2] ~= worker or lease[3] ~= fence then
    return nil
  end
  local time = redis.call('TIME')
  local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
  if now < tonumber(lease[4]) - tonumber(margin) then
    return now, tonumber(lease[5])
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
local now, updated = owned_now(KEYS[1], ARGV[1], ARGV[2], ARGV[3])
if not now then
  return false
end
local deadline = string.format('%d', math.max(now, updated) + tonumber(ARGV[4]))
-- A script error does not roll back writes, so validate the index before renewing control.
redis.call('ZCARD', KEYS[2])
redis.call('HSET', KEYS[1], 'lease_expires_at_ms', deadline)
redis.call('ZADD', KEYS[2], deadline, ARGV[5])
return redis.call('HGETALL', KEYS[1])
"""

# KEYS: control, capacity, then every key the commands touch.
# ARGV: worker ('' when unowned), fence, safety margin, released revision field ('' when none), snapshot field count,
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
if ARGV[4] ~= '' then
  local counts = redis.call('HMGET', KEYS[2], 'nonterminal', ARGV[4])
  for index = 1, 2 do
    local count = counts[index]
    if not count or not string.match(count, '^[1-9]%d*$') or tonumber(count) > 2^53 - 1 then
      return -2
    end
  end
end
-- Check each type-sensitive target before any mutation: Redis scripts have no rollback.
-- HGETALL and HGET above already validate the control and capacity hashes.
local checks = {RPUSH = 'LLEN', ZADD = 'ZCARD', ZREM = 'ZCARD'}
local checked = {}
local reset_streams = {}
local probe = cursor
while probe <= #ARGV do
  local name = ARGV[probe + 1]
  local key = KEYS[tonumber(ARGV[probe + 2])]
  if name == 'XADD' and not checked[key] then
    local kind = redis.call('TYPE', key)
    kind = type(kind) == 'table' and kind.ok or kind
    if kind ~= 'none' and kind ~= 'stream' then
      reset_streams[key] = true
    end
    checked[key] = true
  elseif checks[name] and not checked[key] then
    redis.call(checks[name], key)
    checked[key] = true
  end
  probe = probe + 3 + tonumber(ARGV[probe])
end
for key in pairs(reset_streams) do
  redis.call('DEL', key)
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

# KEYS: binding, control, input, revision runnable index, capacity.
# ARGV: limit, revision count field, input, run ID, binding digest, score, then control pairs.
_SUBMIT_LUA = """
local binding = redis.call('HGETALL', KEYS[1])
if #binding > 0 then
  return {2, binding}
end
if redis.call('EXISTS', KEYS[2]) == 1 then
  return {3}
end
local counts = redis.call('HMGET', KEYS[5], 'nonterminal', ARGV[2])
for index = 1, 2 do
  local count = counts[index]
  if count and ((count ~= '0' and not string.match(count, '^[1-9]%d*$')) or tonumber(count) >= 2^53 - 1) then
    return {5}
  end
end
local limit = tonumber(ARGV[1])
if limit > 0 and tonumber(counts[1] or '0') >= limit then
  return {4}
end
redis.call('ZCARD', KEYS[4])
local time = redis.call('TIME')
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
local stamp = string.format('%d', now)
redis.call('HSET', KEYS[1], 'run_id', ARGV[4], 'binding', ARGV[5])
redis.call('HSET', KEYS[2], 'created_at_ms', stamp, 'updated_at_ms', stamp, unpack(ARGV, 7))
redis.call('SET', KEYS[3], ARGV[3])
redis.call('ZADD', KEYS[4], ARGV[6] ~= '' and ARGV[6] or stamp, ARGV[4])
redis.call('HINCRBY', KEYS[5], 'nonterminal', 1)
redis.call('HINCRBY', KEYS[5], ARGV[2], 1)
return {1, stamp}
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

    def runnable_revision(self, revision: str) -> str:
        return f"{self.base}:runnable:{self._revision_digest(revision)}"

    @staticmethod
    def revision_nonterminal_field(revision: str) -> str:
        return f"nonterminal:{RedisKeys._revision_digest(revision)}"

    @staticmethod
    def _revision_digest(revision: str) -> str:
        return hashlib.sha256(b"hayhooks-durable:revision:" + revision.encode()).hexdigest()

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
            _validate_client(client)
        self.redis = redis
        self.viewer = viewer
        self.deployment = deployment
        self.config = config or StoreConfig()
        self.keys = RedisKeys(key_prefix, deployment)
        self._transaction_retries = transaction_retries
        self._transaction_backoff_ms = transaction_backoff_ms
        self._submit_script = redis.register_script(_SUBMIT_LUA)
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
        submission_plan(control, input_payload)
        encoded = encode_control(control)
        del encoded["created_at_ms"], encoded["updated_at_ms"]
        keys = [
            self.keys.idempotency(control.idempotency_digest),
            self.keys.control(control.run_id),
            self.keys.payload(control.run_id, PayloadKind.INPUT),
            self.keys.runnable_revision(control.definition_revision),
            self.keys.capacity,
        ]
        args = [
            self.config.max_nonterminal_executions,
            RedisKeys.revision_nonterminal_field(control.definition_revision),
            input_payload,
            control.run_id,
            control.idempotency_binding_digest,
            "" if control.available_at_ms is None else control.available_at_ms,
            *chain.from_iterable(encoded.items()),
        ]
        with _redis_errors():
            for attempt in range(self._transaction_retries):
                status, *reply = await self._submit_script(keys=keys, args=args)
                if status == _SUBMITTED:
                    now_ms = int(reply[0])
                    candidate = replace(control, created_at_ms=now_ms, updated_at_ms=now_ms)
                    log.bind(run_id=candidate.run_id, deployment=candidate.deployment).debug(
                        "Submitted durable execution"
                    )
                    return SubmissionResult(created=True, control=candidate)
                if status == _RUN_ID_TAKEN:
                    raise ExecutionIdempotencyConflictError("run ID is bound to a different idempotency key")
                if status == _ADMISSION_FULL:
                    raise ExecutionAdmissionError("nonterminal execution limit reached")
                if status == _CAPACITY_INVALID:
                    raise ExecutionStoreCorruptionError("nonterminal execution counter is invalid")
                if status != _REPLAYED:
                    raise ExecutionStoreCorruptionError("submission script returned an invalid status")
                run_id = self._bound_run_id(reply[0], control)
                if values := await self.redis.hgetall(self.keys.control(run_id)):
                    return SubmissionResult(created=False, control=self._decode(values, run_id))
                await self._backoff(attempt)
        raise ExecutionStoreCorruptionError("idempotency binding points to a missing execution")

    async def read(self, run_id: str) -> StoredExecution | None:
        return await self._read(run_id, private=True)

    async def read_public(self, run_id: str) -> StoredExecution | None:
        return await self._read(run_id, private=False)

    async def read_control(self, run_id: str) -> ExecutionControl | None:
        with _redis_errors():
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.hgetall(self.keys.control(run_id))
                (values,) = await pipe.execute(raise_on_error=False)
            return self._control(values, run_id)

    async def transition(self, run_id: str, command: ExecutionCommand) -> TransitionPlan:
        with _redis_errors():
            _, plan = await self._transition(run_id, command)
        assert plan is not None
        return plan

    async def claim(self, command: Claim) -> TransitionPlan | None:
        if command.lease_duration_ms <= self.config.lease_commit_safety_ms:
            raise ValueError("lease duration must exceed the commit safety margin")
        candidate_index = self.keys.runnable_revision(command.worker_revision)
        with _redis_errors():
            for _ in range(self._transaction_retries):
                entries, now_ms = await self._scan(candidate_index, _CLAIM_CANDIDATES)
                due: list[str] = []
                invalid = []
                for member, raw_score in entries:
                    try:
                        run_id = _text(member)
                        validate_run_id(run_id)
                        available_at_ms = _index_score_ms(raw_score, "runnable score")
                    except (TypeError, UnicodeError, ValueError, ExecutionStoreCorruptionError):
                        invalid.append(member)
                        continue
                    if available_at_ms <= now_ms:
                        due.append(run_id)
                if invalid:
                    await self.redis.zrem(candidate_index, *invalid)
                    log.bind(operation="claim", entries=len(invalid)).error(
                        "Removed invalid entries from a durable scheduling index"
                    )
                if not due:
                    if invalid:
                        continue
                    return None
                run_id = random.choice(due)  # noqa: S311
                try:
                    _, plan = await self._transition(run_id, command, candidate_index=candidate_index)
                except _UndecodableControlError as error:
                    await self.redis.zrem(candidate_index, run_id)
                    log.bind(run_id=run_id, operation="claim", error=str(error)).error(
                        "Removed an undecodable durable execution from the runnable index"
                    )
                    continue
                if plan is not None:
                    return plan
            return None

    async def _scan(self, index: str, count: int) -> tuple[list[tuple[Any, Any]], int]:
        """Read the earliest index entries and Redis time in one round trip."""
        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.zrange(index, 0, count - 1, withscores=True)
            pipe.time()
            entries, now = await pipe.execute()
        return entries, _milliseconds(now)

    async def maintain(
        self,
        *,
        max_run_attempts: int,
        attempts_error: bytes,
    ) -> int:
        requeued = 0
        with _redis_errors():
            for _ in range(MAINTENANCE_MAX_BATCHES):
                entries, now_ms = await self._scan(self.keys.lease_expiry, MAINTENANCE_BATCH_SIZE)
                due: list[tuple[Any, str, int, int]] = []
                invalid = []
                for member, raw_deadline in entries:
                    try:
                        run_id, separator, raw_fence = _text(member).rpartition("|")
                        validate_run_id(run_id)
                        if not separator:
                            raise ValueError
                        fence = _nonnegative_int(raw_fence, "fence")
                        deadline = _index_score_ms(raw_deadline, "lease deadline")
                    except (TypeError, UnicodeError, ValueError, ExecutionStoreCorruptionError):
                        invalid.append(member)
                        continue
                    if deadline <= now_ms:
                        due.append((member, run_id, fence, deadline))
                if invalid:
                    await self.redis.zrem(self.keys.lease_expiry, *invalid)
                    log.bind(operation="maintenance", entries=len(invalid)).error(
                        "Removed invalid entries from a durable scheduling index"
                    )
                random.shuffle(due)
                for member, run_id, fence, deadline in due:
                    try:
                        current, plan = await self._transition(
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
                    except _UndecodableControlError as error:
                        await self.redis.zrem(self.keys.lease_expiry, member)
                        log.bind(run_id=run_id, operation="maintenance", error=str(error)).error(
                            "Removed an undecodable durable execution from the lease index"
                        )
                    else:
                        assert current is not None and plan is not None
                        requeued += (
                            current.status is ExecutionStatus.RUNNING
                            and plan.next_control.status is ExecutionStatus.QUEUED
                        )
                if len(entries) < MAINTENANCE_BATCH_SIZE or len(due) + len(invalid) < len(entries):
                    break
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
        return self._decode_chunks(run_id, entries)

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
        return self._decode_chunks(run_id, streams[0][1] if streams else ())

    async def operational_counts(self, *, revision: str) -> dict[str, int]:
        with _redis_errors():
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.hmget(
                    self.keys.capacity,
                    "nonterminal",
                    RedisKeys.revision_nonterminal_field(revision),
                )
                pipe.zcard(self.keys.runnable_revision(revision))
                pipe.zcard(self.keys.lease_expiry)
                (nonterminal, revision_nonterminal), revision_runnable, lease_expiry = await pipe.execute()
        return {
            "nonterminal": 0 if nonterminal is None else _nonnegative_int(nonterminal, "nonterminal"),
            "revision_nonterminal": (
                0 if revision_nonterminal is None else _nonnegative_int(revision_nonterminal, "revision nonterminal")
            ),
            "revision_runnable": _nonnegative_int(revision_runnable, "revision runnable"),
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
        replies = (values, *raw_payloads, raw_progress)
        with _redis_errors():
            if any(isinstance(reply, Exception) and not _wrong_type(reply) for reply in replies):
                raise next(reply for reply in replies if isinstance(reply, Exception) and not _wrong_type(reply))
        if isinstance(values, Exception) or any(isinstance(reply, Exception) for reply in raw_payloads):
            raise ExecutionStoreCorruptionError("stored execution keys have invalid types")
        if isinstance(raw_progress, Exception):
            raise ExecutionProgressCorruptionError("stored progress key has an invalid type")
        control = self._decode(values, run_id)
        payloads: dict[PayloadKind, bytes] = {}
        for kind, payload in zip(kinds, raw_payloads, strict=True):
            if payload is None:
                continue
            if not isinstance(payload, bytes):
                raise ExecutionStoreCorruptionError(f"stored {kind.value} payload is invalid")
            payloads[kind] = payload
        progress = []
        for entry in raw_progress:
            if not isinstance(entry, bytes) or len(entry) < _PROGRESS_SEQUENCE_BYTES:
                raise ExecutionProgressCorruptionError("stored progress event is invalid")
            event = ProgressEvent(
                int.from_bytes(entry[:_PROGRESS_SEQUENCE_BYTES], "big"),
                entry[_PROGRESS_SEQUENCE_BYTES:],
            )
            if event.sequence < 1:
                raise ExecutionProgressCorruptionError("stored progress event is invalid")
            progress.append(event)
        sequences = [event.sequence for event in progress]
        if (control.progress_sequence and not progress) or sequences != list(
            range(
                control.progress_sequence - len(progress) + 1,
                control.progress_sequence + 1,
            )
        ):
            raise ExecutionProgressCorruptionError("progress sequence contradicts control state")
        stored = StoredExecution(control, payloads, tuple(progress))
        validate_stored_execution(stored, private=private)
        return stored

    async def _transition(  # noqa: PLR0912
        self,
        run_id: str,
        command: ExecutionCommand,
        *,
        candidate_index: str | None = None,
    ) -> tuple[ExecutionControl | None, TransitionPlan | None]:
        """
        Reduce a fresh control snapshot and commit it with one guarded script.

        A claim candidate that cannot be claimed has its runnable indexes repaired
        under the same snapshot guard and yields ``None``.
        """
        if isinstance(command, Heartbeat):
            return None, await self._heartbeat(run_id, command)
        owner = (command.worker_id, command.fence) if isinstance(command, LEASE_COMMANDS) else None
        member = (
            RedisKeys.lease_member(run_id, command.indexed_fence) if isinstance(command, RecoverExpiredLease) else None
        )
        for attempt in range(self._transaction_retries):
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.hgetall(self.keys.control(run_id))
                pipe.time()
                if member is not None:
                    pipe.zscore(self.keys.lease_expiry, member)
                values, now, *indexed = await pipe.execute(raise_on_error=False)
            if isinstance(now, Exception):
                raise now
            current = self._control(values, run_id)
            plan = None
            released = ""
            try:
                if current is None:
                    raise ExecutionNotFoundError(f"execution '{run_id}' was not found")
                now_ms = max(_milliseconds(now), current.updated_at_ms)
                plan = decide(current, bind_store_command(command, now_ms, self.config))
            except (ExecutionNotFoundError, InvalidExecutionTransitionError):
                if candidate_index is None:
                    raise
                commands = self._runnable_commands(run_id, candidate_index, current)
            else:
                validate_transition_plan(plan, self.config)
                if _changes_nothing(
                    plan,
                    current,
                    lease_member_indexed=not indexed or indexed[0] is not None,
                ):
                    return current, plan
                commands = self._plan_commands(current, plan)
                if not current.terminal and plan.next_control.terminal:
                    released = RedisKeys.revision_nonterminal_field(current.definition_revision)
            outcome = await self._commit(run_id, values, commands, owner=owner, released=released)
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
                    or plan.discard_progress
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
            return current, plan
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
        released: str,
    ) -> int:
        """Run ``commands`` only if control still equals ``snapshot`` and every guard holds."""
        keys = [self.keys.control(run_id), self.keys.capacity]
        worker_id, fence = owner or ("", 0)
        args: list[Any] = [
            worker_id,
            fence,
            self.config.lease_commit_safety_ms,
            released,
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
        if plan.discard_progress:
            commands.append(("DEL", progress_key))
        elif plan.progress_events:
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

        if not current.terminal and control.terminal:
            chunks_key = self.keys.chunks(run_id)
            commands.append(("HINCRBY", self.keys.capacity, "nonterminal", -1))
            commands.append(
                (
                    "HINCRBY",
                    self.keys.capacity,
                    RedisKeys.revision_nonterminal_field(control.definition_revision),
                    -1,
                )
            )
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
        commands: list[_Command] = [("ZREM", indexed_revision_key, run_id)]
        if control is not None and control.status is ExecutionStatus.QUEUED:
            score = runnable_score(control)
            commands.append(("ZADD", self.keys.runnable_revision(control.definition_revision), score, run_id))
        return commands

    def _decode(self, values: Mapping[str | bytes, str | bytes | int], run_id: str) -> ExecutionControl:
        control = decode_control(values, expected_run_id=run_id)
        if control.deployment != self.deployment:
            raise ExecutionStoreCorruptionError("control belongs to another deployment")
        return control

    def _control(self, values: Any, run_id: str) -> ExecutionControl | None:
        if isinstance(values, Exception):
            if not _wrong_type(values):
                raise values
            raise _UndecodableControlError("control key has an invalid type") from values
        if not values:
            return None
        try:
            return self._decode(values, run_id)
        except ExecutionStoreCorruptionError as error:
            raise _UndecodableControlError(str(error)) from error

    def _bound_run_id(self, values: Any, control: ExecutionControl) -> str:
        try:
            binding = {_text(key): _text(value) for key, value in zip(values[::2], values[1::2], strict=True)}
        except (TypeError, UnicodeError, ValueError) as error:
            raise ExecutionStoreCorruptionError("idempotency binding contains invalid UTF-8") from error
        if binding.keys() != {"run_id", "binding"}:
            raise ExecutionStoreCorruptionError("idempotency binding has invalid fields")
        if not binding["binding"] or len(binding["binding"].encode()) > MAX_CONTROL_SCALAR_BYTES:
            raise ExecutionStoreCorruptionError("idempotency binding has an invalid digest")
        try:
            self.keys.control(binding["run_id"])
        except ValueError as error:
            raise ExecutionStoreCorruptionError("idempotency binding has an invalid execution ID") from error
        if binding["binding"] != control.idempotency_binding_digest:
            raise ExecutionIdempotencyConflictError("idempotency key is bound to different work")
        return binding["run_id"]

    def _decode_chunks(self, run_id: str, entries: Iterable[tuple[Any, Mapping[Any, Any]]]) -> tuple[StreamChunk, ...]:
        chunks = []
        for entry_id, raw_fields in entries:
            cursor = _text(entry_id)
            attempt = 0
            try:
                values = {_text(key): value for key, value in raw_fields.items()}
                attempt = _nonnegative_int(values.pop("attempt"), "stream chunk attempt")
                if values.keys() == {"terminal"}:
                    chunks.append(StreamChunk(cursor, attempt, b"", terminal=True))
                    continue
                if values.keys() != {"data"} or not isinstance(values["data"], bytes):
                    raise ValueError
                chunks.append(StreamChunk(cursor, attempt, values["data"]))
            except (KeyError, TypeError, UnicodeError, ValueError, ExecutionStoreCorruptionError):
                log.bind(run_id=run_id, cursor=cursor).warning("Skipped an undecodable durable stream entry")
                chunks.append(StreamChunk(cursor, attempt, b"", skipped=True))
        return tuple(chunks)

    async def _backoff(self, attempt: int) -> None:
        if attempt + 1 < self._transaction_retries and self._transaction_backoff_ms:
            await asyncio.sleep(random.uniform(0, self._transaction_backoff_ms) / 1_000)  # noqa: S311


def _milliseconds(redis_time: tuple[int, int]) -> int:
    seconds, microseconds = redis_time
    return int(seconds) * 1_000 + int(microseconds) // 1_000


def _validate_client(client: Any) -> None:
    pool = getattr(client, "connection_pool", None)
    encoder = pool.get_encoder() if pool is not None and hasattr(pool, "get_encoder") else None
    if getattr(encoder, "decode_responses", False) is True:
        raise ValueError("Redis durable storage requires decode_responses=False")
    kwargs = getattr(pool, "connection_kwargs", None)
    if not isinstance(kwargs, Mapping):
        return
    if str(kwargs.get("protocol")) == "3" or kwargs.get("legacy_responses") is False:
        raise ValueError("Redis durable storage requires RESP2-shaped replies: leave protocol unset or pass protocol=2")
    retry = kwargs.get("retry")
    if retry is None:
        retries = int(bool(kwargs.get("retry_on_error") or kwargs.get("retry_on_timeout")))
    else:
        retries = retry.get_retries() if hasattr(retry, "get_retries") else retry._retries
    if retries:
        raise ValueError(
            "Redis durable storage requires a client that does not retry commands, because a resent script "
            "can commit twice: build it with Redis.from_url(...) or pass retry=None"
        )


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


def _wrong_type(reply: object) -> bool:
    return isinstance(reply, ResponseError) and str(reply).startswith("WRONGTYPE")


def _changes_nothing(plan: TransitionPlan, current: ExecutionControl, *, lease_member_indexed: bool) -> bool:
    lease = plan.lease_index_update
    return (
        plan.next_control == current
        and not (plan.payload_writes or plan.payload_deletes or plan.progress_events or plan.discard_progress)
        and (lease is None or (lease.deadline_ms is None and not lease_member_indexed))
    )


@contextmanager
def _redis_errors() -> Iterator[None]:
    try:
        yield
    except RedisError as error:
        code = str(error).partition(" ")[0]
        cause = f"{type(error).__name__} {code}" if code.isalpha() and code.isupper() else type(error).__name__
        raise ExecutionStoreError(f"Redis durable store operation failed: {cause}") from error


__all__ = ["RedisExecutionStore", "decode_control", "encode_control"]
