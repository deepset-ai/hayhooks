"""Redis implementation of the durable execution store."""
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0913, PLR0915

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, replace
from typing import Any, cast

from loguru import logger as log

try:
    from redis.exceptions import RedisError, WatchError
except ImportError as error:  # pragma: no cover - exercised by packaging checks
    raise RuntimeError("Redis durable storage requires `hayhooks[durable]`") from error

from hayhooks.durable.engine import (
    MAX_CONTROL_SCALAR_BYTES,
    Claim,
    ExecutionCommand,
    ExecutionControl,
    ExecutionNotFoundError,
    ExecutionStatus,
    Heartbeat,
    InvalidExecutionTransitionError,
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
    MAINTENANCE_BATCH_SIZE,
    MAX_CHUNK_READ_BYTES,
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
    parse_chunk_cursor,
    runnable_score,
    validate_payload_size,
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
    """Cross-process durable storage using optimistic Redis transactions."""

    def __init__(
        self,
        redis: Any,
        deployment: str,
        *,
        config: StoreConfig | None = None,
        key_prefix: str = "hayhooks:durable",
        transaction_retries: int = _DEFAULT_TRANSACTION_RETRIES,
        transaction_backoff_ms: int = _DEFAULT_TRANSACTION_BACKOFF_MS,
    ) -> None:
        if transaction_retries < 1 or transaction_backoff_ms < 0:
            raise ValueError("transaction retries must be positive and backoff cannot be negative")
        pool = getattr(redis, "connection_pool", None)
        encoder = pool.get_encoder() if pool is not None and hasattr(pool, "get_encoder") else None
        if getattr(encoder, "decode_responses", False) is True:
            raise ValueError("Redis durable storage requires decode_responses=False")
        self.redis = redis
        self.deployment = deployment
        self.config = config or StoreConfig()
        self.keys = RedisKeys(key_prefix, deployment)
        self._transaction_retries = transaction_retries
        self._transaction_backoff_ms = transaction_backoff_ms

    async def initialize(self) -> None:
        with _redis_errors():
            info = await self.redis.info("server")
        try:
            raw_version = info.get("redis_version", info.get(b"redis_version"))
            version = tuple(int(piece) for piece in _text(raw_version).split(".")[:2])
        except (AttributeError, TypeError, ValueError) as error:
            raise ExecutionStoreError("unable to validate Redis server capabilities") from error
        if version < (6, 2):
            raise ExecutionStoreError("durable Redis requires Redis 6.2 or later")

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
                            existing = decode_control(current_values, expected_run_id=binding["run_id"])
                            if existing.deployment != self.deployment:
                                raise ExecutionStoreCorruptionError("control belongs to another deployment")
                            return SubmissionResult(created=False, control=existing)
                        if await pipe.exists(control_key):
                            raise ExecutionIdempotencyConflictError("run ID is bound to a different idempotency key")
                        if self.config.max_nonterminal_executions:
                            raw_count = await pipe.hget(self.keys.capacity, "nonterminal")
                            count = 0 if raw_count is None else _nonnegative_int(raw_count, "nonterminal")
                            if count >= self.config.max_nonterminal_executions:
                                raise ExecutionAdmissionError("nonterminal execution limit reached")

                        now_ms = await self._time_ms(pipe)
                        candidate = replace(control, created_at_ms=now_ms, updated_at_ms=now_ms)
                        plan = submission_plan(candidate, input_payload)
                        pipe.multi()
                        pipe.hset(
                            idempotency_key,
                            mapping={"run_id": candidate.run_id, "binding": candidate.idempotency_binding_digest},
                        )
                        self._apply_plan(pipe, candidate, plan, new_submission=True)
                        await pipe.execute()
                        log.bind(run_id=candidate.run_id, deployment=candidate.deployment).debug(
                            "Submitted durable execution"
                        )
                        return SubmissionResult(created=True, control=candidate)
                    except WatchError:
                        await self._backoff(attempt)
        raise ExecutionContentionError("submission transaction retry budget exhausted")

    async def read(self, run_id: str) -> StoredExecution | None:
        control_key = self.keys.control(run_id)
        with _redis_errors():
            for attempt in range(self._transaction_retries):
                async with self.redis.pipeline(transaction=True) as pipe:
                    try:
                        await pipe.watch(control_key)
                        values = await pipe.hgetall(control_key)
                        if not values:
                            return None
                        control = decode_control(values, expected_run_id=run_id)
                        if control.deployment != self.deployment:
                            raise ExecutionStoreCorruptionError("control belongs to another deployment")
                        pipe.multi()
                        for kind in PayloadKind:
                            pipe.get(self.keys.payload(run_id, kind))
                        pipe.lrange(self.keys.progress(run_id), 0, -1)
                        snapshot = await pipe.execute()
                        payloads: dict[PayloadKind, bytes] = {}
                        for kind, payload in zip(PayloadKind, snapshot[:-1], strict=True):
                            if payload is None:
                                continue
                            if not isinstance(payload, bytes) or len(payload) > self.config.max_payload_bytes:
                                raise ExecutionStoreCorruptionError(f"stored {kind.value} payload is invalid")
                            payloads[kind] = payload
                        progress = []
                        for entry in snapshot[-1]:
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
                        return StoredExecution(control, payloads, tuple(progress))
                    except WatchError:
                        await self._backoff(attempt)
        raise ExecutionContentionError("read transaction retry budget exhausted")

    async def transition(self, run_id: str, command: ExecutionCommand) -> TransitionPlan:
        with _redis_errors():
            plan = await self._transition(run_id, command)
        assert plan is not None
        return plan

    async def claim(self, command: Claim) -> TransitionPlan | None:
        if command.lease_duration_ms <= self.config.lease_commit_safety_ms:
            raise ValueError("lease duration must exceed the commit safety margin")
        with _redis_errors():
            now_ms = await self._time_ms(self.redis)
            candidates = await self.redis.zrangebyscore(
                self.keys.runnable,
                "-inf",
                now_ms,
                start=0,
                num=1,
            )
            if not candidates:
                return None
            try:
                run_id = _text(candidates[0])
                validate_run_id(run_id)
            except (TypeError, UnicodeError, ValueError) as error:
                raise ExecutionStoreCorruptionError("runnable index contains an invalid execution ID") from error
            return await self._transition(run_id, command, candidate=True)

    async def maintain(
        self,
        *,
        max_run_attempts: int,
        worker_revision: str,
        revision_error: bytes,
        attempts_error: bytes,
    ) -> int:
        with _redis_errors():
            now_ms = await self._time_ms(self.redis)
            entries = await self.redis.zrangebyscore(
                self.keys.lease_expiry,
                "-inf",
                now_ms,
                start=0,
                num=MAINTENANCE_BATCH_SIZE,
                withscores=True,
            )
            recovered = 0
            for member, raw_deadline in entries:
                try:
                    run_id, separator, raw_fence = _text(member).rpartition("|")
                    validate_run_id(run_id)
                    if not separator or int(raw_deadline) != raw_deadline:
                        raise ValueError
                    fence = _nonnegative_int(raw_fence, "fence")
                    deadline = _nonnegative_int(str(int(raw_deadline)), "lease deadline")
                except (TypeError, UnicodeError, ValueError, ExecutionStoreCorruptionError):
                    await self.redis.zrem(self.keys.lease_expiry, member)
                    continue
                try:
                    await self.transition(
                        run_id,
                        RecoverExpiredLease(
                            0,
                            fence,
                            deadline,
                            max_run_attempts,
                            worker_revision,
                            revision_error,
                            attempts_error,
                        ),
                    )
                    recovered += 1
                except ExecutionNotFoundError:
                    await self.redis.zrem(self.keys.lease_expiry, member)
            return recovered

    async def append_chunk(self, run_id: str, attempt: int, data: bytes) -> None:
        if not self.config.max_stream_chunks:
            return
        validate_run_id(run_id)
        if not 0 <= attempt <= _MAX_SAFE_INTEGER:
            raise ValueError("stream chunk attempt must be a non-negative safe integer")
        validate_payload_size("stream chunk", data, self.config.max_stream_chunk_bytes)
        with _redis_errors():
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.xadd(
                    self.keys.chunks(run_id),
                    {"attempt": attempt, "data": data},
                    maxlen=self.config.max_stream_chunks,
                    approximate=False,
                )
                pipe.expire(self.keys.chunks(run_id), self.config.terminal_ttl_seconds)
                await pipe.execute()

    async def read_chunks(self, run_id: str, after: str) -> tuple[StreamChunk, ...]:
        validate_run_id(run_id)
        parse_chunk_cursor(after)
        count = max(1, MAX_CHUNK_READ_BYTES // self.config.max_stream_chunk_bytes)
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

        chunks = []
        for entry_id, raw_fields in entries:
            try:
                values = {_text(key): value for key, value in raw_fields.items()}
                if (
                    values.keys() != {"attempt", "data"}
                    or not isinstance(values["data"], bytes)
                    or len(values["data"]) > self.config.max_stream_chunk_bytes
                ):
                    raise ValueError
                chunks.append(
                    StreamChunk(
                        _text(entry_id),
                        _nonnegative_int(values["attempt"], "stream chunk attempt"),
                        values["data"],
                    )
                )
            except (KeyError, TypeError, UnicodeError, ValueError) as error:
                raise ExecutionStoreCorruptionError("stream chunk entry is invalid") from error
        return tuple(chunks)

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

    async def _transition(
        self,
        run_id: str,
        command: ExecutionCommand,
        *,
        candidate: bool = False,
    ) -> TransitionPlan | None:
        control_key = self.keys.control(run_id)
        for attempt in range(self._transaction_retries):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(control_key, *((self.keys.runnable,) if candidate else ()))
                    values = await pipe.hgetall(control_key)
                    if not values:
                        if not candidate:
                            raise ExecutionNotFoundError(f"execution '{run_id}' was not found")
                        pipe.multi()
                        pipe.zrem(self.keys.runnable, run_id)
                        await pipe.execute()
                        return None
                    current = decode_control(values, expected_run_id=run_id)
                    if current.deployment != self.deployment:
                        raise ExecutionStoreCorruptionError("control belongs to another deployment")
                    bound = bind_store_command(command, await self._time_ms(pipe), self.config)
                    try:
                        plan = decide(current, bound)
                    except InvalidExecutionTransitionError:
                        if not candidate:
                            raise
                        pipe.multi()
                        pipe.zrem(self.keys.runnable, run_id)
                        if current.status is ExecutionStatus.QUEUED:
                            pipe.zadd(self.keys.runnable, {run_id: runnable_score(current)})
                        await pipe.execute()
                        return None
                    validate_transition_plan(plan, self.config)
                    if not current.terminal and plan.next_control.terminal:
                        await pipe.watch(self.keys.capacity)
                        raw_count = await pipe.hget(self.keys.capacity, "nonterminal")
                        if raw_count is None or _nonnegative_int(raw_count, "nonterminal") < 1:
                            raise ExecutionStoreCorruptionError("nonterminal execution counter would underflow")

                    pipe.multi()
                    if isinstance(bound, Heartbeat):
                        lease = plan.lease_index_update
                        if lease is None or lease.deadline_ms is None:
                            raise AssertionError("heartbeat must renew a lease")
                        pipe.hset(control_key, "lease_expires_at_ms", lease.deadline_ms)
                        pipe.zadd(
                            self.keys.lease_expiry,
                            {RedisKeys.lease_member(run_id, lease.fence): lease.deadline_ms},
                        )
                    else:
                        self._apply_plan(pipe, current, plan)
                    await pipe.execute()
                    if not isinstance(bound, Heartbeat) and (
                        plan.next_control != current
                        or plan.payload_writes
                        or plan.payload_deletes
                        or plan.progress_events
                        or plan.lease_index_update
                    ):
                        log.bind(
                            run_id=run_id,
                            command=type(bound).__name__,
                            from_status=current.status.value,
                            to_status=plan.next_control.status.value,
                            version=plan.next_control.version,
                            fence=plan.next_control.fence,
                        ).debug("Committed durable execution transition")
                    return plan
                except WatchError:
                    await self._backoff(attempt)
        raise ExecutionContentionError("execution transaction retry budget exhausted")

    def _apply_plan(
        self,
        pipe: Any,
        current: ExecutionControl,
        plan: TransitionPlan,
        *,
        new_submission: bool = False,
    ) -> None:
        control = plan.next_control
        current_fields = encode_control(current)
        next_fields = encode_control(control)
        control_key = self.keys.control(control.run_id)
        pipe.hset(control_key, mapping=next_fields)
        removed_fields = current_fields.keys() - next_fields.keys()
        if removed_fields:
            pipe.hdel(control_key, *removed_fields)
        for write in plan.payload_writes:
            pipe.set(self.keys.payload(control.run_id, write.kind), write.data)
        for kind in plan.payload_deletes:
            pipe.delete(self.keys.payload(control.run_id, kind))
        if plan.progress_events:
            pipe.rpush(
                self.keys.progress(control.run_id),
                *(
                    event.sequence.to_bytes(_PROGRESS_SEQUENCE_BYTES, "big") + event.data
                    for event in plan.progress_events
                ),
            )
            pipe.ltrim(self.keys.progress(control.run_id), -self.config.max_progress_events, -1)

        pipe.zrem(self.keys.runnable, control.run_id)
        if control.status is ExecutionStatus.QUEUED:
            pipe.zadd(self.keys.runnable, {control.run_id: runnable_score(control)})
        if (lease := plan.lease_index_update) is not None:
            member = RedisKeys.lease_member(control.run_id, lease.fence)
            if lease.deadline_ms is None:
                pipe.zrem(self.keys.lease_expiry, member)
            else:
                pipe.zadd(self.keys.lease_expiry, {member: lease.deadline_ms})

        if new_submission:
            pipe.hincrby(self.keys.capacity, "nonterminal", 1)
        elif not current.terminal and control.terminal:
            pipe.hincrby(self.keys.capacity, "nonterminal", -1)
            for key in (
                self.keys.control(control.run_id),
                self.keys.progress(control.run_id),
                self.keys.chunks(control.run_id),
                *(self.keys.payload(control.run_id, kind) for kind in PayloadKind),
            ):
                pipe.expire(key, self.config.terminal_ttl_seconds)
            pipe.expire(self.keys.idempotency(control.idempotency_digest), self.config.terminal_ttl_seconds)

    async def _time_ms(self, client: Any) -> int:
        seconds, microseconds = await client.time()
        return int(seconds) * 1_000 + int(microseconds) // 1_000

    async def _backoff(self, attempt: int) -> None:
        if attempt + 1 < self._transaction_retries and self._transaction_backoff_ms:
            await asyncio.sleep(random.uniform(0, self._transaction_backoff_ms) / 1_000)  # noqa: S311


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


@contextmanager
def _redis_errors() -> Iterator[None]:
    try:
        yield
    except RedisError as error:
        raise ExecutionStoreError("Redis durable store operation failed") from error


__all__ = ["RedisExecutionStore", "decode_control", "encode_control"]
