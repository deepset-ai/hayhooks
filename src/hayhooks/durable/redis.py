"""Redis codecs and optimistic transactions for the lean durable engine."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from typing import Any, cast

from hayhooks.durable.backend import (
    DEFAULT_TRANSACTION_BACKOFF_MAX_MS,
    DEFAULT_TRANSACTION_MAX_RETRIES,
    MAINTENANCE_BATCH_SIZE,
    ExecutionContentionError,
    ExecutionIdempotencyConflictError,
    ExecutionStoreConfig,
    ExecutionStoreCorruptionError,
    SubmissionResult,
    bind_command,
    bind_progress_sequences,
    check_admission,
    parse_idempotency_binding,
    parse_lease_member,
    validate_command_payloads,
)
from hayhooks.durable.engine import (
    MAX_CONTROL_SCALAR_BYTES,
    ExecutionCommand,
    ExecutionControl,
    ExecutionNotFoundError,
    ExecutionPayloadSizeError,
    ExecutionStatus,
    Heartbeat,
    InvalidExecutionTransitionError,
    PayloadKind,
    TransitionPlan,
    decide,
    submission_plan,
    validate_run_id,
)

_DIGEST_PREFIX = "hayhooks-durable-v2"


class RedisKeys:
    """Build private keys without exposing deployment or idempotency values."""

    def __init__(self, key_prefix: str, deployment: str) -> None:
        if not key_prefix.rstrip(":"):
            raise ValueError("key_prefix cannot be empty")
        self.deployment_digest = digest("deployment", deployment)
        prefix = key_prefix.rstrip(":")
        self.base = f"{prefix}:{{{self.deployment_digest}}}"

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
        return f"{self._execution_base(run_id)}:control"

    def input(self, run_id: str) -> str:
        return f"{self._execution_base(run_id)}:input"

    def checkpoint(self, run_id: str) -> str:
        return f"{self._execution_base(run_id)}:checkpoint"

    def result(self, run_id: str) -> str:
        return f"{self._execution_base(run_id)}:result"

    def error(self, run_id: str) -> str:
        return f"{self._execution_base(run_id)}:error"

    def progress(self, run_id: str) -> str:
        return f"{self._execution_base(run_id)}:progress"

    def wait(self, run_id: str) -> str:
        return f"{self._execution_base(run_id)}:wait"

    def idempotency(self, idempotency_digest: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{64}", idempotency_digest):
            raise ValueError("idempotency digest must be a sha256 hex value")
        return f"{self.base}:idem:{idempotency_digest}"

    @staticmethod
    def lease_member(run_id: str, fence: int) -> str:
        validate_run_id(run_id)
        if fence < 0:
            raise ValueError("fence cannot be negative")
        return f"{run_id}|{fence}"

    def _execution_base(self, run_id: str) -> str:
        validate_run_id(run_id)
        return f"{self.base}:exec:{run_id}"


def digest(domain: str, value: str) -> str:
    """Return the stable domain-separated digest used for isolated key material."""
    return hashlib.sha256(f"{_DIGEST_PREFIX}:{domain}:".encode() + value.encode()).hexdigest()


def encode_control(control: ExecutionControl) -> dict[str, str]:
    encoded: dict[str, str] = {}
    for field_name, value in asdict(control).items():
        if value is None:
            continue
        encoded[field_name] = value.value if isinstance(value, ExecutionStatus) else str(value)
    return encoded


def decode_control(values: Mapping[str | bytes, str | bytes | int]) -> ExecutionControl:
    decoded = {_text(key): _text(value) for key, value in values.items()}
    required = {
        "run_id",
        "idempotency_digest",
        "idempotency_binding_digest",
        "deployment",
        "definition_revision",
        "kind",
        "status",
        "version",
        "fence",
        "run_attempt",
        "application_retry_count",
        "progress_sequence",
        "created_at_ms",
        "updated_at_ms",
    }
    missing = required.difference(decoded)
    if missing:
        raise ExecutionStoreCorruptionError(f"control Hash is missing required fields: {', '.join(sorted(missing))}")
    try:
        status = ExecutionStatus(decoded["status"])
    except ValueError as error:
        raise ExecutionStoreCorruptionError("control Hash has an unknown status") from error
    for name in (
        "run_id",
        "idempotency_digest",
        "idempotency_binding_digest",
        "deployment",
        "definition_revision",
        "kind",
    ):
        if not decoded[name] or len(decoded[name].encode()) > MAX_CONTROL_SCALAR_BYTES:
            raise ExecutionStoreCorruptionError(f"control Hash has invalid {name}")
    for name in ("owner_id", "lease_owner", "cancel_reason"):
        if name in decoded and len(decoded[name].encode()) > MAX_CONTROL_SCALAR_BYTES:
            raise ExecutionStoreCorruptionError(f"control Hash has oversized {name}")
    integers = (
        "version",
        "fence",
        "run_attempt",
        "application_retry_count",
        "progress_sequence",
        "created_at_ms",
        "updated_at_ms",
    )
    numeric: dict[str, int | None] = {name: _nonnegative_int(decoded[name], name) for name in integers}
    for name in ("available_at_ms", "lease_expires_at_ms", "cancel_requested_at_ms"):
        numeric[name] = _nonnegative_int(decoded[name], name) if name in decoded else None
    try:
        return ExecutionControl(
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
    except (TypeError, ValueError, ExecutionPayloadSizeError) as error:
        raise ExecutionStoreCorruptionError("control Hash violates durable invariants") from error


async def redis_time_ms(pipe: Any) -> int:
    seconds, micros = await pipe.time()
    return int(seconds) * 1_000 + int(micros) // 1_000


class RedisExecutionStore:
    """Redis implementation using one due-time ZSET and one lease ZSET."""

    def __init__(self, redis: Any, *, deployment: str, config: ExecutionStoreConfig | None = None) -> None:
        self.redis = redis
        self.config = config or ExecutionStoreConfig()
        self.keys = RedisKeys(self.config.key_prefix, deployment)
        self.deployment = deployment

    async def initialize(self) -> None:
        try:
            info = await self.redis.info("server")
            version = tuple(int(piece) for piece in _text(info["redis_version"]).split(".")[:2])
        except Exception as error:
            raise RuntimeError("unable to validate Redis server capabilities") from error
        if version < (6, 2):
            raise RuntimeError("durable Redis requires Redis 6.2 or later")

    async def submit(  # noqa: C901
        self, control: ExecutionControl, input_payload: bytes, *, binding_digest: str
    ) -> SubmissionResult:
        if control.deployment != self.deployment:
            raise ValueError("control deployment does not match this store")
        if len(input_payload) > self.config.max_input_bytes:
            raise ExecutionPayloadSizeError("input payload exceeds configured size")
        idem_key = self.keys.idempotency(control.idempotency_digest)
        control_key = self.keys.control(control.run_id)
        for attempt in range(self.config.transaction_max_retries):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    watch_keys = [idem_key, control_key]
                    if self.config.max_nonterminal_executions:
                        watch_keys.append(self.keys.capacity)
                    await pipe.watch(*watch_keys)
                    existing = await pipe.get(idem_key)
                    if existing is not None:
                        try:
                            existing_run, existing_binding = parse_idempotency_binding(_text(existing))
                        except ValueError as error:
                            raise ExecutionStoreCorruptionError("idempotency binding has an invalid format") from error
                        mapped_control_key = self.keys.control(existing_run)
                        await pipe.watch(mapped_control_key)
                        raw_control = await pipe.hgetall(mapped_control_key)
                        if existing_binding != binding_digest:
                            raise ExecutionIdempotencyConflictError("idempotency key is bound to another request")
                        if raw_control:
                            return SubmissionResult(created=False, control=decode_control(raw_control))
                    if self.config.max_nonterminal_executions:
                        check_admission(await pipe.hgetall(self.keys.capacity), self.config)
                    now_ms = await redis_time_ms(pipe)
                    candidate = replace(control, created_at_ms=now_ms, updated_at_ms=now_ms)
                    plan = submission_plan(candidate, input_payload)
                    pipe.multi()
                    if existing is not None:
                        pipe.delete(idem_key)
                    pipe.set(idem_key, f"{candidate.run_id}|{binding_digest}")
                    self._apply_plan(pipe, candidate, plan, new_submission=True)
                    await pipe.execute()
                    return SubmissionResult(created=True, control=candidate)
                except redis_watch_error():
                    await self._backoff(attempt)
        raise ExecutionContentionError("submission transaction retry budget exhausted")

    async def get(self, run_id: str) -> ExecutionControl | None:
        values = await self.redis.hgetall(self.keys.control(run_id))
        return decode_control(values) if values else None

    async def read_payloads(self, run_id: str, kinds: tuple[PayloadKind, ...]) -> dict[PayloadKind, bytes | None]:
        if not kinds:
            return {}
        async with self.redis.pipeline(transaction=False) as pipe:
            for kind in kinds:
                pipe.get(self._payload_key(run_id, kind))
            values = await pipe.execute()
        return {kind: bytes(value) if value is not None else None for kind, value in zip(kinds, values, strict=True)}

    async def read_progress(self, run_id: str) -> list[bytes]:
        return [bytes(value) for value in await self.redis.lrange(self.keys.progress(run_id), 0, -1)]

    async def transition(  # noqa: C901
        self, run_id: str, command: ExecutionCommand, *, candidate: bool = False
    ) -> TransitionPlan:
        validate_command_payloads(command, self.config)
        control_key = self.keys.control(run_id)
        for attempt in range(self.config.transaction_max_retries):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    watch_keys = [control_key]
                    if candidate:
                        watch_keys.append(self.keys.runnable)
                    await pipe.watch(*watch_keys)
                    current_values = await pipe.hgetall(control_key)
                    if not current_values:
                        if candidate:
                            pipe.multi()
                            pipe.zrem(self.keys.runnable, run_id)
                            await pipe.execute()
                        raise ExecutionNotFoundError(f"execution '{run_id}' was not found")
                    current = decode_control(current_values)
                    try:
                        plan = bind_progress_sequences(
                            decide(
                                current,
                                bind_command(
                                    command,
                                    now_ms=await redis_time_ms(pipe),
                                    lease_commit_safety_ms=self.config.lease_commit_safety_ms,
                                ),
                            ),
                            self.config,
                        )
                    except InvalidExecutionTransitionError:
                        if not candidate:
                            raise
                        pipe.multi()
                        pipe.zrem(self.keys.runnable, run_id)
                        if current.status is ExecutionStatus.QUEUED:
                            pipe.zadd(self.keys.runnable, {run_id: _runnable_score(current)})
                        await pipe.execute()
                        return TransitionPlan(current)
                    pipe.multi()
                    if isinstance(command, Heartbeat):
                        lease = plan.lease_index_update
                        if lease is None or lease.deadline_ms is None:
                            raise AssertionError("heartbeat must extend its lease")
                        pipe.hset(control_key, "lease_expires_at_ms", lease.deadline_ms)
                        pipe.zadd(
                            self.keys.lease_expiry,
                            {RedisKeys.lease_member(run_id, lease.fence): lease.deadline_ms},
                        )
                    else:
                        self._apply_plan(pipe, current, plan)
                    await pipe.execute()
                    return plan
                except redis_watch_error():
                    await self._backoff(attempt)
        raise ExecutionContentionError("execution transition retry budget exhausted")

    async def read_candidate(self) -> str | None:
        now_ms = await self._time_ms()
        members = await self.redis.zrangebyscore(self.keys.runnable, "-inf", now_ms, start=0, num=1)
        return _text(members[0]) if members else None

    async def maintain(self, command_factory: Callable[[int, int], ExecutionCommand]) -> int:
        now_ms = await self._time_ms()
        entries = await self.redis.zrangebyscore(
            self.keys.lease_expiry,
            "-inf",
            now_ms,
            start=0,
            num=MAINTENANCE_BATCH_SIZE,
            withscores=True,
        )
        recovered = 0
        for member, deadline in entries:
            try:
                run_id, fence = parse_lease_member(_text(member))
                await self.transition(run_id, command_factory(fence, int(deadline)))
                recovered += 1
            except (ExecutionNotFoundError, ValueError):
                await self.redis.zrem(self.keys.lease_expiry, member)
        return recovered

    async def operational_counts(self) -> dict[str, int]:
        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.hget(self.keys.capacity, "nonterminal")
            pipe.zcard(self.keys.runnable)
            pipe.zcard(self.keys.lease_expiry)
            nonterminal, runnable, leases = await pipe.execute()
        return {
            "nonterminal": int(_text(nonterminal)) if nonterminal is not None else 0,
            "runnable": int(runnable),
            "lease_expiry": int(leases),
        }

    def _apply_plan(  # noqa: C901
        self, pipe: Any, current: ExecutionControl, plan: TransitionPlan, *, new_submission: bool = False
    ) -> None:
        next_control = plan.next_control
        current_fields = encode_control(current)
        next_fields = encode_control(next_control)
        pipe.hset(self.keys.control(next_control.run_id), mapping=next_fields)
        removed_fields = tuple(set(current_fields).difference(next_fields))
        if removed_fields:
            pipe.hdel(self.keys.control(next_control.run_id), *removed_fields)
        for write in plan.payload_writes:
            pipe.set(self._payload_key(next_control.run_id, write.kind), write.data)
        for kind in plan.payload_deletes:
            pipe.delete(self._payload_key(next_control.run_id, kind))
        for event in plan.progress_events:
            pipe.rpush(self.keys.progress(next_control.run_id), event.data)
            pipe.ltrim(self.keys.progress(next_control.run_id), -self.config.max_progress_events, -1)

        pipe.zrem(self.keys.runnable, next_control.run_id)
        if next_control.status is ExecutionStatus.QUEUED:
            pipe.zadd(self.keys.runnable, {next_control.run_id: _runnable_score(next_control)})

        if plan.lease_index_update is not None:
            member = RedisKeys.lease_member(next_control.run_id, plan.lease_index_update.fence)
            if plan.lease_index_update.deadline_ms is None:
                pipe.zrem(self.keys.lease_expiry, member)
            else:
                pipe.zadd(self.keys.lease_expiry, {member: plan.lease_index_update.deadline_ms})

        if new_submission:
            pipe.hincrby(self.keys.capacity, "nonterminal", 1)
        elif not current.terminal and next_control.terminal:
            pipe.hincrby(self.keys.capacity, "nonterminal", -1)
            for key in self._execution_keys(next_control.run_id):
                pipe.expire(key, self.config.terminal_ttl_seconds)
            pipe.expire(self.keys.idempotency(next_control.idempotency_digest), self.config.terminal_ttl_seconds)

    async def _backoff(self, attempt: int) -> None:
        await redis_transaction_backoff(
            attempt,
            max_retries=self.config.transaction_max_retries,
            max_backoff_ms=self.config.transaction_backoff_max_ms,
        )

    async def _time_ms(self) -> int:
        return await redis_time_ms(self.redis)

    def _execution_keys(self, run_id: str) -> tuple[str, ...]:
        return (
            self.keys.control(run_id),
            self.keys.input(run_id),
            self.keys.checkpoint(run_id),
            self.keys.result(run_id),
            self.keys.error(run_id),
            self.keys.progress(run_id),
            self.keys.wait(run_id),
        )

    def _payload_key(self, run_id: str, kind: PayloadKind) -> str:
        return {
            PayloadKind.INPUT: self.keys.input,
            PayloadKind.CHECKPOINT: self.keys.checkpoint,
            PayloadKind.RESULT: self.keys.result,
            PayloadKind.ERROR: self.keys.error,
            PayloadKind.WAIT: self.keys.wait,
        }[kind](run_id)


def _runnable_score(control: ExecutionControl) -> int:
    return control.available_at_ms if control.available_at_ms is not None else control.updated_at_ms


def _text(value: str | bytes | int) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ExecutionStoreCorruptionError(f"control Hash has non-integer {name}") from error
    if parsed < 0:
        raise ExecutionStoreCorruptionError(f"control Hash has negative {name}")
    return parsed


async def redis_transaction_backoff(
    attempt: int,
    *,
    max_retries: int = DEFAULT_TRANSACTION_MAX_RETRIES,
    max_backoff_ms: int = DEFAULT_TRANSACTION_BACKOFF_MAX_MS,
) -> None:
    """Apply the bounded jitter shared by Redis optimistic transactions."""
    if attempt + 1 < max_retries and max_backoff_ms:
        await asyncio.sleep(random.uniform(0, max_backoff_ms) / 1_000)  # noqa: S311


def redis_watch_error() -> type[Exception]:
    """Load redis-py's optimistic-transaction conflict lazily."""
    try:
        from redis.exceptions import WatchError
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Redis durable storage requires the redis package") from error
    return WatchError
