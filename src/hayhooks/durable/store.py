"""Application-facing durable store backed by the simplified engine."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Mapping
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, cast

from hayhooks.durable.backend import ExecutionBackend, ExecutionContentionError, ExecutionStoreConfig, SubmissionResult
from hayhooks.durable.context import RESUME_INPUT_KEY
from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionControl,
    ExecutionLeaseLostError,
    ExecutionNotFoundError,
    Fail,
    Heartbeat,
    InvalidExecutionTransitionError,
    PayloadKind,
    RecoverExpiredLease,
    ReleaseClaim,
    RequestCancellation,
    Resume,
    ScheduleRetry,
    Suspend,
    initial_control,
    normalize_cancellation_reason,
)
from hayhooks.durable.engine import ExecutionStatus as EngineStatus
from hayhooks.durable.engine import ProgressEvent as EngineProgressEvent
from hayhooks.durable.models import (
    DEFAULT_MAX_PROGRESS_BYTES,
    ExecutionError,
    ExecutionKind,
    ExecutionProgressEvent,
    ExecutionRecord,
    ExecutionRecordSizeError,
    ExecutionStatus,
    ExecutionStoreError,
    JsonValue,
)
from hayhooks.durable.redis import RedisExecutionStore, digest
from hayhooks.durable.reference import InMemoryExecutionStore
from hayhooks.durable.settings import DurableSettings
from hayhooks.server.logger import log

_RECORD_PAYLOADS = (
    PayloadKind.INPUT,
    PayloadKind.CHECKPOINT,
    PayloadKind.RESULT,
    PayloadKind.ERROR,
    PayloadKind.WAIT,
)


def _encode(value: Any, *, limit: int, label: str) -> bytes:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        msg = f"{label} is not JSON serializable"
        raise ExecutionRecordSizeError(msg) from error
    if len(encoded) > limit:
        msg = f"{label} exceeds the {limit}-byte durable execution limit"
        raise ExecutionRecordSizeError(msg)
    return encoded


def _decode(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"durable {label} payload is invalid"
        raise RuntimeError(msg) from error


def _datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1_000, tz=timezone.utc)


def _error_from_payload(payload: bytes | None) -> ExecutionError | None:
    if payload is None:
        return None
    # The engine may terminalize an incompatible revision before application
    # code has had a chance to build an ``ExecutionError``.  Keep that
    # engine-owned reason visible through the public record rather than
    # treating an otherwise healthy terminal record as corrupt.
    try:
        decoded = _decode(payload, label="error")
    except RuntimeError:
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            msg = "durable error payload is invalid"
            raise RuntimeError(msg) from error
    if isinstance(decoded, Mapping):
        return ExecutionError.from_dict(cast(Mapping[str, Any], decoded))
    return ExecutionError(type="DurableExecutionError", message=str(decoded))


class ExecutionClaim:
    """Application-facing fenced claim backed by a durable control fence."""

    def __init__(
        self,
        store: ExecutionStore,
        control: Any,
        record: ExecutionRecord,
        worker_id: str,
        confirmed_at: float,
    ) -> None:
        self.store = store
        self.control = control
        self._record = record
        self.worker_id = worker_id
        self._heartbeat: asyncio.Task[None] | None = None
        self._transition_lock = asyncio.Lock()
        self._finished = False
        self._lost = False
        self._lost_event = asyncio.Event()
        self._confirmed_until = confirmed_at + self.store.lease_safe_duration
        self._last_persisted_progress = record.progress[-1] if record.progress else None

    @property
    def record(self) -> ExecutionRecord:
        return self._record

    @property
    def lost_event(self) -> asyncio.Event:
        return self._lost_event

    async def __aenter__(self) -> ExecutionClaim:
        async with self._transition_lock:
            await self._transition(Heartbeat(self.control.fence, self.worker_id, 0, self.store.lease_duration_ms))
        self._heartbeat = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"durable-heartbeat:{self.record.execution_id}",
        )
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat

    async def checkpoint(self) -> None:
        async with self._transition_lock:
            self._ensure_owned()
            progress, progress_payloads = self._new_progress()
            await self._transition(
                Checkpoint(
                    self.control.fence,
                    self.worker_id,
                    0,
                    self.store.lease_duration_ms,
                    self.store._snapshot(self.record),
                    progress_payloads,
                ),
                record_progress=progress,
            )

    async def cancellation_requested(self) -> bool:
        control = await self.store._core_call(
            "read cancellation state",
            self.store.core.get(self.record.execution_id),
        )
        if control is None:
            self._mark_lost()
            msg = f"Execution '{self.record.execution_id}' no longer exists"
            raise ExecutionLeaseLostError(msg)
        if control.cancel_requested_at_ms is not None:
            self.record.cancel_requested_at = _datetime(control.cancel_requested_at_ms)
            self.record.cancel_reason = control.cancel_reason
            return True
        return False

    async def complete(self) -> None:
        async with self._transition_lock:
            self._ensure_owned()
            if self.record.status is ExecutionStatus.CANCELED and self.control.cancel_requested_at_ms is None:
                cancellation = await self.store._core_call(
                    "request cancellation",
                    self.store.core.transition(
                        self.record.execution_id,
                        RequestCancellation(0, self.record.cancel_reason),
                    ),
                )
                self._sync(cancellation.next_control)
            progress, progress_payloads = self._new_progress()
            if self.record.status is ExecutionStatus.FAILED:
                error = self.record.error or ExecutionError(type="ExecutionError", message="Execution failed")
                await self._transition(
                    Fail(
                        self.control.fence,
                        self.worker_id,
                        0,
                        _encode(error.to_dict(), limit=self.store.config.max_error_bytes, label="error"),
                        progress_payloads,
                    ),
                    record_progress=progress,
                )
            else:
                await self._transition(
                    Complete(
                        self.control.fence,
                        self.worker_id,
                        0,
                        _encode(self.record.result, limit=self.store.config.max_result_bytes, label="result"),
                        progress_payloads,
                    ),
                    record_progress=progress,
                )
            self._finished = True

    async def suspend(self) -> None:
        async with self._transition_lock:
            self._ensure_owned()
            progress, progress_payloads = self._new_progress()
            await self._transition(
                Suspend(
                    self.control.fence,
                    self.worker_id,
                    0,
                    self.store._snapshot(self.record),
                    _encode(self.record.wait, limit=self.store.config.max_wait_bytes, label="wait"),
                    progress_payloads,
                ),
                record_progress=progress,
            )
            self._finished = True

    async def retry(self, error: ExecutionError, *, delay: float) -> None:
        async with self._transition_lock:
            self._ensure_owned()
            await self._transition(
                ScheduleRetry(
                    self.control.fence,
                    self.worker_id,
                    0,
                    max(0, round(delay * 1_000)),
                    self.store.max_application_retries,
                    _encode(error.to_dict(), limit=self.store.config.max_error_bytes, label="retry error"),
                )
            )
            self._finished = True

    async def _transition(self, command: Any, *, record_progress: tuple[ExecutionProgressEvent, ...] = ()) -> Any:
        confirmed_at = time.monotonic()
        try:
            plan = await self.store._core_call(
                "persist execution transition",
                self.store.core.transition(self.record.execution_id, command),
            )
        except ExecutionLeaseLostError:
            self._mark_lost()
            raise
        self._sync(
            plan.next_control,
            confirmed_at=confirmed_at,
            record_progress=record_progress,
            progress_events=plan.progress_events,
        )
        return plan

    async def _heartbeat_loop(self) -> None:
        while not self._finished and not self._lost:
            await asyncio.sleep(self.store.heartbeat_interval)
            try:
                async with self._transition_lock:
                    if self._finished or self._lost:
                        return
                    await self._transition(
                        Heartbeat(self.control.fence, self.worker_id, 0, self.store.lease_duration_ms)
                    )
            except ExecutionLeaseLostError:
                return
            except Exception:
                if time.monotonic() >= self._confirmed_until:
                    self._mark_lost()
                    return
                continue

    def _new_progress(self) -> tuple[tuple[ExecutionProgressEvent, ...], tuple[bytes, ...]]:
        events = self._new_progress_events()
        return (
            events,
            tuple(
                _encode(event.to_dict(), limit=self.store.config.max_progress_event_bytes, label="progress")
                for event in events
            ),
        )

    def _new_progress_events(self) -> tuple[ExecutionProgressEvent, ...]:
        if self._last_persisted_progress is None:
            return tuple(self.record.progress)
        for index, event in enumerate(self.record.progress):
            if event is self._last_persisted_progress:
                return tuple(self.record.progress[index + 1 :])
        return tuple(self.record.progress)

    def _sync(
        self,
        control: Any,
        *,
        confirmed_at: float | None = None,
        record_progress: tuple[ExecutionProgressEvent, ...] = (),
        progress_events: tuple[EngineProgressEvent, ...] = (),
    ) -> None:
        if progress_events:
            for event, persisted in zip(record_progress, progress_events, strict=True):
                event.sequence = persisted.sequence
            self._last_persisted_progress = record_progress[-1]
        self.control = control
        self.record.attempt = control.run_attempt
        self.record.sequence = control.version
        self.record.status = control.status
        if confirmed_at is not None:
            self._confirmed_until = confirmed_at + self.store.lease_safe_duration

    def _ensure_owned(self) -> None:
        if self._lost:
            msg = f"Execution lease for '{self.record.execution_id}' was lost"
            raise ExecutionLeaseLostError(msg)

    def _mark_lost(self) -> None:
        if not self._lost:
            self._lost = True
            self._lost_event.set()


class ExecutionStore:
    """Public durable store contract backed by control records and payloads."""

    def __init__(  # noqa: PLR0913
        self,
        core: ExecutionBackend,
        *,
        definition_revision: str | None = None,
        lease_duration_ms: int = 30_000,
        max_run_attempts: int = 3,
        max_progress_events: int = 100,
        max_record_bytes: int = 1_000_000,
    ) -> None:
        self.core = core
        self.config = core.config
        self.definition_revision = definition_revision
        self.lease_duration_ms = lease_duration_ms
        self.max_run_attempts = max_run_attempts
        self.max_application_retries = max(0, max_run_attempts - 1)
        self.max_progress_events = max_progress_events
        self.max_record_bytes = max_record_bytes
        if self.config.lease_commit_safety_ms >= self.lease_duration_ms:
            msg = "lease_commit_safety_ms must be smaller than lease_duration_ms"
            raise ValueError(msg)
        self.heartbeat_interval = max(0.01, self.lease_duration_ms / 3_000)
        if self.lease_duration_ms / 1_000 - self.config.lease_commit_safety_ms / 1_000 <= self.heartbeat_interval:
            msg = "lease duration minus commit safety must exceed the heartbeat interval"
            raise ValueError(msg)
        self.lease_safe_duration = max(
            0.01,
            self.lease_duration_ms / 1_000 - self.config.lease_commit_safety_ms / 1_000,
        )

    async def initialize(self) -> None:
        """Initialize and validate the backing execution store."""
        await self._core_call("initialize durable store", self.core.initialize())

    async def submit(self, record: ExecutionRecord) -> bool:
        created, _ = await self.submit_with_record(record)
        return created

    async def submit_with_record(self, record: ExecutionRecord) -> tuple[bool, ExecutionRecord]:
        """Atomically submit a record or return its idempotent predecessor."""
        input_payload = self._input(record)
        binding_digest = digest("binding", record.operation_fingerprint)
        control = initial_control(
            run_id=record.execution_id,
            idempotency_digest=digest("idempotency", record.execution_id),
            idempotency_binding_digest=binding_digest,
            deployment=record.deployment_name,
            definition_revision=record.definition_revision,
            owner_id=record.owner_id,
            kind=record.execution_kind.value,
            now_ms=round(time.time() * 1_000),
        )
        result: SubmissionResult = await self._core_call(
            "submit execution",
            self.core.submit(control, input_payload, binding_digest=binding_digest),
        )
        view = await self._read_view(result.control.run_id)
        if view is None:
            msg = f"Submitted execution '{result.control.run_id}' is not available"
            raise ExecutionStoreError(msg)
        return result.created, view[1]

    async def get(self, execution_id: str) -> ExecutionRecord | None:
        view = await self._read_view(execution_id)
        return view[1] if view is not None else None

    async def claim_next(self, worker_name: str) -> ExecutionClaim | None:
        """Claim one due execution with a new ownership fence, if available."""
        run_id = await self._core_call("read runnable candidate", self.core.read_candidate())
        if run_id is None:
            return None
        try:
            confirmed_at = time.monotonic()
            plan = await self._core_call(
                "claim execution",
                self.core.transition(
                    run_id,
                    Claim(
                        worker_name, 0, self.lease_duration_ms, self.max_run_attempts, self.definition_revision or ""
                    ),
                    candidate=True,
                ),
            )
        except (ExecutionLeaseLostError, ExecutionNotFoundError, InvalidExecutionTransitionError):
            return None
        if plan.next_control.status is not EngineStatus.RUNNING:
            return None
        try:
            view = await self._read_view(run_id)
        except BaseException:
            with suppress(Exception):
                await self._release_claim(plan.next_control, worker_name)
            raise
        if view is None:
            await self._release_claim(plan.next_control, worker_name)
            return None
        current, record = view
        if (
            current.status is not EngineStatus.RUNNING
            or current.fence != plan.next_control.fence
            or current.lease_owner != worker_name
        ):
            await self._release_claim(plan.next_control, worker_name)
            return None
        return ExecutionClaim(self, current, record, worker_name, confirmed_at)

    async def _release_claim(self, control: ExecutionControl, worker_name: str) -> None:
        with suppress(ExecutionLeaseLostError, ExecutionNotFoundError, InvalidExecutionTransitionError):
            await self._core_call(
                "release undelivered execution claim",
                self.core.transition(control.run_id, ReleaseClaim(control.fence, worker_name)),
            )

    async def request_cancel(self, execution_id: str, reason: str | None = None) -> bool:
        """Persist a cancellation request, returning whether it was accepted."""
        control = await self._core_call("read execution for cancellation", self.core.get(execution_id))
        if control is None:
            return False
        if control.terminal:
            return False
        event = _encode(
            {
                "sequence": control.progress_sequence + 1,
                "message": "Cancellation requested",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": "cancellation_requested",
                "metadata": {},
            },
            limit=self.config.max_progress_event_bytes,
            label="progress",
        )
        plan = await self._core_call(
            "request cancellation",
            self.core.transition(
                execution_id,
                RequestCancellation(0, normalize_cancellation_reason(reason), (event,)),
            ),
        )
        return bool(plan.progress_events)

    async def resume(self, execution_id: str, update: JsonValue | None = None) -> bool:
        """Resume a waiting execution with an optional JSON-safe application update."""
        view = await self._read_view(execution_id)
        if view is None:
            return False
        control, record = view
        if control.status is not EngineStatus.WAITING:
            return False
        if update is not None:
            record.application_state[RESUME_INPUT_KEY] = update
        record.wait = None
        event = record.append_progress("Execution resumed", kind="resumed")
        try:
            plan = await self._core_call(
                "resume execution",
                self.core.transition(
                    execution_id,
                    Resume(
                        0,
                        self.definition_revision or control.definition_revision,
                        self._snapshot(record),
                        (_encode(event.to_dict(), limit=self.config.max_progress_event_bytes, label="progress"),),
                        expected_version=control.version,
                    ),
                ),
            )
        except InvalidExecutionTransitionError:
            return False
        return plan.next_control.status is EngineStatus.QUEUED

    def set_definition_revision(self, definition_revision: str) -> None:
        """Set the revision accepted by future claims and resumes."""
        self.definition_revision = definition_revision

    async def maintain(self) -> None:
        """Recover expired leases and repair their derived indexes."""
        recover = lambda fence, deadline: RecoverExpiredLease(  # noqa: E731
            0,
            fence,
            deadline,
            self.max_run_attempts,
            self.definition_revision or "",
        )
        recovered = await self._core_call("maintain durable indexes", self.core.maintain(recover))
        if recovered:
            log.bind(deployment=self.core.deployment, recovered=recovered).debug(
                "Recovered expired durable execution leases"
            )

    async def operational_counts(self) -> dict[str, int]:
        """Return authoritative nonterminal, runnable, and lease-index counts."""
        return await self._core_call("read durable operational counts", self.core.operational_counts())

    async def _core_call(self, operation: str, awaitable: Awaitable[Any]) -> Any:
        """Expose backend outages through the public retryable-store contract."""
        try:
            return await awaitable
        except Exception as error:
            if _is_redis_error(error):
                msg = f"durable Redis store failed while attempting to {operation}"
                raise ExecutionStoreError(msg) from error
            raise

    async def _read_view(
        self, execution_id: str, *, control: ExecutionControl | None = None
    ) -> tuple[ExecutionControl, ExecutionRecord] | None:
        """Read one control/payload view that was stable for its construction."""
        for _ in range(self.config.transaction_max_retries):
            control = control or await self._core_call("read execution control", self.core.get(execution_id))
            if control is None:
                return None
            payloads, progress = await asyncio.gather(
                self._core_call("read execution payloads", self.core.read_payloads(control.run_id, _RECORD_PAYLOADS)),
                self._core_call("read execution progress", self.core.read_progress(control.run_id)),
            )
            current = await self._core_call("recheck execution control", self.core.get(execution_id))
            if current is None:
                return None
            if current.version != control.version:
                control = None
                continue

            # A stable control must retain the payload that explains its
            # terminal or waiting state; otherwise the view is corrupt.
            missing = PayloadKind.INPUT.value if payloads[PayloadKind.INPUT] is None else None
            if current.status is EngineStatus.COMPLETED:
                missing = missing or (PayloadKind.RESULT.value if payloads[PayloadKind.RESULT] is None else None)
            elif current.status is EngineStatus.FAILED:
                missing = missing or (PayloadKind.ERROR.value if payloads[PayloadKind.ERROR] is None else None)
            elif current.status is EngineStatus.WAITING:
                missing = missing or (PayloadKind.WAIT.value if payloads[PayloadKind.WAIT] is None else None)
            if missing is not None:
                msg = f"Execution '{execution_id}' is missing its required {missing} payload"
                raise RuntimeError(msg)
            return current, self._record(current, payloads, progress)
        msg = "execution changed while its durable view was being read"
        raise ExecutionContentionError(msg)

    def _record(
        self,
        control: ExecutionControl,
        payloads: Mapping[PayloadKind, bytes | None],
        progress_payloads: list[bytes],
    ) -> ExecutionRecord:
        """Translate one stable engine view into the established public record."""
        input_payload = payloads[PayloadKind.INPUT]
        if input_payload is None:
            msg = f"Execution '{control.run_id}' is missing its immutable input payload"
            raise RuntimeError(msg)
        input_data = _decode(input_payload, label="input")
        checkpoint_payload = payloads[PayloadKind.CHECKPOINT]
        snapshot = _decode(checkpoint_payload, label="checkpoint") if checkpoint_payload is not None else {}
        progress = [_decode(event, label="progress") for event in progress_payloads]
        result_payload = payloads[PayloadKind.RESULT]
        result = _decode(result_payload, label="result") if result_payload is not None else None
        error = _error_from_payload(payloads[PayloadKind.ERROR])
        wait_payload = payloads[PayloadKind.WAIT]
        wait = _decode(wait_payload, label="wait") if wait_payload is not None else None
        return ExecutionRecord(
            execution_id=control.run_id,
            execution_kind=ExecutionKind(control.kind),
            deployment_name=control.deployment,
            definition_revision=control.definition_revision,
            validated_input=input_data["validated_input"],
            operation_fingerprint=input_data["operation_fingerprint"],
            owner_id=control.owner_id,
            status=ExecutionStatus(control.status.value),
            sequence=control.version,
            attempt=control.run_attempt,
            checkpoint=snapshot.get("checkpoint"),
            application_state=snapshot.get("application_state", {}),
            wait=wait,
            progress=progress,
            result=result if control.status is EngineStatus.COMPLETED else None,
            error=error if control.status is EngineStatus.FAILED else None,
            last_retry_error=error if not control.terminal else None,
            retry_at=_datetime(control.available_at_ms) if control.available_at_ms is not None else None,
            cancel_requested_at=(
                _datetime(control.cancel_requested_at_ms) if control.cancel_requested_at_ms is not None else None
            ),
            cancel_reason=control.cancel_reason,
            created_at=_datetime(control.created_at_ms),
            updated_at=_datetime(control.updated_at_ms),
            max_progress_events=self.max_progress_events,
            max_record_bytes=self.max_record_bytes,
        )

    def _input(self, record: ExecutionRecord) -> bytes:
        return _encode(
            {
                "validated_input": record.validated_input,
                "operation_fingerprint": record.operation_fingerprint,
            },
            limit=self.config.max_input_bytes,
            label="validated input",
        )

    def _snapshot(self, record: ExecutionRecord) -> bytes:
        return _encode(
            {
                "checkpoint": record.checkpoint.to_dict() if record.checkpoint else None,
                "application_state": record.application_state,
            },
            limit=self.config.max_checkpoint_bytes,
            label="checkpoint",
        )


class RedisExecutionStoreProvider:
    """Application-owned Redis client and deployment stores."""

    def __init__(  # noqa: PLR0913 - mirrors the configurable Redis task-store provider
        self,
        redis_url: str | None = None,
        *,
        redis: Any | None = None,
        key_prefix: str | None = None,
        close_redis: bool = True,
        durable_settings: DurableSettings | None = None,
        app_settings: Any | None = None,
        socket_timeout: float | None = None,
        socket_connect_timeout: float | None = None,
        health_check_interval: int | None = None,
    ) -> None:
        if durable_settings is not None and app_settings is not None:
            msg = "Pass durable_settings or app_settings, not both"
            raise ValueError(msg)
        self.settings = (
            durable_settings
            or (DurableSettings.from_app_settings(app_settings) if app_settings is not None else DurableSettings())
        ).model_copy(deep=True)
        self.app_settings = self.settings
        self.config = _config(durable_settings=self.settings, key_prefix=key_prefix)
        self.close_redis = close_redis
        self.socket_timeout = (
            socket_timeout if socket_timeout is not None else self.settings.durable_redis_socket_timeout
        )
        self.socket_connect_timeout = (
            socket_connect_timeout
            if socket_connect_timeout is not None
            else self.settings.durable_redis_socket_connect_timeout
        )
        self.health_check_interval = (
            health_check_interval
            if health_check_interval is not None
            else self.settings.durable_redis_health_check_interval
        )
        if redis is None:
            try:
                from redis.asyncio import Redis
            except ImportError as error:  # pragma: no cover - optional dependency guard
                msg = 'Durable Redis storage requires `pip install "hayhooks[durable]`.'
                raise ImportError(msg) from error
            redis = Redis.from_url(
                redis_url or self.settings.durable_redis_url,
                decode_responses=False,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                health_check_interval=self.health_check_interval,
            )
        self.redis = redis
        self.cores: dict[str, RedisExecutionStore] = {}

    def create_execution_store(self, deployment_name: str) -> ExecutionStore:
        core = self.cores.get(deployment_name)
        if core is None:
            core = RedisExecutionStore(self.redis, deployment=deployment_name, config=self.config)
            self.cores[deployment_name] = core
        # A candidate deployment must not mutate the active deployment's accepted
        # definition revision while it is still preparing or rolling back.
        return _execution_store(core, durable_settings=self.settings)

    async def close(self) -> None:
        if self.close_redis:
            await self.redis.aclose()


class InMemoryExecutionStoreProvider:
    """Volatile reference backend for local development and tests."""

    def __init__(
        self,
        *,
        durable_settings: DurableSettings | None = None,
        app_settings: Any | None = None,
    ) -> None:
        if durable_settings is not None and app_settings is not None:
            msg = "Pass durable_settings or app_settings, not both"
            raise ValueError(msg)
        self.settings = (
            durable_settings
            or (DurableSettings.from_app_settings(app_settings) if app_settings is not None else DurableSettings())
        ).model_copy(deep=True)
        self.app_settings = self.settings
        self.config = _config(durable_settings=self.settings)
        self.cores: dict[str, InMemoryExecutionStore] = {}

    def create_execution_store(self, deployment_name: str) -> ExecutionStore:
        core = self.cores.get(deployment_name)
        if core is None:
            core = InMemoryExecutionStore(deployment=deployment_name, config=self.config)
            self.cores[deployment_name] = core
        return _execution_store(core, durable_settings=self.settings)

    async def close(self) -> None:
        return None


def _execution_store(core: ExecutionBackend, *, durable_settings: DurableSettings) -> ExecutionStore:
    """Build the same public adapter for both built-in backend implementations."""
    return ExecutionStore(
        core,
        lease_duration_ms=durable_settings.durable_lease_duration_ms,
        max_run_attempts=durable_settings.durable_max_attempts,
        max_progress_events=durable_settings.durable_max_progress_events,
        max_record_bytes=durable_settings.durable_max_record_bytes,
    )


def _config(*, durable_settings: DurableSettings, key_prefix: str | None = None) -> ExecutionStoreConfig:
    max_record = durable_settings.durable_max_record_bytes
    progress_bytes = DEFAULT_MAX_PROGRESS_BYTES
    return ExecutionStoreConfig(
        key_prefix=key_prefix or durable_settings.durable_redis_key_prefix,
        lease_commit_safety_ms=durable_settings.durable_lease_commit_safety_ms,
        terminal_ttl_seconds=durable_settings.durable_terminal_ttl_seconds,
        max_nonterminal_executions=durable_settings.durable_max_nonterminal_executions,
        max_input_bytes=max_record,
        max_checkpoint_bytes=max_record,
        max_result_bytes=max_record,
        max_error_bytes=max_record,
        max_wait_bytes=max_record,
        max_progress_events=durable_settings.durable_max_progress_events,
        max_progress_event_bytes=progress_bytes,
    )


def _is_redis_error(error: BaseException) -> bool:
    """Avoid exposing Redis client exceptions directly through HTTP routes."""
    try:
        from redis.exceptions import RedisError
    except ImportError:  # pragma: no cover - Redis is an optional dependency
        return False
    return isinstance(error, RedisError)


__all__ = ["ExecutionClaim", "ExecutionStore"]
