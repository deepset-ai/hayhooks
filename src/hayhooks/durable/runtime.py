"""Portable durable deployment and worker runtime."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import random
import secrets
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from loguru import logger as log
from pydantic import BaseModel

from hayhooks.durable._threading import is_async_callable
from hayhooks.durable.context import (
    DurableContext,
    DurableExecutionCancelledError,
    _ClaimedExecution,
    _ExecutionSuspendedError,
    _RetryRequestedError,
    _track,
    durable_context_scope,
)
from hayhooks.durable.engine import (
    MAX_CONTROL_SCALAR_BYTES,
    Claim,
    Complete,
    ExecutionControl,
    ExecutionLeaseLostError,
    ExecutionNotFoundError,
    ExecutionPayloadSizeError,
    ExecutionStatus,
    Fail,
    InvalidExecutionTransitionError,
    PayloadKind,
    ReleaseClaim,
    RequestCancellation,
    Resume,
    ScheduleRetry,
    TransitionPlan,
    initial_control,
)
from hayhooks.durable.models import (
    CheckpointEnvelope,
    ExecutionKind,
    PersistedError,
    decode_json,
    encode_json,
    operation_fingerprint,
)
from hayhooks.durable.store import (
    ExecutionStore,
    ExecutionStoreCorruptionError,
    ExecutionStoreError,
    StoredExecution,
    StreamChunk,
    SubmissionResult,
)

DurableRunner: TypeAlias = Callable[[DurableContext, BaseModel], object]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Worker, lease, retry, and operational retry limits."""

    worker_concurrency: int = 1
    poll_interval_seconds: float = 5.0
    maintenance_interval_seconds: float = 5.0
    shutdown_grace_seconds: float = 5.0
    lease_duration_ms: int = 30_000
    max_run_attempts: int = 3
    max_application_retries: int = 2
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 60.0
    operational_backoff_min_seconds: float = 0.05
    operational_backoff_max_seconds: float = 5.0
    # Hand thread-backed work still running after the shutdown grace to another process.
    release_running_on_close: bool = False

    def __post_init__(self) -> None:
        values = (
            self.poll_interval_seconds,
            self.maintenance_interval_seconds,
            self.shutdown_grace_seconds,
            self.retry_base_delay_seconds,
            self.retry_max_delay_seconds,
            self.operational_backoff_min_seconds,
            self.operational_backoff_max_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("runtime durations must be finite")
        if self.worker_concurrency < 1 or self.max_run_attempts < 1 or self.max_application_retries < 0:
            raise ValueError(
                "worker concurrency and run attempts must be positive; application retries cannot be negative"
            )
        if self.lease_duration_ms < 1:
            raise ValueError("lease_duration_ms must be positive")
        if self.poll_interval_seconds <= 0 or self.maintenance_interval_seconds <= 0 or self.shutdown_grace_seconds < 0:
            raise ValueError("poll and maintenance intervals must be positive; shutdown grace cannot be negative")
        if (
            self.retry_base_delay_seconds < 0
            or self.retry_max_delay_seconds < self.retry_base_delay_seconds
            or self.operational_backoff_min_seconds <= 0
            or self.operational_backoff_max_seconds < self.operational_backoff_min_seconds
        ):
            raise ValueError("runtime retry and backoff bounds are invalid")


class _DeploymentState(Enum):
    """Admission lifecycle; stopped deployments may still have work to drain."""

    CREATED = auto()
    STARTING = auto()
    ACTIVE = auto()
    STOPPED = auto()


class DurableDeployment:
    """
    One typed durable callable backed by an explicit execution store.

    Its lifecycle is one-way: ``start()`` opens admission once, and ``quiesce()`` or ``close()``
    stops it for good. Use a new instance for a new lifecycle.
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        revision: str,
        store: ExecutionStore,
        request_model: type[BaseModel],
        runner: DurableRunner,
        *,
        kind: ExecutionKind = ExecutionKind.PIPELINE,
        result_model: type[BaseModel] | None = None,
        resume_model: type[BaseModel] | None = None,
        adapter: Any | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        if not name or store.deployment != name:
            raise ValueError("deployment name must be non-empty and match its store")
        if not revision.strip():
            raise ValueError("definition revision cannot be empty")
        if not inspect.isclass(request_model) or not issubclass(request_model, BaseModel):
            raise TypeError("request_model must be a Pydantic model class")
        for label, model in (("result_model", result_model), ("resume_model", resume_model)):
            if model is not None and (not inspect.isclass(model) or not issubclass(model, BaseModel)):
                raise TypeError(f"{label} must be a Pydantic model class or None")
        if not callable(runner):
            raise TypeError("runner must be callable")

        self.name = name
        self.revision = revision.strip()
        self.store = store
        self.request_model = request_model
        self.result_model = result_model
        self.resume_model = resume_model
        self.runner = runner
        self.kind = ExecutionKind(kind)
        self.adapter = adapter
        if adapter is not None and adapter.kind is not self.kind:
            raise ValueError("Haystack adapter kind does not match the deployment")
        self.config = config or RuntimeConfig()
        heartbeat_interval_ms = max(10, self.config.lease_duration_ms / 3)
        safe_lease_ms = self.config.lease_duration_ms - store.config.lease_commit_safety_ms
        if safe_lease_ms <= heartbeat_interval_ms:
            raise ValueError("lease duration must leave more than one safe heartbeat interval")
        self._fallback_error = encode_json(
            PersistedError(type="Error", message="").model_dump(mode="json"),
            max_bytes=store.config.max_payload_bytes,
        )
        self._attempts_error = self._encode_error(
            "RunAttemptsExhaustedError",
            "run attempts exhausted",
            code="run_attempts_exhausted",
        )
        self._runner_is_async = is_async_callable(runner)
        self._submission_condition = asyncio.Condition()
        # Local submissions and shutdown wake idle workers before their next poll.
        self._work_available = asyncio.Event()
        self._chunk_waits: set[asyncio.Task[tuple[StreamChunk, ...]]] = set()
        self._active_claims = 0
        self._admitted_submissions = 0
        self._state = _DeploymentState.CREATED
        # Set by the first close(); repeated calls share its grace instead of starting a new one.
        self._shutdown_deadline: float | None = None
        self._worker_identity = secrets.token_hex(8)
        self._workers: dict[int, asyncio.Task[None]] = {}
        self._claims: dict[asyncio.Task[None], _ClaimedExecution] = {}
        # Cancelled application work and engine threads that outlived their claim.
        self._draining_runs: set[asyncio.Future[Any]] = set()
        self._draining_threads: set[asyncio.Future[None]] = set()
        self._worker_store_error_streaks: dict[str, int] = {}
        self._maintenance_error_streak = 0
        self._maintenance_task: asyncio.Task[None] | None = None

    @property
    def accepting(self) -> bool:
        return self._state is _DeploymentState.ACTIVE

    async def start(self) -> None:
        """Initialize storage and open admission and workers; repeated while active, it does nothing."""
        async with self._submission_condition:
            if self.accepting:
                return
            if self._state is not _DeploymentState.CREATED:
                raise RuntimeError(
                    "a quiesced, closed, or failed durable deployment cannot be restarted; create a new instance"
                )
            self._state = _DeploymentState.STARTING
            try:
                await self.store.initialize()
            except BaseException:
                self._state = _DeploymentState.STOPPED
                raise
            self._state = _DeploymentState.ACTIVE
            self._ensure_workers()
            self._maintenance_task = asyncio.create_task(self._maintenance(), name=f"durable-maintenance:{self.name}")
            self._maintenance_task.add_done_callback(self._maintenance_stopped)

    async def quiesce(self) -> None:
        """Permanently close admission, wait for admitted submissions, and stop new claims."""
        async with self._submission_condition:
            self._state = _DeploymentState.STOPPED
            self._work_available.set()
            await self._submission_condition.wait_for(lambda: self._admitted_submissions == 0)
        # Maintenance serves claims this deployment will never make again.
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.wait({self._maintenance_task})

    async def close(self) -> None:
        """
        Quiesce, end open streams, and give workers ``shutdown_grace_seconds`` to finish.

        Async work still running then is cancelled and gets up to another grace period to stop;
        work that stops releases its claim, so another process can take the run over without
        spending an attempt. Cancellation-resistant async work keeps its claim until it exits, and
        so do threads, unless ``release_running_on_close`` hands their claims over. Async runners
        awaiting thread work are cancelled when their last thread exits; no new threads may start.
        A repeated call completes cleanup that an earlier one left unfinished, within the same deadline;
        ``wait_drained()`` waits for the work that close() retains.
        """
        await self.quiesce()
        loop = asyncio.get_running_loop()
        if self._shutdown_deadline is None:
            self._shutdown_deadline = loop.time() + self.config.shutdown_grace_seconds
        # Streams end without a terminal event so clients resume from their cursor, possibly elsewhere.
        waits = tuple(self._chunk_waits)
        for wait in waits:
            wait.cancel()
        await asyncio.gather(*waits, return_exceptions=True)
        workers = [worker for worker in self._workers.values() if not worker.done()]
        if not workers:
            return
        _, pending = await asyncio.wait(workers, timeout=max(0.0, self._shutdown_deadline - loop.time()))
        stopping = set()
        for worker in pending:
            claim = self._claims.get(worker)
            if claim is None:
                worker.cancel()
                stopping.add(worker)
            elif claim.request_shutdown(worker, release_running=self.config.release_running_on_close):
                stopping.add(worker)
        if stopping:
            grace = self.config.shutdown_grace_seconds
            await asyncio.wait(stopping, timeout=max(0.0, self._shutdown_deadline + grace - loop.time()))

    async def wait_drained(self) -> None:
        """
        Wait until work retained past ``close()`` no longer owns claims or store access.

        Call it after ``close()`` before releasing resources such as a shared Redis client. Waiting
        never cancels retained work, so a cancelled wait can be repeated. With
        ``release_running_on_close``, released threads may still be running when it returns.
        """
        if self._state in (_DeploymentState.STARTING, _DeploymentState.ACTIVE):
            raise RuntimeError("close the durable deployment before waiting for it to drain")
        async with self._submission_condition:
            await self._submission_condition.wait_for(lambda: self._admitted_submissions == 0)
        if self._undrained():
            log.bind(deployment=self.name).info("Waiting for retained durable work to finish")
        while undrained := self._undrained():
            await asyncio.wait(undrained)

    async def submit(
        self,
        payload: object,
        *,
        owner_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> SubmissionResult:
        """Validate and durably admit one execution."""
        request = self.request_model.model_validate(payload)
        json_input = request.model_dump(mode="json")
        input_payload = encode_json(json_input, max_bytes=self.store.config.max_payload_bytes)
        binding = operation_fingerprint(
            self.name,
            self.revision,
            owner_id,
            request,
            max_bytes=self.store.config.max_payload_bytes + 3 * MAX_CONTROL_SCALAR_BYTES + 256,
        )
        idempotency_material = idempotency_key if idempotency_key is not None else secrets.token_urlsafe(32)
        owner_scope = (owner_id or "").encode()
        digest = hashlib.sha256(
            len(owner_scope).to_bytes(8, "big") + owner_scope + idempotency_material.encode()
        ).hexdigest()
        control = initial_control(
            run_id=secrets.token_hex(16),
            idempotency_digest=digest,
            idempotency_binding_digest=binding,
            deployment=self.name,
            definition_revision=self.revision,
            owner_id=owner_id,
            kind=self.kind.value,
            now_ms=0,
        )
        async with self._submission_condition:
            if not self.accepting:
                raise RuntimeError(f"durable deployment '{self.name}' is not accepting submissions")
            self._admitted_submissions += 1
        try:
            submission = await self.store.submit(control, input_payload)
            self._work_available.set()
            return submission
        finally:
            async with self._submission_condition:
                self._admitted_submissions -= 1
                if not self._admitted_submissions:
                    self._submission_condition.notify_all()

    async def get(
        self,
        run_id: str,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = True,
        allow_revision_mismatch: bool = False,
    ) -> StoredExecution:
        """Read one execution's public snapshot after deployment, owner, and revision checks."""
        return self._authorize(
            run_id,
            await self.store.read_public(run_id),
            owner_id=owner_id,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=allow_revision_mismatch,
        )

    async def wait_chunks(self, run_id: str, after: str, timeout: float) -> tuple[StreamChunk, ...] | None:
        """Wait for stream entries after ``after``; ``None`` once this deployment closes."""
        if self._shutdown_deadline is not None:
            return None
        wait = asyncio.create_task(self.store.wait_chunks(run_id, after, timeout))
        _track(self._chunk_waits, wait)
        try:
            await asyncio.wait({wait})
        finally:
            wait.cancel()
            await asyncio.gather(wait, return_exceptions=True)
        return None if wait.cancelled() else wait.result()

    async def cancel(
        self,
        run_id: str,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = True,
        reason: str | None = None,
    ) -> TransitionPlan:
        """Request cancellation without exposing owner mismatches."""
        await self.get(run_id, owner_id=owner_id, enforce_owner=enforce_owner, allow_revision_mismatch=True)
        return await self.store.transition(run_id, RequestCancellation(now_ms=0, reason=reason))

    async def resume(
        self,
        run_id: str,
        resume_input: object = None,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = True,
    ) -> TransitionPlan:
        """Validate resume input and atomically requeue a waiting execution."""
        stored = self._authorize(
            run_id,
            await self.store.read(run_id),
            owner_id=owner_id,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=False,
        )
        if stored.control.status is not ExecutionStatus.WAITING:
            raise InvalidExecutionTransitionError("only waiting executions can resume")
        try:
            checkpoint_payload = stored.payloads.get(PayloadKind.CHECKPOINT)
            if checkpoint_payload is None:
                raise ValueError("waiting execution has no checkpoint")
            checkpoint = CheckpointEnvelope.model_validate(
                decode_json(checkpoint_payload, max_bytes=self.store.config.max_payload_bytes)
            )
            if checkpoint.adapter_kind is not self.kind:
                raise ValueError("checkpoint kind does not match the deployment")
        except (ExecutionPayloadSizeError, TypeError, ValueError) as error:
            raise ExecutionStoreCorruptionError("stored checkpoint payload is invalid") from error
        if self.resume_model is not None:
            resume_input = self.resume_model.model_validate(resume_input).model_dump(mode="json")
        checkpoint = CheckpointEnvelope.model_validate(
            {**checkpoint.model_dump(mode="json"), "resume_input": resume_input}
        )
        plan = await self.store.transition(
            run_id,
            Resume(
                now_ms=0,
                worker_revision=self.revision,
                checkpoint=encode_json(
                    checkpoint.model_dump(mode="json"), max_bytes=self.store.config.max_payload_bytes
                ),
                expected_version=stored.control.version,
            ),
        )
        self._work_available.set()
        return plan

    async def health(self) -> dict[str, object]:
        """
        Return local worker state plus bounded store counts.

        ``active_executions`` counts claims this process still runs plus cancelled
        or released work that is still running here, so hosts can report durable
        work as busy.
        """
        live_workers = sum(not worker.done() for worker in self._workers.values())
        running = live_workers if self.accepting else 0
        draining_runs = sum(not run.done() for run in (*self._draining_runs, *self._draining_threads))
        maintenance_running = self._maintenance_task is not None and not self._maintenance_task.done()
        worker_error_streak = max(self._worker_store_error_streaks.values(), default=0)
        health: dict[str, object] = {
            "healthy": (
                self.accepting
                and running == self.config.worker_concurrency
                and maintenance_running
                and not worker_error_streak
                and not self._maintenance_error_streak
            ),
            "configured_slots": self.config.worker_concurrency,
            "running_slots": running,
            "draining_slots": live_workers - running,
            "draining_runs": draining_runs,
            "active_executions": self._active_claims + draining_runs,
            "maintenance_running": maintenance_running,
            "accepting": self.accepting,
            "store_error_streak": max(worker_error_streak, self._maintenance_error_streak),
        }
        try:
            health["counts"] = await self.store.operational_counts()
        except ExecutionStoreError as error:
            health["healthy"] = False
            health["operational_error"] = type(error).__name__
        return health

    def _authorize(
        self,
        run_id: str,
        stored: StoredExecution | None,
        *,
        owner_id: str | None,
        enforce_owner: bool,
        allow_revision_mismatch: bool,
    ) -> StoredExecution:
        if (
            stored is None
            or stored.control.deployment != self.name
            or (enforce_owner and stored.control.owner_id != owner_id)
        ):
            raise ExecutionNotFoundError(f"execution '{run_id}' was not found")
        if (
            not allow_revision_mismatch
            and not stored.control.terminal
            and stored.control.definition_revision != self.revision
        ):
            raise InvalidExecutionTransitionError("execution definition revision is incompatible")
        return stored

    def _undrained(self) -> set[asyncio.Future[Any]]:
        tracked: set[asyncio.Future[Any]] = {*self._workers.values(), *self._draining_runs}
        if self._maintenance_task is not None:
            tracked.add(self._maintenance_task)
        if not self.config.release_running_on_close:
            tracked |= self._draining_threads
        return {future for future in tracked if not future.done()}

    def _ensure_workers(self) -> None:
        for slot, worker in tuple(self._workers.items()):
            if not worker.done():
                continue
            self._workers.pop(slot)
            if not worker.cancelled() and (error := worker.exception()) is not None:
                log.bind(deployment=self.name, exception_type=type(error).__name__).error(
                    "Durable worker slot stopped unexpectedly"
                )
        for slot in range(self.config.worker_concurrency):
            if not self.accepting or slot in self._workers:
                continue
            worker = asyncio.create_task(
                self._worker(f"{self._worker_identity}-{slot}"),
                name=f"durable:{self.name}:{slot}",
            )
            self._workers[slot] = worker
            worker.add_done_callback(self._worker_stopped)

    def _maintenance_stopped(self, maintenance: asyncio.Task[None]) -> None:
        if not maintenance.cancelled() and (error := maintenance.exception()) is not None:
            log.opt(exception=error).bind(deployment=self.name, exception_type=type(error).__name__).error(
                "Durable maintenance stopped unexpectedly"
            )

    def _worker_stopped(self, _worker: asyncio.Task[None]) -> None:
        """Restore worker capacity immediately without tying supervision to Redis maintenance."""
        self._ensure_workers()

    async def _maintenance(self) -> None:
        while self.accepting:
            try:
                if await self.store.maintain(
                    max_run_attempts=self.config.max_run_attempts,
                    attempts_error=self._attempts_error,
                ):
                    self._work_available.set()
            except asyncio.CancelledError:
                raise
            except ExecutionStoreError as error:
                self._maintenance_error_streak += 1
                await self._backoff(error, self._maintenance_error_streak, "maintenance")
            else:
                self._maintenance_error_streak = 0
                await asyncio.sleep(self.config.maintenance_interval_seconds)

    async def _worker(self, worker_id: str) -> None:
        self._worker_store_error_streaks[worker_id] = 0
        while self.accepting:
            control = await self._claim_next_execution(worker_id)
            if control is None:
                continue

            self._active_claims += 1
            try:
                await self._execute_claim(control, worker_id)
            except asyncio.CancelledError:
                raise
            except ExecutionLeaseLostError:
                continue
            except ExecutionStoreError as error:
                await self._backoff_worker(worker_id, error, "transition")
            finally:
                self._active_claims -= 1

    async def _claim_next_execution(self, worker_id: str) -> ExecutionControl | None:
        try:
            claimed = await self.store.claim(
                Claim(
                    worker_id=worker_id,
                    now_ms=0,
                    lease_duration_ms=self.config.lease_duration_ms,
                    max_run_attempts=self.config.max_run_attempts,
                    worker_revision=self.revision,
                    attempts_error=self._attempts_error,
                )
            )
        except ExecutionStoreError as error:
            await self._backoff_worker(worker_id, error, "claim")
            return None
        self._worker_store_error_streaks[worker_id] = 0
        if claimed is None:
            if self.accepting:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._work_available.wait(), self.config.poll_interval_seconds)
            # Consume the wake-up here, so one that arrived while every worker was busy is not lost.
            self._work_available.clear()
            return None
        control = claimed.next_control
        return control if control.status is ExecutionStatus.RUNNING else None

    async def _read_claimed_execution(
        self,
        control: ExecutionControl,
        worker_id: str,
    ) -> StoredExecution | None:
        """Load a claim, releasing it when the post-claim read cannot complete."""
        try:
            stored = await self.store.read(control.run_id)
        except ExecutionStoreError as error:
            with suppress(ExecutionLeaseLostError, ExecutionNotFoundError, ExecutionStoreError):
                await self.store.transition(control.run_id, ReleaseClaim(fence=control.fence, worker_id=worker_id))
            await self._backoff_worker(worker_id, error, "read")
            return None
        if stored is not None:
            return stored
        try:
            await self.store.transition(control.run_id, ReleaseClaim(fence=control.fence, worker_id=worker_id))
        except (ExecutionLeaseLostError, ExecutionNotFoundError):
            pass
        except ExecutionStoreError as error:
            await self._backoff_worker(worker_id, error, "release")
        return None

    async def _prepare_execution(
        self,
        stored: StoredExecution,
        claim: _ClaimedExecution,
        worker_id: str,
    ) -> tuple[DurableContext, BaseModel] | None:
        control = stored.control
        try:
            request = self.request_model.model_validate(
                decode_json(stored.payloads[PayloadKind.INPUT], max_bytes=self.store.config.max_payload_bytes)
            )
            checkpoint_payload = stored.payloads.get(PayloadKind.CHECKPOINT)
            checkpoint = (
                CheckpointEnvelope.model_validate(
                    decode_json(checkpoint_payload, max_bytes=self.store.config.max_payload_bytes)
                )
                if checkpoint_payload is not None
                else CheckpointEnvelope(schema_version=1, adapter_kind=self.kind, adapter_checkpoint=None)
            )
            if checkpoint.adapter_kind is not self.kind:
                raise ValueError("checkpoint kind does not match the deployment")
        except (KeyError, TypeError, ValueError, ExecutionPayloadSizeError) as error:
            await self.store.transition(
                control.run_id,
                Fail(fence=control.fence, worker_id=worker_id, now_ms=0, error=self._encode_exception(error)),
            )
            return None

        claim.control = control
        return DurableContext(claim, checkpoint, adapter=self.adapter), request

    async def _execute_claim(
        self,
        control: ExecutionControl,
        worker_id: str,
    ) -> None:
        claim = _ClaimedExecution(self.store, control, worker_id, self.config.lease_duration_ms)
        worker = cast(asyncio.Task[None], asyncio.current_task())
        self._claims[worker] = claim
        release: asyncio.Task[None] | None = None
        try:
            stored = await self._read_claimed_execution(control, worker_id)
            if stored is None:
                return
            prepared = await self._prepare_execution(stored, claim, worker_id)
            if prepared is None:
                return
            context, request = prepared
            # Include the post-claim read and initial heartbeat in cancellation cleanup.
            await claim.start()
            await self._run_claim(claim, context, request, worker_id)
        except asyncio.CancelledError:
            # The heartbeat stops with this worker, so hand the run back rather than let its lease expire.
            claim.stopping = True
            release = asyncio.create_task(claim.release())
            _track(self._draining_runs, release)
            release.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            await asyncio.shield(release)
            raise
        finally:
            if release is None:
                await claim.stop()
            del self._claims[worker]
            for thread in tuple(claim.threads):
                _track(self._draining_threads, thread)

    async def _run_claim(
        self,
        claim: _ClaimedExecution,
        context: DurableContext,
        request: BaseModel,
        worker_id: str,
    ) -> None:
        try:
            if claim.control.cancel_requested_at_ms is not None:
                await self._acknowledge_cancellation(claim, context, worker_id)
                return

            result = await self._invoke_application(claim, context, request)
            if self.result_model is not None:
                result = self.result_model.model_validate(result).model_dump(mode="json")
            elif isinstance(result, BaseModel):
                result = result.model_dump(mode="json")
            await claim.transition(
                Complete(
                    fence=claim.control.fence,
                    worker_id=worker_id,
                    now_ms=0,
                    result=encode_json(result, max_bytes=self.store.config.max_payload_bytes),
                    progress_events=context._progress_events,
                )
            )
        except _ExecutionSuspendedError:
            return
        except DurableExecutionCancelledError:
            await self._acknowledge_cancellation(claim, context, worker_id)
        except _RetryRequestedError as error:
            await self._schedule_retry(claim, error, worker_id)
        except (asyncio.CancelledError, ExecutionLeaseLostError, ExecutionStoreError):
            raise
        except Exception as error:
            if claim.application_cancelled:
                raise asyncio.CancelledError from error
            # The persisted error only names the exception type, so the log is where operators find the cause.
            log.opt(exception=error).bind(
                deployment=self.name, run_id=claim.control.run_id, exception_type=type(error).__name__, error=str(error)
            ).error("Durable execution failed")
            code = "payload_too_large" if isinstance(error, ExecutionPayloadSizeError) else None
            await claim.transition(
                Fail(
                    fence=claim.control.fence,
                    worker_id=worker_id,
                    now_ms=0,
                    error=self._encode_exception(error, code=code),
                    progress_events=context._progress_events,
                )
            )

    async def _schedule_retry(self, claim: _ClaimedExecution, error: _RetryRequestedError, worker_id: str) -> None:
        """Requeue with backoff and wake a local worker once the retry is due."""
        exponent = min(claim.control.application_retry_count, 30)
        delay = self.config.retry_base_delay_seconds * (2**exponent) if error.delay is None else error.delay
        delay_ms = math.ceil(min(delay, self.config.retry_max_delay_seconds) * 1_000)
        plan = await claim.transition(
            ScheduleRetry(
                fence=claim.control.fence,
                worker_id=worker_id,
                now_ms=0,
                delay_ms=delay_ms,
                max_application_retries=self.config.max_application_retries,
                error=self._encode_exception(error, retryable=True),
                progress_events=error.progress_events,
            )
        )
        if plan.next_control.status is ExecutionStatus.QUEUED:
            asyncio.get_running_loop().call_later(delay_ms / 1_000, self._work_available.set)

    async def _acknowledge_cancellation(
        self,
        claim: _ClaimedExecution,
        context: DurableContext,
        worker_id: str,
    ) -> None:
        """Commit pending progress through the reducer's cancellation-wins rule."""
        await claim.transition(
            Complete(
                fence=claim.control.fence,
                worker_id=worker_id,
                now_ms=0,
                result=b"null",
                progress_events=context._progress_events,
            )
        )

    async def _invoke_application(
        self,
        claim: _ClaimedExecution,
        context: DurableContext,
        request: BaseModel,
    ) -> object:
        with durable_context_scope(context):
            application = (
                asyncio.ensure_future(cast(Awaitable[object], self.runner(context, request)))
                if self._runner_is_async
                else claim.start_thread(
                    lambda: self.runner(context, request), name=f"durable-run:{claim.control.run_id}"
                )
            )
            claim.application = application

        lease_watch = asyncio.create_task(
            claim.lease_lost.wait(),
            name=f"durable-lease-watch:{claim.control.run_id}",
        )
        try:
            done, _ = await asyncio.wait({application, lease_watch}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            self._cancel_application(application)
            raise
        finally:
            lease_watch.cancel()
            with suppress(asyncio.CancelledError):
                await lease_watch
        if lease_watch in done and claim.lease_lost.is_set():
            self._cancel_application(application)
            raise ExecutionLeaseLostError(f"execution lease for '{claim.control.run_id}' was lost")
        try:
            return application.result()
        except asyncio.CancelledError as error:
            # Shutdown cancellation hands the claim back; spontaneous application cancellation is a failure.
            if claim.application_cancelled:
                raise
            raise RuntimeError("the durable application was cancelled") from error

    def _cancel_application(self, application: asyncio.Future[object]) -> None:
        """Stop waiting for the application; cancellation-resistant async work stays tracked until it exits."""
        application.cancel()
        if application.done():
            return
        _track(self._draining_runs, application)
        application.add_done_callback(lambda done: None if done.cancelled() else done.exception())

    async def _backoff_worker(self, worker_id: str, error: ExecutionStoreError, operation: str) -> None:
        self._worker_store_error_streaks[worker_id] += 1
        await self._backoff(error, self._worker_store_error_streaks[worker_id], operation)

    def _encode_error(
        self,
        error_type: str,
        message: str,
        *,
        retryable: bool = False,
        code: str | None = None,
    ) -> bytes:
        value = PersistedError(
            type=error_type,
            message=message,
            retryable=retryable,
            code=code,
        )
        try:
            return encode_json(value.model_dump(mode="json"), max_bytes=self.store.config.max_payload_bytes)
        except ExecutionPayloadSizeError:
            return self._fallback_error

    def _encode_exception(
        self,
        error: BaseException,
        *,
        retryable: bool = False,
        code: str | None = None,
    ) -> bytes:
        return self._encode_error(
            type(error).__name__,
            "Durable execution failed",
            retryable=retryable,
            code=code,
        )

    async def _backoff(self, error: BaseException, streak: int, operation: str) -> None:
        ceiling = min(
            self.config.operational_backoff_max_seconds,
            self.config.operational_backoff_min_seconds * (2 ** min(streak - 1, 20)),
        )
        delay = random.uniform(self.config.operational_backoff_min_seconds, ceiling)  # noqa: S311
        log.bind(
            deployment=self.name,
            operation=operation,
            exception_type=type(error).__name__,
        ).warning("Durable store operation failed; retrying")
        await asyncio.sleep(delay)


class DurableRuntime:
    """Application-owned, fixed set of portable durable deployments."""

    def __init__(self, deployments: Iterable[DurableDeployment] = ()) -> None:
        members: dict[str, DurableDeployment] = {}
        for deployment in deployments:
            if deployment.name in members:
                raise ValueError(f"durable deployment '{deployment.name}' is listed more than once")
            members[deployment.name] = deployment
        self._deployments = MappingProxyType(members)
        self._started = False
        self._closed = False

    async def start(self) -> None:
        """Start every deployment; if one fails, close them all and re-raise, leaving them drainable."""
        if self._closed:
            raise RuntimeError("a closed durable runtime cannot be restarted")
        if self._started:
            return
        self._started = True
        try:
            for deployment in self._deployments.values():
                await deployment.start()
        except BaseException:
            # close() logs its own failures; the start failure is the one to report.
            with suppress(Exception):
                await self.close()
            raise

    async def close(self) -> None:
        """Close every deployment in reverse order, even when one fails, then raise the first failure."""
        self._closed = True
        first_error: Exception | None = None
        for deployment in reversed(tuple(self._deployments.values())):
            try:
                await deployment.close()
            except Exception as error:
                log.opt(exception=error).bind(deployment=deployment.name, exception_type=type(error).__name__).error(
                    "Durable deployment failed to close"
                )
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    async def wait_drained(self) -> None:
        """Wait for every deployment's retained work; call after close() and before releasing shared clients."""
        for deployment in self._deployments.values():
            await deployment.wait_drained()

    async def health(self) -> dict[str, object]:
        deployments = dict(
            zip(
                self._deployments,
                await asyncio.gather(*(deployment.health() for deployment in self._deployments.values())),
                strict=True,
            )
        )
        return {
            "healthy": all(bool(health["healthy"]) for health in deployments.values()),
            "deployments": deployments,
        }
