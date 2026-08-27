"""Portable durable deployment and worker runtime."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import random
import secrets
from collections.abc import Awaitable, Callable
from concurrent.futures import Future as ThreadFuture
from contextlib import suppress
from contextvars import copy_context
from dataclasses import dataclass
from threading import Thread
from typing import Any, TypeAlias, cast

from loguru import logger as log
from pydantic import BaseModel

from hayhooks.durable.context import (
    DurableContext,
    DurableExecutionCancelledError,
    _ClaimedExecution,
    _ExecutionSuspendedError,
    _RetryRequestedError,
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
    SubmissionResult,
)

DurableRunner: TypeAlias = Callable[[DurableContext, BaseModel], object]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Worker, lease, retry, and operational retry limits."""

    worker_concurrency: int = 1
    poll_interval_seconds: float = 1.0
    maintenance_interval_seconds: float = 1.0
    shutdown_grace_seconds: float = 5.0
    lease_duration_ms: int = 30_000
    max_run_attempts: int = 3
    max_application_retries: int = 2
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 60.0
    operational_backoff_min_seconds: float = 0.05
    operational_backoff_max_seconds: float = 5.0

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


class DurableDeployment:
    """One typed durable callable backed by an explicit execution store."""

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
        self._revision_error = self._encode_error(
            "DefinitionRevisionError",
            "definition revision is incompatible",
            code="definition_revision",
        )
        self._attempts_error = self._encode_error(
            "RunAttemptsExhaustedError",
            "run attempts exhausted",
            code="run_attempts_exhausted",
        )
        self._runner_is_async = inspect.iscoroutinefunction(runner) or inspect.iscoroutinefunction(
            type(runner).__call__
        )
        self._submission_condition = asyncio.Condition()
        self._admitted_submissions = 0
        self._accepting_submissions = False
        self._accepting_claims = False
        self._started = False
        self._closed = False
        self._generation = 0
        self._worker_identity = ""
        self._workers: dict[int, asyncio.Task[None]] = {}
        self._thread_workers: set[asyncio.Task[None]] = set()
        self._draining_workers: set[asyncio.Task[None]] = set()
        self._draining_runs: set[asyncio.Future[Any]] = set()
        self._worker_store_error_streaks: dict[str, int] = {}
        self._maintenance_error_streak = 0
        self._maintenance_task: asyncio.Task[None] | None = None

    @property
    def accepting(self) -> bool:
        return self._started and self._accepting_submissions and self._accepting_claims

    async def start(self) -> None:
        """Initialize storage and idempotently activate submissions and workers."""
        if self._closed:
            raise RuntimeError("a closed durable deployment cannot be restarted")
        if not self._started:
            await self.store.initialize()
            self._started = True
        if self.accepting:
            return
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance_task
        self._generation += 1
        self._worker_identity = secrets.token_hex(8)
        self._accepting_submissions = True
        self._accepting_claims = True
        self._worker_store_error_streaks.clear()
        self._ensure_workers()
        self._maintenance_task = asyncio.create_task(
            self._maintenance(self._generation),
            name=f"durable-maintenance:{self.name}",
        )

    async def quiesce(self) -> None:
        """Close admission, wait for admitted submissions, and stop new claims."""
        async with self._submission_condition:
            self._accepting_submissions = False
            self._accepting_claims = False
            self._generation += 1
            await self._submission_condition.wait_for(lambda: self._admitted_submissions == 0)

    async def close(self) -> None:
        """Stop maintenance and workers, retaining thread-backed work until it exits."""
        if self._closed:
            return
        await self.quiesce()
        self._closed = True
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None
        workers = [worker for worker in self._workers.values() if not worker.done()]
        if workers:
            _, pending = await asyncio.wait(workers, timeout=self.config.shutdown_grace_seconds)
            cancellable = pending - self._thread_workers
            for worker in cancellable:
                worker.cancel()
            if cancellable:
                await asyncio.sleep(0)
            pending = {worker for worker in pending if not worker.done()}
            self._draining_workers.update(pending)
            for worker in pending:
                worker.add_done_callback(self._draining_workers.discard)
        self._workers.clear()
        self._started = False

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
            if not self._accepting_submissions:
                raise RuntimeError(f"durable deployment '{self.name}' is not accepting submissions")
            self._admitted_submissions += 1
        try:
            return await self.store.submit(control, input_payload)
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
        """Read one execution after deployment, owner, and revision checks."""
        stored = await self.store.read(run_id)
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
        return await self.store.transition(run_id, RequestCancellation(0, reason))

    async def resume(
        self,
        run_id: str,
        resume_input: object = None,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = True,
    ) -> TransitionPlan:
        """Validate resume input and atomically requeue a waiting execution."""
        stored = await self.get(run_id, owner_id=owner_id, enforce_owner=enforce_owner)
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
        return await self.store.transition(
            run_id,
            Resume(
                0,
                self.revision,
                encode_json(checkpoint.model_dump(mode="json"), max_bytes=self.store.config.max_payload_bytes),
                expected_version=stored.control.version,
            ),
        )

    async def health(self) -> dict[str, object]:
        """Return local worker state plus bounded store counts."""
        running = sum(not worker.done() for worker in self._workers.values())
        maintenance_running = self._maintenance_task is not None and not self._maintenance_task.done()
        worker_error_streak = max(self._worker_store_error_streaks.values(), default=0)
        health: dict[str, object] = {
            "healthy": (
                self._started
                and self.accepting
                and running == self.config.worker_concurrency
                and maintenance_running
                and not worker_error_streak
                and not self._maintenance_error_streak
            ),
            "configured_slots": self.config.worker_concurrency,
            "running_slots": running,
            "draining_slots": sum(not worker.done() for worker in self._draining_workers),
            "draining_runs": sum(not run.done() for run in self._draining_runs),
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
            if not self._accepting_claims or slot in self._workers:
                continue
            worker_id = f"{self._worker_identity}-{slot}"
            worker = asyncio.create_task(
                self._worker(worker_id, self._generation),
                name=f"durable:{self.name}:{slot}",
            )
            self._workers[slot] = worker
            worker.add_done_callback(self._worker_stopped)

    def _worker_stopped(self, _worker: asyncio.Task[None]) -> None:
        """Restore worker capacity immediately without tying supervision to Redis maintenance."""
        self._ensure_workers()

    async def _maintenance(self, generation: int) -> None:
        while self._accepting_claims and generation == self._generation:
            try:
                await self.store.maintain(
                    max_run_attempts=self.config.max_run_attempts,
                    worker_revision=self.revision,
                    revision_error=self._revision_error,
                    attempts_error=self._attempts_error,
                )
            except asyncio.CancelledError:
                raise
            except ExecutionStoreError as error:
                self._maintenance_error_streak += 1
                await self._backoff(error, self._maintenance_error_streak, "maintenance")
            else:
                self._maintenance_error_streak = 0
                await asyncio.sleep(self.config.maintenance_interval_seconds)

    async def _worker(self, worker_id: str, generation: int) -> None:
        self._worker_store_error_streaks[worker_id] = 0
        while self._accepting_claims and generation == self._generation:
            control = await self._claim_next_execution(worker_id)
            if control is None:
                continue

            try:
                stored = await self._read_claimed_execution(control, worker_id)
                if stored is None:
                    continue
                prepared = await self._prepare_execution(stored, worker_id)
                if prepared is None:
                    continue
                claim, context, request = prepared
                await self._execute_claim(claim, context, request, worker_id)
            except asyncio.CancelledError:
                raise
            except ExecutionLeaseLostError:
                continue
            except ExecutionStoreError as error:
                await self._backoff_worker(worker_id, error, "transition")

    async def _claim_next_execution(self, worker_id: str) -> ExecutionControl | None:
        try:
            claimed = await self.store.claim(
                Claim(
                    worker_id,
                    0,
                    self.config.lease_duration_ms,
                    self.config.max_run_attempts,
                    self.revision,
                    self._revision_error,
                    self._attempts_error,
                )
            )
        except ExecutionStoreError as error:
            await self._backoff_worker(worker_id, error, "claim")
            return None
        self._worker_store_error_streaks[worker_id] = 0
        if claimed is None:
            await asyncio.sleep(self.config.poll_interval_seconds)
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
                await self.store.transition(control.run_id, ReleaseClaim(control.fence, worker_id))
            await self._backoff_worker(worker_id, error, "read")
            return None
        if stored is not None:
            return stored
        try:
            await self.store.transition(control.run_id, ReleaseClaim(control.fence, worker_id))
        except (ExecutionLeaseLostError, ExecutionNotFoundError):
            pass
        except ExecutionStoreError as error:
            await self._backoff_worker(worker_id, error, "release")
        return None

    async def _prepare_execution(
        self,
        stored: StoredExecution,
        worker_id: str,
    ) -> tuple[_ClaimedExecution, DurableContext, BaseModel] | None:
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
                else CheckpointEnvelope(adapter_kind=self.kind, adapter_checkpoint=None)
            )
            if checkpoint.adapter_kind is not self.kind:
                raise ValueError("checkpoint kind does not match the deployment")
        except (KeyError, TypeError, ValueError, ExecutionPayloadSizeError) as error:
            await self.store.transition(
                control.run_id,
                Fail(control.fence, worker_id, 0, self._encode_exception(error)),
            )
            return None

        claim = _ClaimedExecution(
            self.store,
            control,
            worker_id,
            self.config.lease_duration_ms,
            checkpoint,
        )
        context = DurableContext(claim)
        context._adapter = self.adapter
        return claim, context, request

    async def _execute_claim(
        self,
        claim: _ClaimedExecution,
        context: DurableContext,
        request: BaseModel,
        worker_id: str,
    ) -> None:
        async with claim:
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
                        claim.control.fence,
                        worker_id,
                        0,
                        encode_json(result, max_bytes=self.store.config.max_payload_bytes),
                        tuple(context._pending_progress),
                    )
                )
            except _ExecutionSuspendedError:
                return
            except DurableExecutionCancelledError:
                await self._acknowledge_cancellation(claim, context, worker_id)
            except _RetryRequestedError as error:
                exponent = min(claim.control.application_retry_count, 30)
                delay = self.config.retry_base_delay_seconds * (2**exponent) if error.delay is None else error.delay
                await claim.transition(
                    ScheduleRetry(
                        claim.control.fence,
                        worker_id,
                        0,
                        math.ceil(min(delay, self.config.retry_max_delay_seconds) * 1_000),
                        self.config.max_application_retries,
                        self._encode_exception(error, retryable=True),
                        error.progress_events,
                    )
                )
            except ExecutionPayloadSizeError as error:
                await claim.transition(
                    Fail(
                        claim.control.fence,
                        worker_id,
                        0,
                        self._encode_exception(error, code="payload_too_large"),
                        tuple(context._pending_progress),
                    )
                )
            except (asyncio.CancelledError, ExecutionLeaseLostError, ExecutionStoreError):
                raise
            except Exception as error:
                await claim.transition(
                    Fail(
                        claim.control.fence,
                        worker_id,
                        0,
                        self._encode_exception(error),
                        tuple(context._pending_progress),
                    )
                )

    async def _acknowledge_cancellation(
        self,
        claim: _ClaimedExecution,
        context: DurableContext,
        worker_id: str,
    ) -> None:
        """Commit pending progress through the reducer's cancellation-wins rule."""
        await claim.transition(
            Complete(
                claim.control.fence,
                worker_id,
                0,
                b"null",
                tuple(context._pending_progress),
            )
        )

    async def _invoke_application(
        self,
        claim: _ClaimedExecution,
        context: DurableContext,
        request: BaseModel,
    ) -> object:
        thread_done: asyncio.Event | None = None
        worker_task = asyncio.current_task()
        with durable_context_scope(context):
            if self._runner_is_async:
                application = asyncio.ensure_future(cast(Awaitable[object], self.runner(context, request)))
            else:
                thread_done = asyncio.Event()
                event_loop = asyncio.get_running_loop()
                thread_result: ThreadFuture[object] = ThreadFuture()
                thread_result.set_running_or_notify_cancel()
                application = asyncio.wrap_future(thread_result)
                active_context = copy_context()

                def run_in_thread(
                    bound_context: DurableContext = context,
                    bound_request: BaseModel = request,
                    bound_loop: asyncio.AbstractEventLoop = event_loop,
                    bound_completion: asyncio.Event = thread_done,
                    bound_result: ThreadFuture[object] = thread_result,
                ) -> None:
                    try:
                        bound_result.set_result(self.runner(bound_context, bound_request))
                    except BaseException as error:
                        bound_result.set_exception(error)
                    finally:
                        with suppress(RuntimeError):
                            bound_loop.call_soon_threadsafe(bound_completion.set)

                # A daemon thread keeps shutdown bounded; Python cannot safely stop it once running.
                Thread(
                    target=active_context.run,
                    args=(run_in_thread,),
                    name=f"durable-run:{claim.control.run_id}",
                    daemon=True,
                ).start()
                if worker_task is not None:
                    self._thread_workers.add(worker_task)

        lease_watch = asyncio.create_task(
            claim.lease_lost.wait(),
            name=f"durable-lease-watch:{claim.control.run_id}",
        )
        try:
            done, _ = await asyncio.wait({application, lease_watch}, return_when=asyncio.FIRST_COMPLETED)
            if lease_watch in done and claim.lease_lost.is_set():
                self._cancel_application(application, thread_done)
                raise ExecutionLeaseLostError(f"execution lease for '{claim.control.run_id}' was lost")
            return application.result()
        except asyncio.CancelledError:
            self._cancel_application(application, thread_done)
            raise
        finally:
            lease_watch.cancel()
            with suppress(asyncio.CancelledError):
                await lease_watch
            if worker_task is not None:
                self._thread_workers.discard(worker_task)

    def _cancel_application(
        self,
        application: asyncio.Future[object],
        thread_done: asyncio.Event | None,
    ) -> None:
        application.cancel()
        if thread_done is not None and not thread_done.is_set():
            draining: asyncio.Future[Any] = asyncio.create_task(thread_done.wait())
        elif not application.done():
            draining = application
        else:
            return
        self._draining_runs.add(draining)
        draining.add_done_callback(self._draining_runs.discard)
        draining.add_done_callback(lambda done: None if done.cancelled() else done.exception())

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
        return self._encode_error(type(error).__name__, str(error), retryable=retryable, code=code)

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
    """Application-owned collection of portable durable deployments."""

    def __init__(self) -> None:
        self._deployments: dict[str, DurableDeployment] = {}
        self._started = False
        self._closed = False

    @property
    def started(self) -> bool:
        return self._started

    def add(self, deployment: DurableDeployment) -> None:
        """Register a deployment before this runtime starts."""
        if self._closed:
            raise RuntimeError("a closed durable runtime cannot install deployments")
        if self._started:
            raise RuntimeError("use install after the durable runtime has started")
        if deployment.name in self._deployments:
            raise ValueError(f"durable deployment '{deployment.name}' is already installed")
        self._deployments[deployment.name] = deployment

    def discard(self, name: str) -> DurableDeployment:
        """Drop an unstarted deployment that failed host-side publication."""
        if self._started:
            raise RuntimeError("use remove after the durable runtime has started")
        try:
            return self._deployments.pop(name)
        except KeyError:
            raise KeyError(f"durable deployment '{name}' is not installed") from None

    async def install(self, deployment: DurableDeployment) -> None:
        if not self._started:
            self.add(deployment)
            return
        if self._closed:
            raise RuntimeError("a closed durable runtime cannot install deployments")
        if deployment.name in self._deployments:
            raise ValueError(f"durable deployment '{deployment.name}' is already installed")
        self._deployments[deployment.name] = deployment
        try:
            await deployment.start()
        except BaseException:
            del self._deployments[deployment.name]
            raise

    async def remove(self, name: str, *, close: bool = True) -> DurableDeployment:
        """Remove a deployment, optionally retaining a quiesced instance for rollback."""
        if not self._started:
            return self.discard(name)
        try:
            deployment = self._deployments[name]
        except KeyError:
            raise KeyError(f"durable deployment '{name}' is not installed") from None
        if close:
            await deployment.close()
        del self._deployments[name]
        return deployment

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("a closed durable runtime cannot be restarted")
        if self._started:
            return
        started: list[DurableDeployment] = []
        try:
            for deployment in self._deployments.values():
                await deployment.start()
                started.append(deployment)
        except BaseException:
            for deployment in reversed(started):
                await deployment.quiesce()
            raise
        self._started = True

    async def close(self) -> None:
        if self._closed:
            return
        for deployment in reversed(tuple(self._deployments.values())):
            await deployment.close()
        self._started = False
        self._closed = True

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
