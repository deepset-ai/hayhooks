"""Durable worker lifecycle and execution state transitions."""

from __future__ import annotations

import asyncio
import random
import socket
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import TypeAlias, cast

from hayhooks.durable.adapters import HaystackDurableAdapter
from hayhooks.durable.context import DurableContext, execution_context_scope
from hayhooks.durable.models import (
    ExecutionCanceledError,
    ExecutionError,
    ExecutionLeaseLostError,
    ExecutionRecord,
    ExecutionRecordSizeError,
    ExecutionStatus,
    ExecutionStoreError,
    ExecutionSuspendedError,
    JsonValue,
    RetryableExecutionError,
    validate_json,
)
from hayhooks.durable.store import ExecutionClaim, ExecutionStore
from hayhooks.server.logger import log

RecordRunner: TypeAlias = Callable[[DurableContext], Awaitable[JsonValue]]


class SubmissionGate:
    """Atomically admit submissions or close and wait for admitted work."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._open = False
        self._active = 0

    @property
    def open(self) -> bool:
        return self._open

    def activate(self) -> None:
        self._open = True

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        async with self._condition:
            if not self._open:
                msg = "Durable deployment is not accepting submissions"
                raise RuntimeError(msg)
            self._active += 1
        try:
            yield
        finally:
            async with self._condition:
                self._active -= 1
                if not self._active:
                    self._condition.notify_all()

    async def close_and_wait(self) -> None:
        async with self._condition:
            self._open = False
            await self._condition.wait_for(lambda: self._active == 0)


class DurableExecutionManager:
    """Bounded worker manager shared by REST and A2A adapters."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        store: ExecutionStore,
        runner: RecordRunner,
        adapter: HaystackDurableAdapter,
        *,
        concurrency: int = 1,
        poll_interval: float = 1.0,
        shutdown_grace_period: float = 5.0,
        max_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        if concurrency < 1:
            msg = "durable execution concurrency must be at least one"
            raise ValueError(msg)
        self.name = name
        self.store = store
        self.runner = runner
        self.adapter = adapter
        self.concurrency = concurrency
        self.poll_interval = poll_interval
        self.shutdown_grace_period = shutdown_grace_period
        self.max_attempts = max(1, max_attempts)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self.retry_max_delay = max(self.retry_base_delay, retry_max_delay)
        self._workers: list[asyncio.Task[None]] = []
        self._draining_workers: set[asyncio.Task[None]] = set()
        self._draining_runs: set[asyncio.Future[JsonValue]] = set()
        self._maintenance_task: asyncio.Task[None] | None = None
        self._maintenance_error_streak = 0
        self._worker_store_error_streaks: dict[str, int] = {}
        self._prepared = False
        self._started = False
        self._accepting_claims = False
        self._worker_generation = 0
        self._submission_gate = SubmissionGate()

    async def start(self) -> None:
        """Prepare storage and activate this manager's workers."""
        if self._started:
            self._accepting_claims = True
            return
        await self.prepare()
        self.activate()

    async def prepare(self) -> None:
        """Initialize storage while keeping workers and submissions disabled."""
        if self._prepared:
            return
        await self.store.initialize()
        self._prepared = True
        log.bind(deployment=self.name).debug("Prepared durable execution store")

    def activate(self) -> None:
        """Start workers for an initialized deployment without an await gap."""
        if self._started:
            self._accepting_claims = True
            return
        if not self._prepared:
            msg = "durable execution manager must be prepared before activation"
            raise RuntimeError(msg)
        self._started = True
        self._accepting_claims = True
        self._submission_gate.activate()
        self._worker_generation += 1
        self._worker_store_error_streaks.clear()
        generation = self._worker_generation
        identity = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._workers = [self._start_worker(identity, slot, generation) for slot in range(self.concurrency)]
        self._start_maintenance(generation)
        log.bind(deployment=self.name, workers=self.concurrency).debug("Activated durable execution workers")

    def deactivate(self) -> None:
        self._accepting_claims = False
        self._worker_generation += 1

    async def quiesce(self) -> None:
        """Close admission before workers stop or stranded work is counted."""
        await self._submission_gate.close_and_wait()
        self.deactivate()
        log.bind(deployment=self.name).debug("Quiesced durable execution manager")

    @property
    def started(self) -> bool:
        return self._started

    @property
    def accepting(self) -> bool:
        return self._started and self._accepting_claims and self._submission_gate.open

    @property
    def health(self) -> dict[str, JsonValue]:
        """Return a payload-safe worker projection for readiness and diagnostics."""
        running = sum(not worker.done() for worker in self._workers)
        maintenance_running = self._maintenance_task is not None and not self._maintenance_task.done()
        maintenance_healthy = self._maintenance_task is None or (
            maintenance_running and self._maintenance_error_streak == 0
        )
        worker_store_error_streak = max(self._worker_store_error_streaks.values(), default=0)
        return {
            "healthy": not self._prepared
            or (
                self._started
                and self._accepting_claims
                and running == self.concurrency
                and maintenance_healthy
                and worker_store_error_streak == 0
            ),
            "configured_slots": self.concurrency,
            "running_slots": running,
            "draining_slots": sum(not worker.done() for worker in self._draining_workers),
            "draining_runs": sum(not runner.done() for runner in self._draining_runs),
            "maintenance_running": maintenance_running,
            "worker_store_error_streak": worker_store_error_streak,
            "accepting": self.accepting,
        }

    async def health_snapshot(self) -> dict[str, JsonValue]:
        """Add storage-level queue/state counts to the local readiness view."""
        health = self.health
        try:
            health["counts"] = cast(JsonValue, await self.store.operational_counts())
        except Exception as error:
            health["healthy"] = False
            health["operational_error"] = type(error).__name__
        return health

    @property
    def draining(self) -> bool:
        return any(not worker.done() for worker in self._draining_workers) or any(
            not runner.done() for runner in self._draining_runs
        )

    async def wait_drained(self) -> None:
        """Wait for detached application work before closing shared storage."""
        pending = [worker for worker in self._draining_workers if not worker.done()]
        pending_runs = [runner for runner in self._draining_runs if not runner.done()]
        if pending or pending_runs:
            await asyncio.gather(*pending, *pending_runs, return_exceptions=True)

    async def close(self) -> None:
        """Stop claims and retain lease-lost work until application code exits."""
        await self.quiesce()
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None
        if not self._workers:
            self._started = False
            log.bind(deployment=self.name).debug("Closed durable execution manager")
            return
        done, pending = await asyncio.wait(self._workers, timeout=self.shutdown_grace_period)
        if pending:
            self._draining_workers.update(pending)
            for worker in pending:
                worker.add_done_callback(self._draining_workers.discard)
            log.warning(
                "{} | {} durable worker slot(s) exceeded the {:.2f}s shutdown grace period; "
                "claims remain fenced and heartbeating until application work exits",
                self.name,
                len(pending),
                self.shutdown_grace_period,
            )
        self._log_worker_failures(done)
        self._workers = []
        self._started = False
        log.bind(deployment=self.name, draining=len(pending)).debug("Closed durable execution manager")

    async def submit_with_record(self, record: ExecutionRecord) -> tuple[bool, ExecutionRecord]:
        """Admit one submission while the deployment gate remains open."""
        async with self._submission_gate.admit():
            return await self.store.submit_with_record(record)

    def _start_worker(self, identity: str, slot: int, generation: int) -> asyncio.Task[None]:
        worker_name = f"{self.name}:{identity}:{slot}"
        worker = asyncio.create_task(
            self._worker(worker_name, generation),
            name=f"durable:{self.name}:{slot}",
        )
        worker.add_done_callback(lambda completed: self._worker_done(identity, slot, generation, completed))
        return worker

    def _start_maintenance(self, generation: int) -> None:
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(self.store.maintain, self.poll_interval, generation),
            name=f"durable-maintenance:{self.name}",
        )

    async def _maintenance_loop(
        self,
        maintain: Callable[[], Awaitable[None]],
        interval: float,
        generation: int,
    ) -> None:
        """Supervise bounded store maintenance independently from worker claims."""
        while self._accepting_claims and generation == self._worker_generation:
            try:
                await maintain()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._maintenance_error_streak += 1
                log.opt(exception=error).warning("{} | durable maintenance failed; retrying", self.name)
            else:
                self._maintenance_error_streak = 0
            await asyncio.sleep(interval)

    def _worker_done(
        self,
        identity: str,
        slot: int,
        generation: int,
        worker: asyncio.Task[None],
    ) -> None:
        if self._accepting_claims and generation == self._worker_generation:
            error = None if worker.cancelled() else worker.exception()
            if error is not None:
                log.opt(exception=error).error(
                    "{} | durable worker slot {} stopped unexpectedly; restarting",
                    self.name,
                    slot,
                )
            else:
                log.error("{} | durable worker slot {} stopped unexpectedly; restarting", self.name, slot)
            replacement = self._start_worker(identity, slot, generation)
            try:
                index = self._workers.index(worker)
            except ValueError:
                replacement.cancel()
            else:
                self._workers[index] = replacement

    def _log_worker_failures(self, workers: set[asyncio.Task[None]]) -> None:
        for worker in workers:
            if worker.cancelled():
                continue
            error = worker.exception()
            if error is not None:
                log.opt(exception=error).warning(
                    "{} | durable worker '{}' ended with an error during shutdown",
                    self.name,
                    worker.get_name(),
                )

    async def _worker(self, worker_name: str, generation: int) -> None:
        self._worker_store_error_streaks[worker_name] = 0
        while self._accepting_claims and generation == self._worker_generation:
            try:
                claim = await self.store.claim_next(worker_name)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._worker_store_error_streaks[worker_name] += 1
                await self._backoff_store_error(error, self._worker_store_error_streaks[worker_name], operation="claim")
                continue
            self._worker_store_error_streaks[worker_name] = 0
            if claim is None:
                await asyncio.sleep(self.poll_interval)
                continue
            if not self._accepting_claims or generation != self._worker_generation:
                return
            attempt_log = log.bind(
                deployment=self.name,
                execution_id=claim.record.execution_id,
                attempt=claim.record.attempt,
            )
            attempt_log.debug("Claimed durable execution")
            try:
                await self._process_claim(claim)
                self._worker_store_error_streaks[worker_name] = 0
                attempt_log.bind(status=claim.record.status.value).debug("Finished durable execution attempt")
            except ExecutionLeaseLostError:
                self._worker_store_error_streaks[worker_name] = 0
                log.warning("{} | lost durable execution claim {}", self.name, claim.record.execution_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._worker_store_error_streaks[worker_name] += 1
                await self._backoff_store_error(
                    error, self._worker_store_error_streaks[worker_name], operation="transition"
                )

    async def _process_claim(self, claim: ExecutionClaim) -> None:  # noqa: C901, PLR0912 - explicit attempt outcomes
        """Run one fenced claim and leave every terminal decision to the store."""
        async with claim:
            # A reclaimed delivery has already incremented ``attempt`` in the
            # store. Do not invoke application code again after the total
            # execution-attempt budget has been consumed by prior crashes,
            # lease losses, or explicit retries.
            if claim.record.attempt > self.max_attempts:
                await self._terminalize_attempt_exhaustion(claim)
                return

            context = DurableContext(claim, self.adapter)
            try:
                if await claim.cancellation_requested():
                    await self._complete_canceled(claim)
                    return

                with execution_context_scope(context):
                    result = await self._run_with_lease_guard(claim, context)
            except ExecutionSuspendedError:
                pass
            except ExecutionCanceledError:
                await self._complete_canceled(claim)
            except RetryableExecutionError as error:
                if claim.record.attempt >= self.max_attempts:
                    await self._terminalize_attempt_exhaustion(claim)
                    return

                exponent = min(max(0, claim.record.attempt - 1), 30)
                delay = error.delay if error.delay > 0 else self.retry_base_delay * (2**exponent)
                await claim.retry(
                    ExecutionError.from_exception(error, retryable=True),
                    delay=min(max(0.0, delay), self.retry_max_delay),
                )
            except ExecutionRecordSizeError:
                await self._fail_oversized_record(claim)
            except (ExecutionLeaseLostError, ExecutionStoreError, asyncio.CancelledError):
                raise
            except Exception as error:
                claim.record.mark_failed(error)
                await claim.complete()
            else:
                try:
                    # A cancellation accepted after the runner returns wins over
                    # the result, so clients never observe a completed canceled run.
                    if await claim.cancellation_requested():
                        claim.record.mark_canceled()
                    else:
                        claim.record.result = validate_json(result, limit=claim.record.max_record_bytes, label="result")
                        claim.record.error = None
                        claim.record.status = ExecutionStatus.COMPLETED
                        claim.record.wait = None
                        claim.record.retry_at = None

                    claim.record.touch()
                    await claim.complete()
                except ExecutionRecordSizeError:
                    await self._fail_oversized_record(claim)

    async def _complete_canceled(self, claim: ExecutionClaim) -> None:
        """Terminalize a cooperative cancellation with the current ownership fence."""
        claim.record.mark_canceled()
        await claim.complete()

    async def _terminalize_attempt_exhaustion(self, claim: ExecutionClaim) -> None:
        """Fail an execution without allowing an attempt beyond the configured bound."""
        exhausted = ExecutionError(
            type="RetryExhausted",
            message=f"Execution exhausted its {self.max_attempts} permitted attempts",
            retryable=False,
            code="retry_exhausted",
        )
        claim.record.mark_failed(exhausted)
        claim.record.append_progress("Execution retry limit reached", kind="retry_exhausted")
        await claim.complete()

    async def _run_with_lease_guard(self, claim: ExecutionClaim, context: DurableContext) -> JsonValue:
        """
        Stop cooperative work when a store reports definitive lease loss.

        Concrete stores expose an event that flips once the fenced lease is
        definitively lost.
        """
        lost_event = claim.lost_event
        runner = asyncio.ensure_future(self.runner(context))
        loss_waiter = asyncio.create_task(
            lost_event.wait(),
            name=f"durable-lease-watch:{self.name}:{claim.record.execution_id}",
        )
        try:
            done, _ = await asyncio.wait({runner, loss_waiter}, return_when=asyncio.FIRST_COMPLETED)
            if runner in done and not lost_event.is_set():
                return runner.result()
            if not runner.done():
                runner.cancel()
                self._track_draining_run(runner, claim.record.execution_id)
            else:
                # Retrieve a finished task's result so an application exception
                # is not reported as an unhandled background-task failure.
                with suppress(asyncio.CancelledError, Exception):
                    runner.result()
            msg = f"Execution lease for '{claim.record.execution_id}' was lost"
            raise ExecutionLeaseLostError(msg)
        except asyncio.CancelledError:
            if not runner.done():
                runner.cancel()
                self._track_draining_run(runner, claim.record.execution_id)
            raise
        finally:
            loss_waiter.cancel()
            with suppress(asyncio.CancelledError):
                await loss_waiter

    def _track_draining_run(self, runner: asyncio.Future[JsonValue], execution_id: str) -> None:
        """Retain non-cooperative work until it exits after a lost lease."""
        if runner.done() or runner in self._draining_runs:
            return
        self._draining_runs.add(runner)

        def completed(done: asyncio.Future[JsonValue]) -> None:
            self._draining_runs.discard(done)
            if done.cancelled():
                return
            with suppress(Exception):
                done.result()
            log.warning("{} | lease-lost durable work drained for execution {}", self.name, execution_id)

        runner.add_done_callback(completed)

    async def _fail_oversized_record(self, claim: ExecutionClaim) -> None:
        """Persist a small terminal record after application state exceeded its bound."""
        record = claim.record
        record.validated_input = {}
        record.checkpoint = None
        record.application_state = {}
        record.wait = None
        record.progress = []
        record.result = None
        record.error = None
        record.last_retry_error = None
        record.retry_at = None
        record.mark_failed(
            ExecutionError(
                type="ExecutionRecordTooLarge",
                message=f"Execution exceeded its {record.max_record_bytes}-byte durable record limit",
                retryable=False,
                code="record_too_large",
            )
        )
        await claim.complete()

    async def _backoff_store_error(self, error: BaseException, failures: int, *, operation: str) -> None:
        exponent = min(max(0, failures - 1), 10)
        ceiling = min(max(self.poll_interval, 0.01) * (2**exponent), 5.0)
        delay = random.uniform(ceiling / 2, ceiling)  # noqa: S311 - jitter is not security-sensitive
        log.opt(exception=error).warning(
            "{} | durable worker store {} failed; retrying in {:.2f}s: {}",
            self.name,
            operation,
            delay,
            error,
        )
        await asyncio.sleep(delay)
