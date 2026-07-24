"""Durable worker lifecycle and execution state transitions."""

from __future__ import annotations

import asyncio
import random
import socket
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TypeAlias, cast

from hayhooks.durable.context import (
    DurableAdapter,
    DurableContext,
    DurableExecutionStore,
    ExecutionClaim,
    execution_context_scope,
)
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
    utc_now,
    validate_json,
)
from hayhooks.server.logger import log

RecordRunner: TypeAlias = Callable[[DurableContext], Awaitable[JsonValue]]


class DurableExecutionManager:
    """Bounded worker manager shared by REST and A2A adapters."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        store: DurableExecutionStore,
        runner: RecordRunner,
        adapter: DurableAdapter,
        *,
        concurrency: int = 1,
        poll_interval: float = 0.05,
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
        self._prepared = False
        self._started = False
        self._accepting_claims = False
        self._worker_generation = 0
        self._store_error_count = 0
        self._last_successful_claim_at: datetime | None = None
        self._active_claims = 0
        self._metrics = {
            "attempts_started": 0,
            "completed": 0,
            "failed": 0,
            "canceled": 0,
            "suspended": 0,
            "retries_scheduled": 0,
            "retries_exhausted": 0,
            "lease_losses": 0,
            "worker_restarts": 0,
            "record_size_failures": 0,
        }

    async def start(self) -> None:
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
        self._worker_generation += 1
        generation = self._worker_generation
        identity = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._workers = [self._start_worker(identity, slot, generation) for slot in range(self.concurrency)]

    def deactivate(self) -> None:
        self._accepting_claims = False
        self._worker_generation += 1

    @property
    def started(self) -> bool:
        return self._started

    @property
    def accepting(self) -> bool:
        return self._started and self._accepting_claims

    @property
    def health(self) -> dict[str, JsonValue]:
        """Return a payload-safe worker projection for readiness and diagnostics."""
        running = sum(not worker.done() for worker in self._workers)
        return {
            "healthy": not self._prepared or (self._started and self._accepting_claims and running == self.concurrency),
            "configured_slots": self.concurrency,
            "running_slots": running,
            "draining_slots": sum(not worker.done() for worker in self._draining_workers),
            "store_error_count": self._store_error_count,
            "last_successful_claim_at": (
                self._last_successful_claim_at.isoformat() if self._last_successful_claim_at else None
            ),
            "accepting": self.accepting,
            "active_claims": self._active_claims,
            "metrics": cast(JsonValue, dict(self._metrics)),
        }

    async def health_snapshot(self) -> dict[str, JsonValue]:
        """Add storage-level queue/state counts to the local readiness view."""
        health = self.health
        counts = getattr(self.store, "operational_counts", None)
        if not callable(counts):
            return health
        try:
            health["counts"] = cast(JsonValue, await counts())
        except Exception as error:
            health["healthy"] = False
            health["operational_error"] = type(error).__name__
        return health

    @property
    def draining(self) -> bool:
        return any(not worker.done() for worker in self._draining_workers)

    async def wait_drained(self) -> None:
        """Wait for detached application work before closing shared storage."""
        pending = [worker for worker in self._draining_workers if not worker.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self) -> None:
        self._accepting_claims = False
        if not self._workers:
            self._started = False
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

    async def submit(self, record: ExecutionRecord) -> bool:
        return await self.store.submit(record)

    def _start_worker(self, identity: str, slot: int, generation: int) -> asyncio.Task[None]:
        worker = asyncio.create_task(
            self._worker(f"{self.name}:{identity}:{slot}", generation),
            name=f"durable:{self.name}:{slot}",
        )
        worker.add_done_callback(lambda completed: self._worker_done(identity, slot, generation, completed))
        return worker

    def _worker_done(
        self,
        identity: str,
        slot: int,
        generation: int,
        worker: asyncio.Task[None],
    ) -> None:
        if self._accepting_claims and generation == self._worker_generation and not worker.cancelled():
            error = worker.exception()
            if error is not None:
                self._metrics["worker_restarts"] += 1
                log.opt(exception=error).error(
                    "{} | durable worker slot {} stopped unexpectedly; restarting",
                    self.name,
                    slot,
                )
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
        consecutive_store_errors = 0
        while self._accepting_claims and generation == self._worker_generation:
            try:
                claim = await self.store.claim_next(worker_name)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                consecutive_store_errors += 1
                await self._backoff_store_error(error, consecutive_store_errors, operation="claim")
                continue
            if claim is None:
                await asyncio.sleep(self.poll_interval)
                continue
            self._last_successful_claim_at = utc_now()
            if not self._accepting_claims or generation != self._worker_generation:
                try:
                    async with claim:
                        pass
                except Exception as error:
                    consecutive_store_errors += 1
                    await self._backoff_store_error(error, consecutive_store_errors, operation="abandon")
                return
            try:
                self._active_claims += 1
                self._metrics["attempts_started"] += 1
                try:
                    await self._process_claim(claim)
                finally:
                    self._active_claims -= 1
                consecutive_store_errors = 0
            except ExecutionLeaseLostError:
                consecutive_store_errors = 0
                self._metrics["lease_losses"] += 1
                log.warning("{} | lost durable execution claim {}", self.name, claim.record.execution_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                consecutive_store_errors += 1
                await self._backoff_store_error(error, consecutive_store_errors, operation="transition")

    async def _process_claim(  # noqa: C901, PLR0912, PLR0915 - explicit durable transition table
        self,
        claim: ExecutionClaim,
    ) -> None:
        async with claim:
            try:
                context = DurableContext(claim, self.adapter)
                try:
                    if await claim.cancellation_requested():
                        claim.record.mark_canceled()
                        await claim.complete()
                        self._record_terminal_metric(claim.record)
                        return
                    with execution_context_scope(context):
                        result = await self.runner(context)
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
                    self._record_terminal_metric(claim.record)
                except ExecutionSuspendedError:
                    # ``DurableContext.suspend`` already persisted and released this claim.
                    if claim.record.terminal:
                        self._record_terminal_metric(claim.record)
                    else:
                        self._metrics["suspended"] += 1
                    return
                except ExecutionCanceledError:
                    claim.record.mark_canceled()
                    await claim.complete()
                    self._record_terminal_metric(claim.record)
                except RetryableExecutionError as error:
                    if claim.record.attempt >= self.max_attempts:
                        exhausted = ExecutionError(
                            type="RetryExhausted",
                            message=f"Execution exhausted its {self.max_attempts} permitted attempts",
                            retryable=False,
                            code="retry_exhausted",
                        )
                        claim.record.mark_failed(exhausted)
                        claim.record.append_progress("Execution retry limit reached", kind="retry_exhausted")
                        await claim.complete()
                        self._metrics["retries_exhausted"] += 1
                        self._record_terminal_metric(claim.record)
                        return
                    delay = self._retry_delay(claim.record.attempt, error.delay)
                    await claim.retry(ExecutionError.from_exception(error, retryable=True), delay=delay)
                    if claim.record.terminal:
                        self._record_terminal_metric(claim.record)
                    else:
                        self._metrics["retries_scheduled"] += 1
                except ExecutionRecordSizeError:
                    raise
                except ExecutionLeaseLostError:
                    raise
                except ExecutionStoreError:
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    claim.record.mark_failed(error)
                    await claim.complete()
                    self._record_terminal_metric(claim.record)
            except ExecutionRecordSizeError:
                await self._fail_oversized_record(claim)

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
        self._metrics["record_size_failures"] += 1
        self._record_terminal_metric(record)

    def _record_terminal_metric(self, record: ExecutionRecord) -> None:
        if record.status is ExecutionStatus.COMPLETED:
            self._metrics["completed"] += 1
        elif record.status is ExecutionStatus.FAILED:
            self._metrics["failed"] += 1
        elif record.status is ExecutionStatus.CANCELED:
            self._metrics["canceled"] += 1

    def _retry_delay(self, attempt: int, requested_delay: float) -> float:
        exponent = min(max(0, attempt - 1), 30)
        delay = requested_delay if requested_delay > 0 else self.retry_base_delay * (2**exponent)
        return min(max(0.0, delay), self.retry_max_delay)

    async def _backoff_store_error(self, error: BaseException, failures: int, *, operation: str) -> None:
        self._store_error_count += 1
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
