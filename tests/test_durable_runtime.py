"""Portable durable runtime behavior."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ValidationError

from hayhooks.durable.context import DurableContext
from hayhooks.durable.engine import (
    Claim,
    ExecutionLeaseLostError,
    ExecutionNotFoundError,
    ExecutionStatus,
    Heartbeat,
    PayloadKind,
    ReleaseClaim,
    initial_control,
)
from hayhooks.durable.models import PersistedError, decode_json
from hayhooks.durable.runtime import DurableDeployment, DurableRuntime, RuntimeConfig
from hayhooks.durable.store import (
    CHUNK_CURSOR_START,
    ExecutionIdempotencyConflictError,
    ExecutionStoreError,
    MemoryExecutionStore,
    StoreConfig,
    StoredExecution,
    SubmissionResult,
)
from tests.test_durable_haystack import requires_haystack_v3

# Another worker taking over a released run.
SUCCESSOR = Claim("successor", 0, 300, 3, "v1", b"{}")


class Request(BaseModel):
    value: int


class Result(BaseModel):
    value: int


class ResumeInput(BaseModel):
    value: int


class ControlledStore(MemoryExecutionStore):
    """Memory store with reusable synchronization and failure controls."""

    def __init__(self, deployment: str) -> None:
        super().__init__(deployment, config=StoreConfig(lease_commit_safety_ms=10))
        self.initialize_calls = 0
        self.block_submissions = False
        self.submission_started = asyncio.Event()
        self.submission_release = asyncio.Event()
        self.claim_error: BaseException | None = None
        self.read_error: BaseException | None = None
        self.maintenance_error: BaseException | None = None
        self.failure_seen = asyncio.Event()
        self.claim_calls = 0
        self.maintenance_calls = 0
        self.second_claim_seen = asyncio.Event()

    async def initialize(self) -> None:
        self.initialize_calls += 1

    async def submit(self, control, input_payload: bytes) -> SubmissionResult:
        if self.block_submissions:
            self.submission_started.set()
            await self.submission_release.wait()
        return await super().submit(control, input_payload)

    async def claim(self, command):
        self.claim_calls += 1
        if self.claim_calls >= 2:
            self.second_claim_seen.set()
        if self.claim_error is not None:
            error, self.claim_error = self.claim_error, None
            self.failure_seen.set()
            raise error
        return await super().claim(command)

    async def read(self, run_id: str) -> StoredExecution | None:
        if self.read_error is not None:
            error, self.read_error = self.read_error, None
            self.failure_seen.set()
            raise error
        return await super().read(run_id)

    async def maintain(
        self,
        *,
        max_run_attempts: int,
        attempts_error: bytes,
    ) -> int:
        self.maintenance_calls += 1
        if self.maintenance_error is not None:
            error, self.maintenance_error = self.maintenance_error, None
            self.failure_seen.set()
            raise error
        return await super().maintain(
            max_run_attempts=max_run_attempts,
            attempts_error=attempts_error,
        )


async def echo_runner(_context: DurableContext, request: BaseModel) -> Result:
    return Result(value=Request.model_validate(request).value)


async def wait_for_execution(
    deployment: DurableDeployment,
    run_id: str,
    predicate: Callable[[StoredExecution], bool],
    *,
    timeout: float = 1.0,
) -> StoredExecution:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        stored = await deployment.get(run_id, enforce_owner=False, allow_revision_mismatch=True)
        if predicate(stored):
            return stored
        await asyncio.sleep(0.005)
    message = f"execution '{run_id}' did not reach the expected state"
    raise AssertionError(message)


async def wait_for_health(
    deployment: DurableDeployment,
    predicate: Callable[[dict[str, object]], bool],
    *,
    timeout: float = 1.0,
) -> dict[str, object]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        health = await deployment.health()
        if predicate(health):
            return health
        await asyncio.sleep(0.005)
    message = "deployment health did not reach the expected state"
    raise AssertionError(message)


@pytest.fixture
async def deployment_factory():
    deployments: list[DurableDeployment] = []

    async def create(
        runner=echo_runner,
        *,
        store: MemoryExecutionStore | None = None,
        name: str | None = None,
        revision: str = "v1",
        result_model: type[BaseModel] | None = Result,
        resume_model: type[BaseModel] | None = None,
        config: RuntimeConfig | None = None,
        start: bool = True,
    ) -> DurableDeployment:
        name = name or (store.deployment if store is not None else f"jobs-{len(deployments)}")
        store = store or MemoryExecutionStore(name, config=StoreConfig(lease_commit_safety_ms=10))
        deployment = DurableDeployment(
            name,
            revision,
            store,
            Request,
            runner,
            result_model=result_model,
            resume_model=resume_model,
            config=config
            or RuntimeConfig(
                poll_interval_seconds=0.005,
                lease_duration_ms=300,
                operational_backoff_min_seconds=0.005,
                operational_backoff_max_seconds=0.01,
            ),
        )
        deployments.append(deployment)
        if start:
            await deployment.start()
        return deployment

    yield create

    for deployment in reversed(deployments):
        await deployment.close()


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"worker_concurrency": 0}, id="workers"),
        pytest.param({"poll_interval_seconds": float("nan")}, id="finite"),
        pytest.param({"maintenance_interval_seconds": 0}, id="maintenance"),
        pytest.param({"retry_base_delay_seconds": 2, "retry_max_delay_seconds": 1}, id="retry"),
        pytest.param({"operational_backoff_min_seconds": 0}, id="backoff"),
    ],
)
def test_runtime_config_rejects_invalid_limits(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RuntimeConfig(**changes)


def test_lease_config_leaves_a_safe_heartbeat_window() -> None:
    store = MemoryExecutionStore("jobs", config=StoreConfig(lease_commit_safety_ms=10))
    with pytest.raises(ValueError, match="safe heartbeat"):
        DurableDeployment("jobs", "v1", store, Request, echo_runner, config=RuntimeConfig(lease_duration_ms=20))


async def test_submission_is_detached_idempotent_and_owner_scoped(deployment_factory) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(_context: DurableContext, request: Request) -> Result:
        started.set()
        await release.wait()
        return Result(value=request.value + 1)

    deployment = await deployment_factory(runner)
    submitted = await deployment.submit({"value": 1}, owner_id="owner", idempotency_key="same")
    await asyncio.wait_for(started.wait(), timeout=1)
    replayed = await deployment.submit({"value": 1}, owner_id="owner", idempotency_key="same")
    assert submitted.created and not replayed.created
    assert replayed.control.run_id == submitted.control.run_id
    with pytest.raises(ExecutionIdempotencyConflictError):
        await deployment.submit({"value": 2}, owner_id="owner", idempotency_key="same")
    with pytest.raises(ExecutionNotFoundError):
        await deployment.get(submitted.control.run_id, owner_id="other")

    release.set()
    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    assert stored.control.status is ExecutionStatus.COMPLETED
    assert decode_json(stored.payloads[PayloadKind.RESULT], max_bytes=1_000) == {"value": 2}


async def test_cancellation_wins_the_result_race(deployment_factory) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(_context: DurableContext, request: Request) -> Result:
        started.set()
        await release.wait()
        return Result(value=request.value)

    deployment = await deployment_factory(runner)
    submitted = await deployment.submit({"value": 1})
    await asyncio.wait_for(started.wait(), timeout=1)
    await deployment.cancel(submitted.control.run_id, reason="stop")
    release.set()

    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    assert stored.control.status is ExecutionStatus.CANCELED
    assert PayloadKind.RESULT not in stored.payloads


async def test_retry_delay_and_application_budget(deployment_factory) -> None:
    attempts = 0
    first_attempt = asyncio.Event()

    async def runner(context: DurableContext, _request: Request) -> None:
        nonlocal attempts
        attempts += 1
        first_attempt.set()
        await context.retry("again")

    deployment = await deployment_factory(
        runner,
        result_model=None,
        config=RuntimeConfig(
            poll_interval_seconds=0.005,
            lease_duration_ms=300,
            max_application_retries=1,
            retry_base_delay_seconds=0.04,
            retry_max_delay_seconds=0.04,
            operational_backoff_min_seconds=0.005,
            operational_backoff_max_seconds=0.01,
        ),
    )
    submitted = await deployment.submit({"value": 1})
    await asyncio.wait_for(first_attempt.wait(), timeout=1)
    queued = await wait_for_execution(
        deployment,
        submitted.control.run_id,
        lambda value: value.control.application_retry_count == 1,
    )
    assert queued.control.available_at_ms == queued.control.updated_at_ms + 40

    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    error = PersistedError.model_validate(decode_json(stored.payloads[PayloadKind.ERROR], max_bytes=1_000))
    assert (stored.control.status, stored.control.run_attempt, attempts, error.retryable) == (
        ExecutionStatus.FAILED,
        2,
        2,
        True,
    )


async def test_explicit_zero_retry_delay_is_immediate(deployment_factory) -> None:
    attempts = 0

    async def runner(context: DurableContext, request: Request) -> Result:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await context.retry("now", delay=0)
        return Result(value=request.value)

    deployment = await deployment_factory(
        runner,
        config=RuntimeConfig(
            poll_interval_seconds=0.005,
            lease_duration_ms=300,
            retry_base_delay_seconds=60,
            retry_max_delay_seconds=60,
        ),
    )
    submitted = await deployment.submit({"value": 1})

    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    assert (stored.control.status, stored.control.run_attempt, attempts) == (ExecutionStatus.COMPLETED, 2, 2)


async def test_failed_post_claim_read_releases_without_consuming_attempt(deployment_factory) -> None:
    store = ControlledStore("jobs")
    deployment = await deployment_factory(store=store)
    store.read_error = ExecutionStoreError("unavailable")
    submitted = await deployment.submit({"value": 1})
    await asyncio.wait_for(store.failure_seen.wait(), timeout=1)

    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    assert (stored.control.status, stored.control.run_attempt) == (ExecutionStatus.COMPLETED, 1)


async def test_exhausted_claim_fails_without_running_application(deployment_factory) -> None:
    calls = 0

    async def runner(_context: DurableContext, _request: Request) -> Result:
        nonlocal calls
        calls += 1
        return Result(value=1)

    store = MemoryExecutionStore(
        "jobs",
        config=StoreConfig(lease_commit_safety_ms=10),
    )
    control = replace(
        initial_control(
            run_id="run_1",
            idempotency_digest="idem",
            idempotency_binding_digest="binding",
            deployment="jobs",
            definition_revision="v1",
            owner_id=None,
            kind="pipeline",
            now_ms=0,
        ),
        run_attempt=1,
    )
    await store.submit(control, b'{"value":1}')
    deployment = await deployment_factory(
        runner,
        store=store,
        config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300, max_run_attempts=1),
    )

    stored = await wait_for_execution(deployment, control.run_id, lambda value: value.control.terminal)
    error = PersistedError.model_validate(
        decode_json(stored.payloads[PayloadKind.ERROR], max_bytes=store.config.max_payload_bytes)
    )
    assert stored.control.status is ExecutionStatus.FAILED
    assert (calls, error.code) == (0, "run_attempts_exhausted")


async def test_typed_resume_reconstructs_waiting_execution(deployment_factory) -> None:
    async def runner(context: DurableContext, _request: Request) -> Result:
        resume_input = context.resume_input
        if resume_input is None:
            await context.suspend({"kind": "approval", "message": "Continue?"})
        assert isinstance(resume_input, dict)
        return Result(value=int(resume_input["value"]))

    deployment = await deployment_factory(runner, resume_model=ResumeInput)
    submitted = await deployment.submit({"value": 1})
    waiting = await wait_for_execution(
        deployment,
        submitted.control.run_id,
        lambda value: value.control.status is ExecutionStatus.WAITING,
    )
    with pytest.raises(ValidationError):
        await deployment.resume(waiting.control.run_id, {"value": "invalid"})
    await deployment.resume(waiting.control.run_id, {"value": 7})

    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    assert decode_json(stored.payloads[PayloadKind.RESULT], max_bytes=1_000) == {"value": 7}


async def test_oversized_output_becomes_a_bounded_failure(deployment_factory) -> None:
    async def runner(_context: DurableContext, _request: Request) -> dict[str, str]:
        return {"large": "x" * 512}

    store = MemoryExecutionStore(
        "jobs",
        config=StoreConfig(lease_commit_safety_ms=10, max_payload_bytes=256),
    )
    deployment = await deployment_factory(runner, store=store, result_model=None)
    submitted = await deployment.submit({"value": 1})
    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    error = PersistedError.model_validate(decode_json(stored.payloads[PayloadKind.ERROR], max_bytes=256))
    assert (stored.control.status, error.code) == (ExecutionStatus.FAILED, "payload_too_large")


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(RuntimeError("password=runtime-secret"), id="application"),
        pytest.param(ValueError("https://example.test/?token=query-secret"), id="validation"),
    ],
)
async def test_exception_details_are_not_persisted(deployment_factory, error: Exception) -> None:
    async def runner(_context: DurableContext, _request: Request) -> Result:
        raise error

    deployment = await deployment_factory(runner)
    submitted = await deployment.submit({"value": 1})
    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    payload = stored.payloads[PayloadKind.ERROR]
    persisted = PersistedError.model_validate(decode_json(payload, max_bytes=1_000))

    assert persisted.message == "Durable execution failed"
    assert b"runtime-secret" not in payload and b"query-secret" not in payload


async def test_quiesce_waits_for_admitted_submission_and_rejects_later_work(deployment_factory) -> None:
    store = ControlledStore("jobs")
    store.block_submissions = True
    deployment = await deployment_factory(store=store)
    submission = asyncio.create_task(deployment.submit({"value": 1}))
    await asyncio.wait_for(store.submission_started.wait(), timeout=1)
    quiesce = asyncio.create_task(deployment.quiesce())
    await asyncio.sleep(0)
    assert not quiesce.done()
    with pytest.raises(RuntimeError, match="not accepting"):
        await deployment.submit({"value": 2})

    store.submission_release.set()
    submitted = await submission
    await quiesce
    assert submitted.created and not deployment.accepting


@pytest.mark.parametrize("operation", ["claim", "maintenance"])
async def test_store_error_health_streak_clears_after_success(deployment_factory, operation: str) -> None:
    store = ControlledStore("jobs")
    setattr(store, f"{operation}_error", ExecutionStoreError("unavailable"))
    deployment = await deployment_factory(
        store=store,
        config=RuntimeConfig(
            poll_interval_seconds=0.005,
            lease_duration_ms=300,
            operational_backoff_min_seconds=0.1,
            operational_backoff_max_seconds=0.1,
        ),
    )
    await asyncio.wait_for(store.failure_seen.wait(), timeout=1)
    assert (await deployment.health())["store_error_streak"] == 1
    await wait_for_health(deployment, lambda health: health["store_error_streak"] == 0)


@pytest.mark.parametrize("exit_mode", ["cancel", "crash"])
async def test_worker_slots_restart(deployment_factory, exit_mode: str) -> None:
    store = ControlledStore("jobs")
    if exit_mode == "crash":
        store.claim_error = RuntimeError("worker bug")
    deployment = await deployment_factory(
        store=store,
        config=RuntimeConfig(
            worker_concurrency=2,
            poll_interval_seconds=0.005,
            maintenance_interval_seconds=60,
            lease_duration_ms=300,
        ),
    )
    workers = set(deployment._workers.values())
    if exit_mode == "cancel":
        deployment._workers[0].cancel()
    await wait_for_health(
        deployment,
        lambda health: health["running_slots"] == 2 and bool(set(deployment._workers.values()) - workers),
    )
    assert set(deployment._workers) == {0, 1}


async def test_worker_and_maintenance_loops_use_independent_intervals(deployment_factory) -> None:
    store = ControlledStore("jobs")
    await deployment_factory(
        store=store,
        config=RuntimeConfig(
            poll_interval_seconds=0.001,
            maintenance_interval_seconds=60,
            lease_duration_ms=300,
        ),
    )

    await asyncio.wait_for(store.second_claim_seen.wait(), timeout=1)
    assert store.maintenance_calls == 1


async def test_runtime_membership_is_fixed_at_construction(deployment_factory) -> None:
    empty = DurableRuntime()
    await empty.start()
    assert await empty.health() == {"healthy": True, "deployments": {}}
    await empty.close()
    await empty.wait_drained()

    first = await deployment_factory(start=False)
    members = [first]
    runtime = DurableRuntime(members)
    members.append(await deployment_factory(start=False))
    assert list((await runtime.health())["deployments"]) == [first.name]
    assert not any(hasattr(runtime, name) for name in ("add", "install", "remove", "discard"))
    with pytest.raises(ValueError, match="more than once"):
        DurableRuntime((first, first))


async def test_runtime_start_failure_closes_started_deployments(deployment_factory) -> None:
    started = await deployment_factory(start=False)
    failing_store = ControlledStore("failing")
    failing_store.initialize = AsyncMock(side_effect=ExecutionStoreError("unavailable"))
    failing = await deployment_factory(store=failing_store, start=False)
    runtime = DurableRuntime((started, failing))

    with pytest.raises(ExecutionStoreError):
        await runtime.start()

    await asyncio.wait_for(runtime.wait_drained(), timeout=1)
    health = await started.health()
    assert (health["accepting"], health["running_slots"], health["maintenance_running"]) == (False, 0, False)
    with pytest.raises(RuntimeError, match="closed durable runtime"):
        await runtime.start()


@pytest.mark.parametrize("stop", ["quiesce", "close", "failed-activation"])
async def test_stopped_deployment_is_never_reactivated(deployment_factory, stop: str) -> None:
    store = ControlledStore("jobs")
    deployment = await deployment_factory(store=store, start=False)
    if stop == "failed-activation":
        store.initialize = AsyncMock(side_effect=ExecutionStoreError("unavailable"))
        with pytest.raises(ExecutionStoreError):
            await deployment.start()
    else:
        await deployment.start()
        await deployment.start()
        assert store.initialize_calls == 1
        # Repeated calls complete any cleanup an earlier call left unfinished.
        await getattr(deployment, stop)()
        await getattr(deployment, stop)()
    tasks = (tuple(deployment._workers.values()), deployment._maintenance_task)

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await deployment.start()

    assert not deployment.accepting
    assert (tuple(deployment._workers.values()), deployment._maintenance_task) == tasks
    await asyncio.wait_for(deployment.wait_drained(), timeout=1)
    with pytest.raises(RuntimeError, match="not accepting"):
        await deployment.submit({"value": 1})


async def test_quiesce_lets_an_in_flight_claim_finish_and_stops_claiming(deployment_factory, monkeypatch) -> None:
    store = ControlledStore("jobs")
    deployment = await deployment_factory(store=store)
    claim = store.claim
    claim_calls: list[object] = []
    claiming, release = asyncio.Event(), asyncio.Event()

    async def blocked_claim(command):
        claim_calls.append(command)
        claiming.set()
        await release.wait()
        return await claim(command)

    monkeypatch.setattr(store, "claim", blocked_claim)
    await asyncio.wait_for(claiming.wait(), timeout=1)
    submitted = await deployment.submit({"value": 1})
    await deployment.quiesce()
    release.set()

    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    await wait_for_health(deployment, lambda health: health["draining_slots"] == 0)
    assert stored.control.status is ExecutionStatus.COMPLETED
    assert len(claim_calls) == 1


async def test_runtime_close_reaches_every_deployment_and_raises_the_first_error(deployment_factory, monkeypatch):
    deployments = [await deployment_factory() for _ in range(3)]
    for deployment in deployments[1:]:
        close = deployment.close

        async def failing_close(close=close, name=deployment.name) -> None:
            await close()
            raise RuntimeError(name)

        monkeypatch.setattr(deployment, "close", failing_close)
    runtime = DurableRuntime(deployments)
    await runtime.start()

    # Deployments close in reverse order, so the last one fails first.
    with pytest.raises(RuntimeError, match=deployments[-1].name):
        await runtime.close()

    assert not any(deployment.accepting for deployment in deployments)
    await asyncio.wait_for(runtime.wait_drained(), timeout=1)


async def test_wait_drained_requires_closed_admission(deployment_factory) -> None:
    deployment = await deployment_factory()

    with pytest.raises(RuntimeError, match="close the durable deployment"):
        await deployment.wait_drained()


@pytest.mark.parametrize("cancel_requested", [False, True], ids=["requeued", "cancel-wins"])
async def test_close_releases_async_work_it_cancels(deployment_factory, cancel_requested: bool) -> None:
    started = asyncio.Event()

    async def runner(context: DurableContext, _request: Request) -> Result:
        await context.stream_chunk({"token": "partial"})
        started.set()
        # A nested task takes several loop iterations to process the cancellation.
        await asyncio.gather(asyncio.Event().wait())
        return Result(value=0)

    deployment = await deployment_factory(
        runner,
        config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300, shutdown_grace_seconds=0.01),
    )
    store = deployment.store
    run_id = (await deployment.submit({"value": 1})).control.run_id
    await asyncio.wait_for(started.wait(), timeout=1)
    if cancel_requested:
        await deployment.cancel(run_id)

    await deployment.close()

    stored = await store.read(run_id)
    assert stored is not None
    # Buffered chunks reach the stream before the claim is handed back.
    assert (await store.read_chunks(run_id, CHUNK_CURSOR_START))[0].data == b'{"token":"partial"}'
    if cancel_requested:
        assert stored.control.status is ExecutionStatus.CANCELED
        return
    assert (stored.control.status, stored.control.run_attempt) == (ExecutionStatus.QUEUED, 0)
    reclaimed = await store.claim(SUCCESSOR)
    assert reclaimed is not None and reclaimed.next_control.run_attempt == 1


@pytest.mark.parametrize("stage", ["read", "heartbeat"])
async def test_close_releases_claim_before_application_starts(deployment_factory, monkeypatch, stage) -> None:
    store = ControlledStore("jobs")
    entered = asyncio.Event()
    read, transition = store.read, store.transition

    async def blocked_read(run_id):
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return await read(run_id)

    async def blocked_heartbeat(run_id, command):
        if isinstance(command, Heartbeat):
            entered.set()
            await asyncio.Event().wait()
        return await transition(run_id, command)

    attribute, blocked = {"read": ("read", blocked_read), "heartbeat": ("transition", blocked_heartbeat)}[stage]
    monkeypatch.setattr(store, attribute, blocked)
    runner = AsyncMock(return_value=Result(value=1))
    deployment = await deployment_factory(runner, store=store, config=RuntimeConfig(shutdown_grace_seconds=0))
    run_id = (await deployment.submit({"value": 1})).control.run_id
    await asyncio.wait_for(entered.wait(), timeout=1)

    await asyncio.wait_for(deployment.close(), timeout=1)
    await asyncio.wait_for(deployment.wait_drained(), timeout=1)

    stored = await read(run_id)
    assert (stored.control.status, stored.control.run_attempt) == (ExecutionStatus.QUEUED, 0)
    reclaimed = await store.claim(SUCCESSOR)
    assert reclaimed is not None and reclaimed.next_control.run_attempt == 1
    runner.assert_not_called()


@pytest.mark.parametrize("threaded", [False, True], ids=["async", "thread"])
@pytest.mark.parametrize("cancel_close", [False, True], ids=["repeated", "cancelled"])
async def test_repeated_close_waits_for_the_claim_release_in_progress(
    deployment_factory, monkeypatch, thread_work, threaded, cancel_close
) -> None:
    started, releasing, proceed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def runner(_context: DurableContext, _request: Request) -> Result:
        started.set()
        await asyncio.Event().wait()
        return Result(value=0)

    deployment = await deployment_factory(
        thread_work[0]("runner") if threaded else runner,
        config=RuntimeConfig(
            poll_interval_seconds=0.005,
            lease_duration_ms=300,
            shutdown_grace_seconds=0.01,
            release_running_on_close=threaded,
        ),
    )
    store = deployment.store
    run_id = (await deployment.submit({"value": 1})).control.run_id
    if threaded:
        assert await asyncio.to_thread(thread_work[1].wait, 1)
    else:
        await asyncio.wait_for(started.wait(), timeout=1)
    transition = store.transition

    async def slow_release(execution_id: str, command):
        if isinstance(command, ReleaseClaim):
            releasing.set()
            await proceed.wait()
        return await transition(execution_id, command)

    monkeypatch.setattr(store, "transition", slow_release)
    first = asyncio.create_task(deployment.close())
    await asyncio.wait_for(releasing.wait(), timeout=1)
    if cancel_close:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    # close() returns at its deadline; the release it started must not be interrupted, and draining waits for it.
    await asyncio.wait_for(deployment.close(), timeout=1)
    drained = asyncio.create_task(deployment.wait_drained())
    await asyncio.sleep(0.05)

    assert not drained.done()
    proceed.set()
    await asyncio.wait_for(drained, timeout=1)
    if not cancel_close:
        await first
    assert (await store.read(run_id)).control.status is ExecutionStatus.QUEUED


@pytest.mark.parametrize("threaded", [False, True], ids=["async", "thread"])
async def test_cancellation_raised_by_the_application_fails_the_execution(deployment_factory, threaded: bool) -> None:
    calls = 0

    def cancel(_context: DurableContext, _request: Request) -> Result:
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError

    async def cancel_async(context: DurableContext, request: Request) -> Result:
        return cancel(context, request)

    deployment = await deployment_factory(cancel if threaded else cancel_async)
    submitted = await deployment.submit({"value": 1})

    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    assert (stored.control.status, stored.control.run_attempt, calls) == (ExecutionStatus.FAILED, 1, 1)


@pytest.mark.parametrize("reraise", [False, True], ids=["suppressed", "cleanup"])
async def test_cancelled_async_work_keeps_ownership_until_it_exits(deployment_factory, reraise) -> None:
    started, cancelled, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def runner(context: DurableContext, request: Request) -> Result:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await finish.wait()
            context.state["cleanup"] = True
            await context.checkpoint()
            if reraise:
                raise
        return Result(value=request.value)

    deployment = await deployment_factory(
        runner, config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300, shutdown_grace_seconds=0)
    )
    run_id = (await deployment.submit({"value": 1})).control.run_id
    await asyncio.wait_for(started.wait(), timeout=1)
    drain = None
    try:
        await asyncio.wait_for(deployment.close(), timeout=1)
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        drain = asyncio.create_task(deployment.wait_drained())
        await asyncio.sleep(0.4)  # Retained work must keep heartbeating beyond the initial lease.
        await asyncio.wait_for(deployment.close(), timeout=1)
        assert not drain.done()
        assert (await deployment.get(run_id)).control.status is ExecutionStatus.RUNNING
        assert await deployment.store.claim(SUCCESSOR) is None
    finally:
        finish.set()
        await asyncio.wait_for(deployment.wait_drained(), timeout=1)
        if drain is not None:
            await drain

    stored = await deployment.store.read(run_id)
    assert decode_json(stored.payloads[PayloadKind.CHECKPOINT], max_bytes=10_000)["application_state"] == {
        "cleanup": True
    }
    expected = ExecutionStatus.QUEUED if reraise else ExecutionStatus.COMPLETED
    assert (stored.control.status, stored.control.run_attempt) == (expected, 0 if reraise else 1)


THREAD_SOURCES = [
    pytest.param("runner", id="runner-thread"),
    pytest.param("adapter", id="adapter-thread", marks=requires_haystack_v3),
]


@pytest.fixture
def thread_work(monkeypatch):
    """A runner whose engine thread blocks, then checkpoints, recording that call's outcome."""
    started, release = threading.Event(), threading.Event()
    outcomes: list[BaseException | None] = []

    def work(context: DurableContext) -> dict[str, bool]:
        started.set()
        release.wait()
        try:
            context.checkpoint_sync()
        except ExecutionLeaseLostError as error:
            outcomes.append(error)
        else:
            outcomes.append(None)
        return {"value": 1}

    def create(source: str):
        if source == "runner":
            return lambda context, _request: work(context)
        from haystack import Pipeline

        from hayhooks.durable.haystack import HaystackDurableAdapter

        adapter = HaystackDurableAdapter(Pipeline())
        monkeypatch.setattr(adapter, "run_pipeline", lambda context, _data, **_options: work(context))

        async def run_nested(context: DurableContext, _request: Request) -> dict[str, bool]:
            return await adapter.run_pipeline_async(context, {})

        return run_nested

    yield create, started, release, outcomes
    release.set()


@pytest.mark.parametrize("source", THREAD_SOURCES)
async def test_retained_thread_work_keeps_its_claim_until_drained(deployment_factory, thread_work, source) -> None:
    create, started, release, outcomes = thread_work
    deployment = await deployment_factory(
        create(source),
        config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300, shutdown_grace_seconds=0.2),
    )
    runtime = DurableRuntime((deployment,))
    run_id = (await deployment.submit({"value": 1})).control.run_id
    assert await asyncio.to_thread(started.wait, 1)
    await runtime.close()

    drain = asyncio.create_task(runtime.wait_drained())
    await asyncio.sleep(0.4)  # longer than the lease: heartbeats must keep the claim alive
    drain.cancel()
    await asyncio.wait({drain})
    # A repeated close() shares the first call's expired grace instead of waiting a new one.
    await asyncio.wait_for(runtime.close(), timeout=0.1)
    assert (await deployment.health())["active_executions"] == 1

    release.set()
    await asyncio.wait_for(runtime.wait_drained(), timeout=1)
    stored = await deployment.get(run_id)
    assert (stored.control.status, outcomes) == (ExecutionStatus.COMPLETED, [None])


@pytest.mark.parametrize("handoff", ["release-on-close", "cancelled-worker"])
@pytest.mark.parametrize("source", THREAD_SOURCES)
async def test_thread_work_is_handed_over_when_its_claim_is_released(
    deployment_factory, thread_work, source, handoff
) -> None:
    create, started, release, outcomes = thread_work
    deployment = await deployment_factory(
        create(source),
        config=RuntimeConfig(
            poll_interval_seconds=0.005,
            lease_duration_ms=300,
            shutdown_grace_seconds=0.01,
            release_running_on_close=handoff == "release-on-close",
        ),
    )
    store = deployment.store
    run_id = (await deployment.submit({"value": 1})).control.run_id
    assert await asyncio.to_thread(started.wait, 1)
    running = (await store.read(run_id)).control

    if handoff == "release-on-close":
        await deployment.close()
        await asyncio.wait_for(deployment.wait_drained(), timeout=1)
    else:
        # A host that cancels the worker stops its heartbeat, so the claim is handed back too.
        await deployment.quiesce()
        [worker] = deployment._workers.values()
        worker.cancel()
        await asyncio.wait({worker})

    reclaimed = await store.claim(SUCCESSOR)
    assert reclaimed is not None and reclaimed.next_control.run_attempt == running.run_attempt
    transitions = AsyncMock(wraps=store.transition)
    store.transition = transitions
    release.set()
    await wait_for_health(deployment, lambda health: health["draining_runs"] == 0)
    assert [type(outcome) for outcome in outcomes] == [ExecutionLeaseLostError]
    transitions.assert_not_called()


@pytest.mark.parametrize("runner_mode", ["sync", "shielded_async"])
async def test_lease_loss_retains_running_work_until_it_exits(deployment_factory, runner_mode: str) -> None:
    started = threading.Event()
    release = threading.Event()
    contexts: list[DurableContext] = []

    def run(context: DurableContext, request: Request) -> Result:
        contexts.append(context)
        started.set()
        release.wait()
        return Result(value=request.value)

    async def run_async(context: DurableContext, request: Request) -> Result:
        task = asyncio.create_task(asyncio.to_thread(run, context, request))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            return await task

    deployment = await deployment_factory(
        run if runner_mode == "sync" else run_async,
        config=RuntimeConfig(poll_interval_seconds=0.005, shutdown_grace_seconds=0.01, lease_duration_ms=300),
    )
    submitted = await deployment.submit({"value": 1})
    assert await asyncio.to_thread(started.wait, 1)
    try:
        contexts[0]._claim.mark_lost()
        await wait_for_health(deployment, lambda health: health["draining_runs"] == 1)
        release.set()
        await wait_for_health(deployment, lambda health: health["draining_runs"] == 0)
        stored = await deployment.get(submitted.control.run_id)
        assert stored.control.status is ExecutionStatus.RUNNING
    finally:
        release.set()


def test_shutdown_grace_bounds_event_loop_teardown_for_sync_runner() -> None:
    source_root = Path(__file__).parents[1] / "src"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import asyncio
                import threading

                from pydantic import BaseModel

                from hayhooks.durable.context import DurableContext
                from hayhooks.durable.runtime import DurableDeployment, RuntimeConfig
                from hayhooks.durable.store import MemoryExecutionStore, StoreConfig


                class Request(BaseModel):
                    value: int


                started = threading.Event()
                blocked = threading.Event()


                def run(_context: DurableContext, request: Request) -> Request:
                    started.set()
                    blocked.wait()
                    return request


                async def main() -> None:
                    deployment = DurableDeployment(
                        "jobs",
                        "v1",
                        MemoryExecutionStore("jobs", config=StoreConfig(lease_commit_safety_ms=10)),
                        Request,
                        run,
                        result_model=Request,
                        config=RuntimeConfig(
                            poll_interval_seconds=0.005,
                            shutdown_grace_seconds=0.01,
                            lease_duration_ms=300,
                        ),
                    )
                    await deployment.start()
                    await deployment.submit({"value": 1})
                    assert await asyncio.to_thread(started.wait, 1)
                    await deployment.close()


                asyncio.run(main())
                """
            ),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(source_root)},
        timeout=3,
    )


def test_core_engine_runs_without_integration_dependencies() -> None:
    source_root = Path(__file__).parents[1] / "src"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import asyncio
                import importlib.abc
                import sys

                BLOCKED = ("fastapi", "haystack", "redis", "hayhooks.server", "hayhooks.settings")


                class BlockIntegrations(importlib.abc.MetaPathFinder):
                    def find_spec(self, name, path=None, target=None):
                        if any(name == blocked or name.startswith(blocked + ".") for blocked in BLOCKED):
                            raise ImportError(f"{name} is blocked")


                sys.meta_path.insert(0, BlockIntegrations())

                import hayhooks.durable._threading
                import hayhooks.durable.context
                import hayhooks.durable.engine
                import hayhooks.durable.models
                import hayhooks.durable.runtime
                import hayhooks.durable.store
                from hayhooks.durable import (
                    DurableContext,
                    DurableDeployment,
                    DurableExecutionCancelledError,
                    DurableRuntime,
                    ExecutionStore,
                    MemoryExecutionStore,
                    RuntimeConfig,
                    StoreConfig,
                    current_durable_context,
                    durable_context_scope,
                    durable_streaming_callback,
                )
                from hayhooks.durable.engine import ExecutionStatus
                from pydantic import BaseModel

                try:
                    from hayhooks.durable import create_durable_router
                except ImportError:
                    pass
                else:
                    raise AssertionError("the FastAPI transport must load only on demand")


                class Request(BaseModel):
                    value: int


                def double(_context: DurableContext, request: Request) -> Request:
                    return Request(value=request.value * 2)


                async def main() -> None:
                    deployment = DurableDeployment(
                        "jobs",
                        "v1",
                        MemoryExecutionStore("jobs", config=StoreConfig(lease_commit_safety_ms=10)),
                        Request,
                        double,
                        result_model=Request,
                        config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300),
                    )
                    runtime = DurableRuntime((deployment,))
                    await runtime.start()
                    run_id = (await deployment.submit({"value": 21})).control.run_id
                    while not (stored := await deployment.get(run_id)).control.terminal:
                        await asyncio.sleep(0.005)
                    await runtime.close()
                    await runtime.wait_drained()
                    assert stored.control.status is ExecutionStatus.COMPLETED


                modules = set(sys.modules)
                dont_write_bytecode = sys.dont_write_bytecode
                asyncio.run(main())
                # Haystack cannot be imported, so its global tracer cannot have been replaced either.
                assert not {name for name in set(sys.modules) - modules if not name.startswith("hayhooks.durable")}
                assert sys.dont_write_bytecode is dont_write_bytecode
                """
            ),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(source_root)},
        timeout=10,
    )


@pytest.mark.parametrize(
    ("source", "poll_interval_seconds"),
    [
        pytest.param("local", 60, id="local-wakes-idle-worker"),
        pytest.param("during-claim", 60, id="local-while-worker-busy"),
        pytest.param("remote", 0.05, id="remote-within-poll"),
    ],
)
async def test_workers_pick_up_submissions(
    deployment_factory, monkeypatch, source: str, poll_interval_seconds: float
) -> None:
    store = ControlledStore("jobs")
    deployment = await deployment_factory(
        store=store,
        config=RuntimeConfig(
            poll_interval_seconds=poll_interval_seconds,
            maintenance_interval_seconds=60,
            lease_duration_ms=300,
        ),
    )
    submissions: list[SubmissionResult] = []
    if source == "during-claim":
        claim = store.claim

        async def claim_then_submit(command):
            plan = await claim(command)
            if plan is None and not submissions:
                submissions.append(await deployment.submit({"value": 1}))
            return plan

        monkeypatch.setattr(store, "claim", claim_then_submit)
    await asyncio.sleep(0.05)
    if source == "local":
        submissions.append(await deployment.submit({"value": 1}))
    elif source == "remote":
        submissions.append(
            await store.submit(
                initial_control(
                    run_id="remote_run",
                    idempotency_digest="remote",
                    idempotency_binding_digest="remote",
                    deployment="jobs",
                    definition_revision="v1",
                    owner_id=None,
                    kind="pipeline",
                    now_ms=0,
                ),
                b'{"value":1}',
            )
        )

    await wait_for_execution(
        deployment,
        submissions[0].control.run_id,
        lambda stored: stored.control.status is ExecutionStatus.COMPLETED,
        timeout=0.5,
    )


@pytest.mark.parametrize("threaded", [False, True], ids=["async", "thread"])
async def test_health_counts_active_executions_until_retained_work_finishes(deployment_factory, threaded: bool) -> None:
    release = threading.Event()

    def blocking(_context: DurableContext, request: BaseModel) -> Result:
        release.wait()
        return Result(value=Request.model_validate(request).value)

    async def blocking_async(context: DurableContext, request: BaseModel) -> Result:
        return await asyncio.to_thread(blocking, context, request)

    deployment = await deployment_factory(
        blocking if threaded else blocking_async,
        config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300, shutdown_grace_seconds=0),
    )
    assert (await deployment.health())["active_executions"] == 0
    await deployment.submit({"value": 1})
    await wait_for_health(deployment, lambda health: health["active_executions"] == 1)
    if threaded:
        await deployment.close()
        assert (await deployment.health())["active_executions"] == 1
    release.set()
    await wait_for_health(deployment, lambda health: health["active_executions"] == 0)


@pytest.mark.parametrize("source", ["retry", "recovered-lease"])
async def test_locally_requeued_work_wakes_idle_workers(deployment_factory, source: str) -> None:
    attempts: list[int] = []

    async def retry_once(context: DurableContext, request: BaseModel) -> Result:
        attempts.append(context.attempt)
        if source == "retry" and len(attempts) == 1:
            await context.retry("again", delay=0.05)
        return Result(value=Request.model_validate(request).value)

    store = MemoryExecutionStore("jobs", config=StoreConfig(lease_commit_safety_ms=10))
    if source == "recovered-lease":
        # A crashed replica's claim, recovered by this process's maintenance.
        await store.submit(
            initial_control(
                run_id="crashed_run",
                idempotency_digest="crashed",
                idempotency_binding_digest="crashed",
                deployment="jobs",
                definition_revision="v1",
                owner_id=None,
                kind="pipeline",
                now_ms=0,
            ),
            b'{"value":1}',
        )
        assert await store.claim(Claim("crashed", 0, 50, 3, "v1", b"{}")) is not None
    deployment = await deployment_factory(
        retry_once,
        store=store,
        config=RuntimeConfig(poll_interval_seconds=60, maintenance_interval_seconds=0.05, lease_duration_ms=300),
    )
    run_id = "crashed_run" if source == "recovered-lease" else (await deployment.submit({"value": 1})).control.run_id

    await wait_for_execution(
        deployment, run_id, lambda stored: stored.control.status is ExecutionStatus.COMPLETED, timeout=0.5
    )
