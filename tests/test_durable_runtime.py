"""Portable durable runtime behavior."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import replace

import pytest
from pydantic import BaseModel, ValidationError

from hayhooks.durable.context import DurableContext
from hayhooks.durable.engine import ExecutionNotFoundError, ExecutionStatus, PayloadKind, initial_control
from hayhooks.durable.models import PersistedError, decode_json
from hayhooks.durable.runtime import DurableDeployment, DurableRuntime, RuntimeConfig
from hayhooks.durable.store import (
    ExecutionIdempotencyConflictError,
    ExecutionStoreError,
    MemoryExecutionStore,
    StoreConfig,
    StoredExecution,
    SubmissionResult,
)


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

    async def initialize(self) -> None:
        self.initialize_calls += 1

    async def submit(self, control, input_payload: bytes) -> SubmissionResult:
        if self.block_submissions:
            self.submission_started.set()
            await self.submission_release.wait()
        return await super().submit(control, input_payload)

    async def claim(self, command):
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
        worker_revision: str,
        revision_error: bytes,
        attempts_error: bytes,
    ) -> int:
        if self.maintenance_error is not None:
            error, self.maintenance_error = self.maintenance_error, None
            self.failure_seen.set()
            raise error
        return await super().maintain(
            max_run_attempts=max_run_attempts,
            worker_revision=worker_revision,
            revision_error=revision_error,
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

    async def create(  # noqa: PLR0913
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


async def test_failed_post_claim_read_releases_without_consuming_attempt(deployment_factory) -> None:
    store = ControlledStore("jobs")
    deployment = await deployment_factory(store=store)
    store.read_error = ExecutionStoreError("unavailable")
    submitted = await deployment.submit({"value": 1})
    await asyncio.wait_for(store.failure_seen.wait(), timeout=1)

    stored = await wait_for_execution(deployment, submitted.control.run_id, lambda value: value.control.terminal)
    assert (stored.control.status, stored.control.run_attempt) == (ExecutionStatus.COMPLETED, 1)


@pytest.mark.parametrize(
    ("revision", "run_attempt", "max_payload_bytes", "code"),
    [
        pytest.param("old", 0, 128, None, id="revision-bounded-fallback"),
        pytest.param("v1", 1, 1_000_000, "run_attempts_exhausted", id="attempt-budget"),
    ],
)
async def test_incompatible_claims_fail_without_running_application(
    deployment_factory,
    revision: str,
    run_attempt: int,
    max_payload_bytes: int,
    code: str | None,
) -> None:
    calls = 0

    async def runner(_context: DurableContext, _request: Request) -> Result:
        nonlocal calls
        calls += 1
        return Result(value=1)

    store = MemoryExecutionStore(
        "jobs",
        config=StoreConfig(lease_commit_safety_ms=10, max_payload_bytes=max_payload_bytes),
    )
    control = replace(
        initial_control(
            run_id="run_1",
            idempotency_digest="idem",
            idempotency_binding_digest="binding",
            deployment="jobs",
            definition_revision=revision,
            owner_id=None,
            kind="pipeline",
            now_ms=0,
        ),
        run_attempt=run_attempt,
    )
    await store.submit(control, b'{"value":1}')
    deployment = await deployment_factory(
        runner,
        store=store,
        config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=300, max_run_attempts=1),
    )

    stored = await wait_for_execution(deployment, control.run_id, lambda value: value.control.terminal)
    error = PersistedError.model_validate(decode_json(stored.payloads[PayloadKind.ERROR], max_bytes=max_payload_bytes))
    assert stored.control.status is ExecutionStatus.FAILED
    assert (calls, error.code) == (0, code)


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
        config=RuntimeConfig(worker_concurrency=2, poll_interval_seconds=0.005, lease_duration_ms=300),
    )
    workers = set(deployment._workers.values())
    if exit_mode == "cancel":
        deployment._workers[0].cancel()
    await wait_for_health(
        deployment,
        lambda health: health["running_slots"] == 2 and bool(set(deployment._workers.values()) - workers),
    )
    assert set(deployment._workers) == {0, 1}


async def test_runtime_instances_are_isolated_and_empty_start_is_inert(deployment_factory) -> None:
    empty = DurableRuntime()
    await empty.start()
    assert await empty.health() == {"healthy": True, "deployments": {}}
    await empty.close()

    first_store, second_store = ControlledStore("first"), ControlledStore("second")
    first = await deployment_factory(store=first_store, start=False)
    second = await deployment_factory(store=second_store, start=False)
    first_runtime, second_runtime = DurableRuntime(), DurableRuntime()
    await first_runtime.install(first)
    await second_runtime.install(second)
    await first_runtime.start()
    await second_runtime.start()
    assert await first_runtime.remove("first") is first
    await first_runtime.close()
    assert not first.accepting and second.accepting
    submitted = await second.submit({"value": 2})
    stored = await wait_for_execution(second, submitted.control.run_id, lambda value: value.control.terminal)
    assert stored.control.status is ExecutionStatus.COMPLETED
    assert (first_store.initialize_calls, second_store.initialize_calls) == (1, 1)
    await second_runtime.close()


@pytest.mark.parametrize("loss_mode", ["shutdown", "lease"])
async def test_thread_work_is_retained_until_exit(deployment_factory, loss_mode: str) -> None:
    started = threading.Event()
    release = threading.Event()
    contexts: list[DurableContext] = []

    def runner(context: DurableContext, request: Request) -> Result:
        contexts.append(context)
        started.set()
        release.wait()
        return Result(value=request.value)

    deployment = await deployment_factory(
        runner,
        config=RuntimeConfig(
            poll_interval_seconds=0.005,
            shutdown_grace_seconds=0.01,
            lease_duration_ms=300,
        ),
    )
    submitted = await deployment.submit({"value": 1})
    assert await asyncio.to_thread(started.wait, 1)
    try:
        if loss_mode == "shutdown":
            await deployment.close()
            assert (await deployment.health())["draining_slots"] == 1
            release.set()
            await wait_for_health(deployment, lambda health: health["draining_slots"] == 0)
            stored = await wait_for_execution(
                deployment, submitted.control.run_id, lambda value: value.control.terminal
            )
            assert stored.control.status is ExecutionStatus.COMPLETED
        else:
            contexts[0]._claim.mark_lost()
            await wait_for_health(deployment, lambda health: health["draining_runs"] == 1)
            release.set()
            await wait_for_health(deployment, lambda health: health["draining_runs"] == 0)
            stored = await deployment.get(submitted.control.run_id)
            assert stored.control.status is ExecutionStatus.RUNNING
    finally:
        release.set()
