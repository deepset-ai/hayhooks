"""Application-contract tests for the durable store adapter."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hayhooks.durable.backend import ExecutionStoreConfig
from hayhooks.durable.context import RESUME_INPUT_KEY
from hayhooks.durable.engine import Claim, Heartbeat, RequestCancellation, Resume
from hayhooks.durable.manager import DurableExecutionManager
from hayhooks.durable.models import (
    ExecutionCheckpoint,
    ExecutionKind,
    ExecutionProgressEvent,
    ExecutionRecord,
    ExecutionRecordSizeError,
    ExecutionStatus,
    ExecutionStoreError,
    RetryableExecutionError,
)
from hayhooks.durable.reference import InMemoryExecutionStore
from hayhooks.durable.runtime import DurableRuntime
from hayhooks.durable.store import ExecutionStore, InMemoryExecutionStoreProvider, RedisExecutionStoreProvider
from hayhooks.settings import AppSettings, settings


def _config() -> ExecutionStoreConfig:
    return ExecutionStoreConfig(
        max_input_bytes=512,
        max_checkpoint_bytes=512,
        max_result_bytes=512,
        max_error_bytes=512,
        max_wait_bytes=512,
        max_progress_events=2,
        max_progress_event_bytes=256,
    )


def _store(*, config: ExecutionStoreConfig | None = None, **options) -> ExecutionStore:
    return ExecutionStore(
        InMemoryExecutionStore(deployment="deployment", config=config or _config()),
        definition_revision="rev-1",
        **options,
    )


def _record() -> ExecutionRecord:
    return ExecutionRecord(
        execution_id="run_1",
        execution_kind=ExecutionKind.PIPELINE,
        deployment_name="deployment",
        definition_revision="rev-1",
        validated_input={"question": "hello"},
        operation_fingerprint="request-fingerprint",
        owner_id="owner",
        max_progress_events=2,
        max_record_bytes=512,
    )


def test_builtin_providers_snapshot_explicit_durable_settings() -> None:
    app_settings = AppSettings(
        durable_redis_key_prefix="portable:durable",
        durable_lease_duration_ms=45_000,
        durable_lease_commit_safety_ms=2_000,
        durable_terminal_ttl_seconds=123,
        durable_max_nonterminal_executions=12,
        durable_max_attempts=7,
        durable_max_progress_events=17,
        durable_max_record_bytes=32_768,
    )
    memory_store = InMemoryExecutionStoreProvider(app_settings=app_settings).create_execution_store("portable")
    redis_provider = RedisExecutionStoreProvider(
        redis=AsyncMock(),
        app_settings=app_settings,
        socket_timeout=1.5,
        socket_connect_timeout=2.5,
        health_check_interval=0,
    )
    redis_store = redis_provider.create_execution_store("portable")

    for store in (memory_store, redis_store):
        assert store.lease_duration_ms == 45_000
        assert store.max_run_attempts == 7
        assert store.max_progress_events == 17
        assert store.max_record_bytes == 32_768
        assert store.config.key_prefix == "portable:durable"
        assert store.config.lease_commit_safety_ms == 2_000
        assert store.config.terminal_ttl_seconds == 123
        assert store.config.max_nonterminal_executions == 12

    assert redis_provider.socket_timeout == 1.5
    assert redis_provider.socket_connect_timeout == 2.5
    assert redis_provider.health_check_interval == 0


def test_runtime_uses_its_explicit_settings_for_the_default_provider() -> None:
    app_settings = AppSettings(durable_store="memory", durable_lease_duration_ms=45_000)
    runtime = DurableRuntime(app_settings=app_settings)

    provider = runtime._provider()

    assert isinstance(provider, InMemoryExecutionStoreProvider)
    assert provider.app_settings.durable_lease_duration_ms == 45_000


async def test_runtime_uses_implicit_provider_settings_until_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    original_attempts = settings.durable_max_attempts
    runtime = DurableRuntime()
    provider = runtime._provider()

    monkeypatch.setattr(settings, "durable_max_attempts", original_attempts + 1)

    assert runtime.app_settings.durable_max_attempts == provider.app_settings.durable_max_attempts == original_attempts
    await runtime.close()
    assert runtime.app_settings.durable_max_attempts == original_attempts + 1


def test_runtime_uses_supplied_builtin_provider_as_its_settings_source() -> None:
    app_settings = AppSettings(durable_store="memory", durable_lease_duration_ms=45_000, durable_max_attempts=7)
    provider = InMemoryExecutionStoreProvider(app_settings=app_settings)
    runtime = DurableRuntime(provider)

    store = provider.create_execution_store("portable")

    assert runtime.app_settings.durable_lease_duration_ms == store.lease_duration_ms == 45_000
    assert runtime.app_settings.durable_max_attempts == store.max_run_attempts == 7


def test_runtime_rejects_conflicting_builtin_provider_settings() -> None:
    provider = InMemoryExecutionStoreProvider(app_settings=AppSettings(durable_lease_duration_ms=45_000))

    with pytest.raises(ValueError, match="settings must match"):
        DurableRuntime(provider, app_settings=AppSettings(durable_lease_duration_ms=60_000))


async def test_runtime_provider_cannot_be_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = InMemoryExecutionStoreProvider()
    close = AsyncMock()
    monkeypatch.setattr(provider, "close", close)
    runtime = DurableRuntime(provider)

    with pytest.raises(AttributeError):
        setattr(runtime, "provider", InMemoryExecutionStoreProvider())

    assert runtime.provider is provider
    await runtime.close()
    close.assert_awaited_once()


async def test_store_preserves_public_checkpoint_progress_wait_resume_and_result_contract() -> None:
    store = _store(
        lease_duration_ms=10_000,
        max_run_attempts=3,
        max_progress_events=2,
        max_record_bytes=512,
    )
    await store.initialize()

    created, submitted = await store.submit_with_record(_record())
    replayed, replay = await store.submit_with_record(_record())
    assert created and not replayed
    assert replay.execution_id == submitted.execution_id

    claim = await store.claim_next("worker")
    assert claim is not None
    async with claim:
        claim.record.application_state["step"] = "checkpointed"
        claim.record.checkpoint = ExecutionCheckpoint(ExecutionKind.PIPELINE, {"component": "search"})
        claim.record.append_progress("checkpoint saved", kind="checkpoint")
        await claim.checkpoint()
        claim.record.wait = {"kind": "approval", "message": "continue?"}
        claim.record.status = ExecutionStatus.WAITING
        claim.record.append_progress("waiting", kind="waiting")
        await claim.suspend()

    waiting = await store.get("run_1")
    assert waiting is not None
    assert waiting.status is ExecutionStatus.WAITING
    assert waiting.application_state == {"step": "checkpointed"}
    assert [event.kind for event in waiting.progress] == ["checkpoint", "waiting"]

    assert await store.resume("run_1", {"approved": True})
    resumed = await store.claim_next("worker")
    assert resumed is not None
    async with resumed:
        assert resumed.record.application_state.pop(RESUME_INPUT_KEY) == {"approved": True}
        resumed.record.result = {"answer": "done"}
        resumed.record.status = ExecutionStatus.COMPLETED
        await resumed.complete()

    completed = await store.get("run_1")
    assert completed is not None
    assert completed.status is ExecutionStatus.COMPLETED
    assert completed.result == {"answer": "done"}
    assert completed.wait is None
    assert [event.kind for event in completed.progress] == ["waiting", "resumed"]


async def test_cancel_and_checkpoint_assign_distinct_persisted_progress_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = InMemoryExecutionStore(deployment="deployment", config=_config())
    store = ExecutionStore(core, definition_revision="rev-1")
    await store.submit(_record())
    entered = asyncio.Event()
    release = asyncio.Event()
    original_transition = core.transition

    async def gated_transition(run_id, command, *, candidate=False):
        if isinstance(command, RequestCancellation):
            entered.set()
            await release.wait()
        return await original_transition(run_id, command, candidate=candidate)

    monkeypatch.setattr(core, "transition", gated_transition)
    claim = await store.claim_next("worker")
    assert claim is not None
    async with claim:
        cancellation = asyncio.create_task(store.request_cancel("run_1"))
        await entered.wait()
        claim.record.append_progress("checkpoint", kind="checkpoint")
        await claim.checkpoint()
        release.set()
        assert await cancellation

    record = await store.get("run_1")
    assert record is not None
    assert [event.sequence for event in record.progress] == [1, 2]


async def test_checkpoint_keeps_progress_added_after_a_concurrent_cancellation() -> None:
    store = _store()
    await store.submit(_record())
    claim = await store.claim_next("worker")
    assert claim is not None

    async with claim:
        claim.record.append_progress("before cancellation")
        assert await store.request_cancel("run_1")
        await claim.checkpoint()
        claim.record.append_progress("after cancellation")
        await claim.checkpoint()

    record = await store.get("run_1")
    assert record is not None
    assert [(event.sequence, event.message) for event in record.progress] == [
        (2, "before cancellation"),
        (3, "after cancellation"),
    ]


async def test_losing_resume_race_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    core = InMemoryExecutionStore(deployment="deployment", config=_config())
    store = ExecutionStore(core, definition_revision="rev-1")
    await store.submit(_record())
    claim = await store.claim_next("worker")
    assert claim is not None
    async with claim:
        claim.record.status = ExecutionStatus.WAITING
        claim.record.wait = {"kind": "approval"}
        await claim.suspend()

    entered = asyncio.Event()
    release = asyncio.Event()
    original_transition = core.transition

    async def gated_transition(run_id, command, *, candidate=False):
        if isinstance(command, Resume):
            entered.set()
            await release.wait()
        return await original_transition(run_id, command, candidate=candidate)

    monkeypatch.setattr(core, "transition", gated_transition)
    resumed = asyncio.create_task(store.resume("run_1", {"approved": True}))
    await entered.wait()
    await store.request_cancel("run_1")
    release.set()

    assert not await resumed
    record = await store.get("run_1")
    assert record is not None and record.status is ExecutionStatus.CANCELED


async def test_retry_exhaustion_persists_its_progress_event() -> None:
    store = _store(max_run_attempts=1)

    async def runner(context):
        await context.retry("again", delay=0)

    manager = DurableExecutionManager(
        "deployment", store, runner, adapter=object(), poll_interval=0.001, max_attempts=1
    )
    await manager.start()
    try:
        await store.submit(_record())
        for _ in range(100):
            record = await store.get("run_1")
            if record is not None and record.terminal:
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("retry exhaustion did not become terminal")
    finally:
        await manager.close()

    assert record is not None
    assert [event.kind for event in record.progress] == ["retry_exhausted"]


async def test_store_runs_through_the_existing_durable_manager_contract() -> None:
    store = _store(
        lease_duration_ms=10_000,
        max_run_attempts=3,
        max_progress_events=2,
        max_record_bytes=512,
    )

    async def runner(context):
        context.record.application_state["phase"] = "running"
        await context.report_progress("started")
        return {"answer": "done"}

    manager = DurableExecutionManager("deployment", store, runner, adapter=object(), poll_interval=0.001)
    await manager.start()
    try:
        await store.submit(_record())
        for _ in range(100):
            record = await store.get("run_1")
            if record and record.terminal:
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("manager did not complete the submitted execution")
        assert record.status is ExecutionStatus.COMPLETED
        assert record.result == {"answer": "done"}
        assert record.application_state == {"phase": "running"}
        assert [event.message for event in record.progress] == ["started"]
    finally:
        await manager.close()


async def test_manager_health_reports_worker_store_failures() -> None:
    store = SimpleNamespace(
        initialize=AsyncMock(),
        claim_next=AsyncMock(side_effect=ExecutionStoreError("claim failed")),
        maintain=AsyncMock(),
        operational_counts=AsyncMock(return_value={"nonterminal": 1, "runnable": 1, "lease_expiry": 0}),
    )
    manager = DurableExecutionManager(
        "deployment", store, AsyncMock(), adapter=object(), poll_interval=0.001, shutdown_grace_period=0.01
    )
    await manager.start()
    try:
        for _ in range(100):
            if store.claim_next.await_count:
                break
            await asyncio.sleep(0.001)
        health = await manager.health_snapshot()
    finally:
        await manager.close()

    assert not health["healthy"]
    assert health["worker_store_error_streak"] >= 1


async def test_canceled_runner_restarts_its_worker_slot() -> None:
    store = _store(
        config=replace(_config(), lease_commit_safety_ms=1),
        lease_duration_ms=50,
    )
    calls = 0

    async def runner(_context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError
        return {"answer": "done"}

    manager = DurableExecutionManager("deployment", store, runner, adapter=object(), poll_interval=0.001)
    await manager.start()
    try:
        await store.submit(_record())
        for _ in range(200):
            record = await store.get("run_1")
            if record is not None and record.terminal:
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("canceled runner did not recover")
    finally:
        await manager.close()

    assert calls == 2
    assert record is not None
    assert record.status is ExecutionStatus.COMPLETED


async def test_oversized_result_fails_without_replaying_the_runner() -> None:
    store = _store(max_record_bytes=512)
    calls = 0

    async def runner(_context):
        nonlocal calls
        calls += 1
        return {"answer": "x" * 512}

    manager = DurableExecutionManager("deployment", store, runner, adapter=object())
    await store.initialize()
    assert await store.submit(_record())
    claim = await store.claim_next("worker")
    assert claim is not None

    await manager._process_claim(claim)

    record = await store.get("run_1")
    assert calls == 1
    assert record is not None
    assert record.status is ExecutionStatus.FAILED
    assert record.error is not None
    assert record.error.code == "record_too_large"


def test_progress_event_enforces_the_full_serialized_size() -> None:
    with pytest.raises(ExecutionRecordSizeError, match="progress event"):
        ExecutionProgressEvent(sequence=1, message="", kind="", metadata={"data": "x" * 8_150})


async def test_in_memory_store_uses_real_time_for_delayed_retries() -> None:
    store = _store(
        config=replace(_config(), lease_commit_safety_ms=1),
        lease_duration_ms=10_000,
        max_run_attempts=3,
        max_progress_events=2,
        max_record_bytes=512,
    )
    attempts = 0

    async def runner(_context):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            msg = "try again"
            raise RetryableExecutionError(msg, delay=0.01)
        return {"answer": "done"}

    manager = DurableExecutionManager("deployment", store, runner, adapter=object(), poll_interval=0.001)
    await manager.start()
    try:
        await store.submit(_record())
        for _ in range(100):
            record = await store.get("run_1")
            if record is not None and record.terminal:
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("in-memory durable retry did not become due")
    finally:
        await manager.close()

    assert attempts == 2
    assert record.status is ExecutionStatus.COMPLETED


def test_store_rejects_a_lease_safety_margin_equal_to_the_lease() -> None:
    with pytest.raises(ValueError, match="lease_commit_safety_ms"):
        _store(
            config=replace(_config(), lease_commit_safety_ms=100),
            lease_duration_ms=100,
        )


def test_store_requires_the_safe_lease_window_to_cover_one_heartbeat() -> None:
    with pytest.raises(ValueError, match="heartbeat interval"):
        _store(
            config=replace(_config(), lease_commit_safety_ms=0),
            lease_duration_ms=1,
        )


async def test_local_lease_deadlines_exclude_backend_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    core = InMemoryExecutionStore(deployment="deployment", config=_config())
    store = ExecutionStore(core, definition_revision="rev-1", lease_duration_ms=10_000)
    await store.submit(_record())
    original_transition = core.transition
    started: dict[type, float] = {}

    async def delayed_transition(run_id, command, *, candidate=False):
        if isinstance(command, Claim | Heartbeat):
            started[type(command)] = time.monotonic()
            await asyncio.sleep(0.05)
        return await original_transition(run_id, command, candidate=candidate)

    monkeypatch.setattr(core, "transition", delayed_transition)
    claim = await store.claim_next("worker")
    assert claim is not None
    assert claim._confirmed_until == pytest.approx(started[Claim] + store.lease_safe_duration, abs=0.02)

    async with claim:
        assert claim._confirmed_until == pytest.approx(started[Heartbeat] + store.lease_safe_duration, abs=0.02)


async def test_store_normalizes_redis_client_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from redis.exceptions import ConnectionError as RedisConnectionError

    store = _store()

    async def unavailable(_execution_id: str):
        msg = "offline"
        raise RedisConnectionError(msg)

    monkeypatch.setattr(store.core, "get", unavailable)
    with pytest.raises(ExecutionStoreError, match="read execution"):
        await store.get("run_1")


async def test_store_retries_a_view_changed_while_reading_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    core = InMemoryExecutionStore(deployment="deployment", config=_config())
    store = ExecutionStore(core, definition_revision="rev-1")
    await store.initialize()
    await store.submit(_record())
    original_read_progress = core.read_progress
    changed = False

    async def read_progress_and_cancel(run_id: str) -> list[bytes]:
        nonlocal changed
        progress = await original_read_progress(run_id)
        if not changed:
            changed = True
            await core.transition(run_id, RequestCancellation(0, "cancel"))
        return progress

    monkeypatch.setattr(core, "read_progress", read_progress_and_cancel)
    record = await store.get("run_1")

    assert changed
    assert record is not None
    assert record.status is ExecutionStatus.CANCELED


async def test_concurrent_adapter_claims_return_only_the_fenced_owner() -> None:
    store = _store()
    await store.submit(_record())
    claims = await asyncio.gather(*(store.claim_next(f"worker-{index}") for index in range(3)))
    owners = [claim for claim in claims if claim is not None]
    assert len(owners) == 1
    assert owners[0].control.lease_owner == owners[0].worker_id
