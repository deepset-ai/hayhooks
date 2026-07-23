import asyncio

import pytest

from hayhooks.execution import (
    DurableExecutionManager,
    ExecutionCheckpoint,
    ExecutionError,
    ExecutionKind,
    ExecutionRecord,
    ExecutionStatus,
    ExecutionStoreError,
    InMemoryExecutionStore,
)


def _record(execution_id: str = "execution", *, revision: str = "revision") -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=execution_id,
        execution_kind=ExecutionKind.PIPELINE,
        deployment_name="test",
        definition_revision=revision,
        validated_input={"value": 1},
    )


def test_execution_errors_redact_structured_and_transport_secrets() -> None:
    raw = (
        "request failed: Authorization: Bearer header-token, "
        '"password": "json secret", api_key=plain-secret, '
        "https://service.test/path?access_token=query-secret&safe=value"
    )

    error = ExecutionError.from_exception(RuntimeError(raw))

    assert "header-token" not in error.message
    assert "json secret" not in error.message
    assert "plain-secret" not in error.message
    assert "query-secret" not in error.message
    assert error.message.count("<redacted>") == 4
    assert "safe=value" in error.message


def test_record_serialization_trims_old_progress_to_fit_total_size_limit() -> None:
    record = _record()
    record.max_record_bytes = 1_024
    for index in range(20):
        record.append_progress(f"event-{index}-" + "x" * 100)

    payload = record.to_json()

    assert len(payload.encode("utf-8")) <= record.max_record_bytes
    assert 0 < len(record.progress) < 20
    assert record.progress[-1].message.startswith("event-19-")


@pytest.mark.asyncio
async def test_memory_store_waiting_work_requires_explicit_resume() -> None:
    store = InMemoryExecutionStore()
    await store.initialize()
    assert await store.submit(_record())

    claim = await store.claim_next("worker")
    assert claim is not None
    async with claim:
        claim.record.status = ExecutionStatus.WAITING
        claim.record.wait = {"reason": "input"}
        await claim.suspend()

    assert (await store.get("execution")).status is ExecutionStatus.WAITING
    assert await store.claim_next("other-worker") is None
    assert await store.resume("execution", {"approved": True})

    resumed = await store.claim_next("other-worker")
    assert resumed is not None
    from hayhooks.execution import DurableContext

    context = DurableContext(resumed, adapter=object())
    assert context.resume_input == {"approved": True}
    assert context.take_resume_input() == {"approved": True}
    assert context.resume_input is None
    async with resumed:
        resumed.record.status = ExecutionStatus.COMPLETED
        await resumed.complete()


@pytest.mark.asyncio
async def test_revision_retirement_terminalizes_unclaimed_work_and_rechecks_after_drain() -> None:
    store = InMemoryExecutionStore()
    await store.initialize()
    assert await store.submit(_record("waiting", revision="old"))
    waiting = await store.claim_next("worker")
    assert waiting is not None
    async with waiting:
        waiting.record.status = ExecutionStatus.WAITING
        waiting.record.wait = {"kind": "input"}
        await waiting.suspend()

    assert await store.submit(_record("running", revision="old"))
    running = await store.claim_next("worker")
    assert running is not None
    async with running:
        assert await store.submit(_record("queued", revision="old"))
        assert await store.submit(_record("cancel-requested", revision="old"))
        assert await store.request_cancel("cancel-requested")
        assert await store.submit(_record("current", revision="current"))

        assert await store.retire_incompatible("current") == 3
        for execution_id in ("waiting", "queued"):
            retired = await store.get(execution_id)
            assert retired is not None
            assert retired.status is ExecutionStatus.FAILED
            assert retired.error is not None
            assert retired.error.code == "definition_revision_conflict"
        canceled = await store.get("cancel-requested")
        assert canceled is not None
        assert canceled.status is ExecutionStatus.CANCELED
        assert (await store.get("running")).status is ExecutionStatus.RUNNING
        assert (await store.get("current")).status is ExecutionStatus.QUEUED

    assert (await store.get("running")).status is ExecutionStatus.QUEUED
    assert await store.retire_incompatible("current") == 1
    assert (await store.get("running")).status is ExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_prepared_manager_without_live_worker_slots_is_unhealthy() -> None:
    manager = DurableExecutionManager(
        "test",
        InMemoryExecutionStore(),
        lambda _context: asyncio.sleep(0),
        adapter=object(),
    )

    assert manager.health["healthy"]
    await manager.prepare()
    assert not manager.health["healthy"]
    manager.activate()
    assert manager.health["healthy"]
    await manager.close()
    assert not manager.health["healthy"]


@pytest.mark.asyncio
async def test_manager_persists_safe_terminal_result() -> None:
    store = InMemoryExecutionStore()

    async def runner(_context):
        return {"done": True}

    manager = DurableExecutionManager("test", store, runner, adapter=object(), poll_interval=0.001)
    await manager.start()
    try:
        assert await manager.submit(_record())
        for _ in range(100):
            result = await store.get("execution")
            if result is not None and result.terminal:
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("durable manager did not complete execution")
    finally:
        await manager.close()

    assert result.status is ExecutionStatus.COMPLETED
    assert result.result == {"done": True}
    assert "validated_input" not in result.safe_view()
    assert manager.health["active_claims"] == 0
    assert manager.health["metrics"]["attempts_started"] == 1
    assert manager.health["metrics"]["completed"] == 1


@pytest.mark.asyncio
async def test_manager_terminalizes_a_checkpoint_that_exceeds_the_total_record_limit() -> None:
    store = InMemoryExecutionStore()

    async def runner(context):
        await context.checkpoint(
            ExecutionCheckpoint(
                ExecutionKind.PIPELINE,
                {"snapshot": "x" * 2_000},
            )
        )
        return {"unreachable": True}

    manager = DurableExecutionManager("test", store, runner, adapter=object(), poll_interval=0.001)
    record = _record()
    record.max_record_bytes = 1_024
    await manager.start()
    try:
        assert await manager.submit(record)
        for _ in range(100):
            result = await store.get(record.execution_id)
            if result is not None and result.terminal:
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("oversized checkpoint did not become terminal")
    finally:
        await manager.close()

    assert result.status is ExecutionStatus.FAILED
    assert result.checkpoint is None
    assert result.error is not None
    assert result.error.code == "record_too_large"


@pytest.mark.asyncio
async def test_manager_retries_transient_claim_errors_without_losing_worker() -> None:
    class TransientStore(InMemoryExecutionStore):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 1

        async def claim_next(self, worker_name):
            if self.failures:
                self.failures -= 1
                msg = "temporary store outage"
                raise ConnectionError(msg)
            return await super().claim_next(worker_name)

    store = TransientStore()

    async def runner(_context):
        return {"recovered": True}

    manager = DurableExecutionManager("test", store, runner, adapter=object(), poll_interval=0.001)
    await manager.start()
    try:
        assert await manager.submit(_record())
        for _ in range(200):
            result = await store.get("execution")
            if result is not None and result.terminal:
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("durable worker did not recover after a transient store error")
        assert not manager._workers[0].done()
    finally:
        await manager.close()

    assert result.status is ExecutionStatus.COMPLETED
    assert result.result == {"recovered": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["checkpoint", "complete", "retry", "suspend", "release"])
async def test_manager_store_transition_failure_does_not_kill_worker(operation) -> None:
    class FailingClaim:
        def __init__(self, delegate):
            self.delegate = delegate
            self.record = delegate.record
            self.failed = False

        async def __aenter__(self):
            await self.delegate.__aenter__()
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            if operation == "release" and not self.failed:
                self.failed = True
                await self.delegate.__aexit__(exc_type, exc, traceback)
                raise ExecutionStoreError("temporary release failure")
            return await self.delegate.__aexit__(exc_type, exc, traceback)

        async def cancellation_requested(self):
            return await self.delegate.cancellation_requested()

        async def checkpoint(self):
            if operation == "checkpoint" and not self.failed:
                self.failed = True
                raise ExecutionStoreError("temporary checkpoint failure")
            return await self.delegate.checkpoint()

        async def complete(self):
            if operation == "complete" and not self.failed:
                self.failed = True
                raise ExecutionStoreError("temporary complete failure")
            return await self.delegate.complete()

        async def retry(self, error, *, delay):
            if operation == "retry" and not self.failed:
                self.failed = True
                raise ExecutionStoreError("temporary retry failure")
            return await self.delegate.retry(error, delay=delay)

        async def suspend(self):
            if operation == "suspend" and not self.failed:
                self.failed = True
                raise ExecutionStoreError("temporary suspend failure")
            return await self.delegate.suspend()

    class TransientStore(InMemoryExecutionStore):
        def __init__(self):
            super().__init__()
            self.inject = True

        async def claim_next(self, worker_name):
            claim = await super().claim_next(worker_name)
            if claim is not None and self.inject:
                self.inject = False
                return FailingClaim(claim)
            return claim

    store = TransientStore()

    async def runner(context):
        if operation == "checkpoint":
            await context.checkpoint()
        if operation == "retry" and context.attempt == 1:
            from hayhooks.execution import RetryableExecutionError

            raise RetryableExecutionError("again")
        if operation == "suspend" and context.attempt == 1:
            context.record.status = ExecutionStatus.WAITING
            await context.claim.suspend()
        return {"done": True}

    manager = DurableExecutionManager(
        "test",
        store,
        runner,
        adapter=object(),
        poll_interval=0.001,
        retry_base_delay=0,
    )
    await manager.start()
    try:
        assert await manager.submit(_record())
        for _ in range(500):
            record = await store.get("execution")
            if record is not None and (
                record.terminal or (operation == "suspend" and record.status is ExecutionStatus.WAITING)
            ):
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail(f"worker did not recover from {operation} failure")
        assert not manager._workers[0].done()
        assert manager.health["store_error_count"] >= 1
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_retry_exhaustion_is_terminal_and_success_clears_retry_error() -> None:
    from hayhooks.execution import RetryableExecutionError

    store = InMemoryExecutionStore()

    async def runner(context):
        if context.record.execution_id == "exhaust":
            raise RetryableExecutionError("still unavailable")
        if context.attempt == 1:
            raise RetryableExecutionError("temporary")
        return {"ok": True}

    manager = DurableExecutionManager(
        "test",
        store,
        runner,
        adapter=object(),
        poll_interval=0.001,
        max_attempts=2,
        retry_base_delay=0,
    )
    await manager.start()
    try:
        assert await manager.submit(_record("exhaust"))
        assert await manager.submit(_record("success"))
        for _ in range(500):
            exhausted = await store.get("exhaust")
            success = await store.get("success")
            if exhausted and exhausted.terminal and success and success.terminal:
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("retry records did not become terminal")
    finally:
        await manager.close()

    assert exhausted.status is ExecutionStatus.FAILED
    assert exhausted.error == ExecutionError(
        type="RetryExhausted",
        message="Execution exhausted its 2 permitted attempts",
        retryable=False,
        code="retry_exhausted",
    )
    assert success.status is ExecutionStatus.COMPLETED
    assert success.error is None


@pytest.mark.asyncio
async def test_cancellation_wins_terminal_write_race() -> None:
    store = InMemoryExecutionStore()
    await store.initialize()
    assert await store.submit(_record())
    claim = await store.claim_next("worker")
    assert claim is not None

    async with claim:
        claim.record.status = ExecutionStatus.COMPLETED
        assert await store.request_cancel("execution")
        await claim.complete()

    assert (await store.get("execution")).status is ExecutionStatus.CANCELED


def test_legacy_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="Legacy durable execution records"):
        ExecutionRecord.from_dict({"execution_id": "prototype"})


def test_record_deserialization_reapplies_store_progress_limit() -> None:
    record = _record()
    record.append_progress("first")
    record.append_progress("second")
    record.append_progress("third")

    restored = ExecutionRecord.from_json(record.to_json(), max_progress_events=2)

    assert [event.message for event in restored.progress] == ["second", "third"]
    assert restored.max_progress_events == 2
