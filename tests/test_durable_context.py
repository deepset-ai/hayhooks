"""Claimed durable context behavior."""

from __future__ import annotations

import asyncio

import pytest

from hayhooks.durable.context import (
    DurableContext,
    DurableExecutionCancelledError,
    _ClaimedExecution,
    _ExecutionSuspendedError,
    _RetryRequestedError,
    current_durable_context,
    durable_context_scope,
    durable_streaming_callback,
)
from hayhooks.durable.engine import (
    ExecutionLeaseLostError,
    ExecutionStatus,
    PayloadKind,
    ReleaseClaim,
    RequestCancellation,
    Resume,
)
from hayhooks.durable.models import decode_json, encode_json
from hayhooks.durable.store import CHUNK_CURSOR_START
from tests.durable_store_contract import decode_checkpoint


def test_root_exports_durable_streaming_callback() -> None:
    from hayhooks import durable_streaming_callback as public_callback

    assert public_callback is durable_streaming_callback


async def test_checkpoint_commits_progress_once_and_preserves_concurrent_cancellation(context_factory) -> None:
    store, create = context_factory
    context, _ = await create()
    before = await store.read(context.execution_id)
    assert before is not None

    context.state["step"] = 1
    await context.report_progress("checkpointing", metadata={"percent": 50})
    buffered = await store.read(context.execution_id)
    assert buffered is not None and buffered.control.version == before.control.version
    assert not buffered.progress

    await context.checkpoint({"component": "fetch"})
    checkpointed = await store.read(context.execution_id)
    assert checkpointed is not None and checkpointed.control.version == before.control.version + 1
    assert decode_checkpoint(checkpointed.payloads[PayloadKind.CHECKPOINT]).application_state == {"step": 1}
    assert decode_json(checkpointed.progress[0].data, max_bytes=1_024)["message"] == "checkpointing"

    await context.report_progress("finishing")
    await asyncio.gather(
        store.transition(context.execution_id, RequestCancellation(0, "stop")),
        context.checkpoint(),
    )
    canceled = await store.read(context.execution_id)
    assert canceled is not None and canceled.control.cancel_requested_at_ms is not None
    assert [event.sequence for event in canceled.progress] == [1, 2]
    with pytest.raises(DurableExecutionCancelledError):
        await context.check_cancelled()


async def test_progress_buffer_keeps_only_configured_history(context_factory) -> None:
    store, create = context_factory
    context, _ = await create()
    limit = store.config.max_progress_events
    for value in range(limit + 1):
        await context.report_progress(str(value))

    assert len(context._pending_progress) == limit
    await context.checkpoint()
    stored = await store.read(context.execution_id)
    assert stored is not None
    assert [decode_json(event.data, max_bytes=1_024)["message"] for event in stored.progress] == [
        str(value) for value in range(1, limit + 1)
    ]


async def test_suspend_and_resume_persist_one_reconstructable_checkpoint(context_factory) -> None:
    store, create = context_factory
    context, _ = await create()
    context.state["step"] = 1
    await context.report_progress("waiting")

    with pytest.raises(_ExecutionSuspendedError):
        await context.suspend(
            {"kind": "approval", "message": "Continue?"},
            update={"pending": True},
            adapter_checkpoint={"component": "review"},
        )

    waiting = await store.read(context.execution_id)
    assert waiting is not None and waiting.control.status is ExecutionStatus.WAITING
    snapshot = decode_checkpoint(waiting.payloads[PayloadKind.CHECKPOINT])
    assert snapshot.application_state == {"step": 1, "pending": True}
    assert snapshot.adapter_checkpoint == {"component": "review"}
    assert decode_json(waiting.payloads[PayloadKind.WAIT], max_bytes=4_096)["kind"] == "approval"
    assert len(waiting.progress) == 1

    resumed_snapshot = snapshot.model_copy(update={"resume_input": {"approved": True}})
    await store.transition(
        context.execution_id,
        Resume(
            0,
            "v1",
            encode_json(resumed_snapshot.model_dump(mode="json"), max_bytes=4_096),
            expected_version=waiting.control.version,
        ),
    )
    resumed, claim = await create(context.execution_id, submit=False)
    assert resumed.attempt == 2
    assert resumed.resume_input == {"approved": True}
    assert resumed.resume_input is None

    persisted = await store.read(context.execution_id)
    assert persisted is not None
    reconstructed = DurableContext(
        _ClaimedExecution(
            store,
            persisted.control,
            claim.worker_id,
            claim.lease_duration_ms,
            decode_checkpoint(persisted.payloads[PayloadKind.CHECKPOINT]),
        )
    )
    assert reconstructed.resume_input == {"approved": True}

    await resumed.checkpoint()
    persisted = await store.read(context.execution_id)
    assert persisted is not None
    assert decode_checkpoint(persisted.payloads[PayloadKind.CHECKPOINT]).resume_input is None


async def test_sync_bridge_and_callbacks_keep_concurrent_contexts_isolated(context_factory) -> None:
    store, create = context_factory
    first, _ = await create("run_1")
    second, _ = await create("run_2")
    first.state["sync"] = True

    with durable_context_scope(first):
        assert current_durable_context() is first
        await asyncio.to_thread(first.checkpoint_sync)
    assert current_durable_context() is None
    with pytest.raises(RuntimeError, match="runtime event loop"):
        first.checkpoint_sync()

    async def emit(context: DurableContext, value: int) -> None:
        with durable_context_scope(context):
            await asyncio.to_thread(durable_streaming_callback, {"value": value})

    await asyncio.gather(emit(first, 1), emit(second, 2))
    first_chunks = await store.read_chunks(first.execution_id, CHUNK_CURSOR_START)
    second_chunks = await store.read_chunks(second.execution_id, CHUNK_CURSOR_START)
    assert decode_json(first_chunks[0].data, max_bytes=1_024) == {"value": 1}
    assert decode_json(second_chunks[0].data, max_bytes=1_024) == {"value": 2}

    version = first._claim.control.version
    await first.stream_chunk(object())
    assert first._claim.control.version == version
    assert await store.read_chunks(first.execution_id, CHUNK_CURSOR_START) == first_chunks


@pytest.mark.parametrize(
    ("method", "args"),
    [
        pytest.param("checkpoint", (), id="checkpoint"),
        pytest.param("report_progress", ("working",), id="progress"),
        pytest.param("check_cancelled", (), id="cancellation"),
        pytest.param("retry", ("again",), id="retry"),
        pytest.param("suspend", ({"kind": "approval"},), id="suspend"),
        pytest.param("stream_chunk", ({"chunk": 1},), id="chunk"),
    ],
)
async def test_lost_claim_rejects_owned_context_operations(
    context_factory, method: str, args: tuple[object, ...]
) -> None:
    _, create = context_factory
    context, claim = await create()
    claim.mark_lost()
    with pytest.raises(ExecutionLeaseLostError):
        await getattr(context, method)(*args)


async def test_heartbeat_marks_a_rejected_claim_lost(context_factory) -> None:
    store, create = context_factory
    context, claim = await create(lease_duration_ms=60)
    await store.transition(context.execution_id, ReleaseClaim(claim.control.fence, claim.worker_id))
    await asyncio.wait_for(claim.lease_lost.wait(), timeout=0.3)


async def test_missing_execution_marks_claim_lost(context_factory) -> None:
    store, create = context_factory
    context, claim = await create()
    store._controls.pop(context.execution_id)
    with pytest.raises(ExecutionLeaseLostError, match="no longer exists"):
        await context.checkpoint()
    assert claim.lease_lost.is_set()


async def test_stream_chunk_propagates_store_detected_lease_loss(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()

    async def reject_stale_chunk(*_args) -> None:
        raise ExecutionLeaseLostError

    monkeypatch.setattr(store, "append_chunk", reject_stale_chunk)
    with pytest.raises(ExecutionLeaseLostError):
        await context.stream_chunk({"chunk": 1})
    assert claim.lease_lost.is_set()


async def test_retry_request_carries_buffered_progress(context_factory) -> None:
    _, create = context_factory
    context, _ = await create()
    with pytest.raises(ValueError, match="finite non-negative"):
        await context.retry("later", delay=-1)
    await context.report_progress("retrying")
    with pytest.raises(_RetryRequestedError) as raised:
        await context.retry("later", delay=1.5)
    assert (str(raised.value), raised.value.delay, len(raised.value.progress_events)) == ("later", 1.5, 1)
