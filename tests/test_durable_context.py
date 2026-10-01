"""Claimed durable context behavior."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, Mock

import pytest

from hayhooks.durable._threading import start_daemon_thread
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
    Checkpoint,
    Complete,
    ExecutionLeaseLostError,
    ExecutionStatus,
    Fail,
    Heartbeat,
    PayloadKind,
    ReleaseClaim,
    RequestCancellation,
    Resume,
    ScheduleRetry,
    Suspend,
)
from hayhooks.durable.models import decode_json, encode_json
from hayhooks.durable.store import CHUNK_CURSOR_START, ExecutionStoreCorruptionError, ExecutionStoreError
from tests.durable_store_contract import decode_checkpoint


def test_root_exports_durable_streaming_callback() -> None:
    from hayhooks import durable_streaming_callback as public_callback

    assert public_callback is durable_streaming_callback


@pytest.mark.parametrize(
    "exception_type",
    [SystemExit, KeyboardInterrupt, GeneratorExit, StopIteration, StopAsyncIteration],
)
async def test_daemon_thread_converts_exit_exceptions(exception_type: type[BaseException]) -> None:
    error = exception_type()

    def raise_error() -> None:
        raise error

    result, exited = start_daemon_thread(raise_error, name="test-exit")
    with pytest.raises(RuntimeError, match=f"^durable work raised {exception_type.__name__}$") as raised:
        await asyncio.wait_for(result, 1)
    assert raised.value.__cause__ is error
    await asyncio.wait_for(exited, 1)


async def test_daemon_thread_forwards_control_signals() -> None:
    error = _RetryRequestedError("x", None)

    def raise_error() -> None:
        raise error

    result, exited = start_daemon_thread(raise_error, name="test-signal")
    with pytest.raises(_RetryRequestedError) as raised:
        await result
    assert raised.value is error
    await asyncio.wait_for(exited, 1)


async def test_checkpoint_commits_progress_once_and_preserves_concurrent_cancellation(
    context_factory, monkeypatch
) -> None:
    store, create = context_factory
    context, _ = await create()
    before = await store.read(context.execution_id)
    assert before is not None

    context.state["step"] = 1
    await context.report_progress("checkpointing", metadata={"percent": 50})
    buffered = await store.read(context.execution_id)
    assert buffered is not None and buffered.control.version == before.control.version
    assert not buffered.progress

    transition = store.transition
    unavailable = True

    async def fail_once(run_id, command):
        nonlocal unavailable
        if isinstance(command, Checkpoint) and unavailable:
            unavailable = False
            message = "unavailable"
            raise ExecutionStoreError(message)
        return await transition(run_id, command)

    with monkeypatch.context() as patch:
        patch.setattr(store, "transition", fail_once)
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

    assert len(context._claim.pending_progress) == limit
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
            confirmed_at=time.monotonic(),
        ),
        decode_checkpoint(persisted.payloads[PayloadKind.CHECKPOINT]),
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
    await asyncio.gather(first._claim.flush_chunks(), second._claim.flush_chunks())
    first_chunks = await store.read_chunks(first.execution_id, CHUNK_CURSOR_START)
    second_chunks = await store.read_chunks(second.execution_id, CHUNK_CURSOR_START)
    assert decode_json(first_chunks[0].data, max_bytes=1_024) == {"value": 1}
    assert decode_json(second_chunks[0].data, max_bytes=1_024) == {"value": 2}

    version = first._claim.control.version
    await first.stream_chunk(object())
    await first._claim.flush_chunks()
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


async def test_claim_stops_owning_once_its_window_passes(context_factory, monkeypatch, log_records) -> None:
    store, create = context_factory
    context, claim = await create()
    transition = AsyncMock(wraps=store.transition)
    monkeypatch.setattr(store, "transition", transition)
    claim._confirmed_until = time.monotonic() - 0.001

    assert claim.owned is False
    for operation in (
        context.stream_chunk({"chunk": 1}),
        context.report_progress("late"),
        context.checkpoint(),
    ):
        with pytest.raises(ExecutionLeaseLostError):
            await operation
    with pytest.raises(ExecutionLeaseLostError):
        await claim.transition(Heartbeat(claim.control.fence, claim.worker_id, 0, claim.lease_duration_ms))

    assert claim.lease_lost.is_set()
    transition.assert_not_awaited()
    losses = [record for record in log_records if record["message"].startswith("Durable execution lease lost")]
    assert len(losses) == 1 and "window" in losses[0]["extra"]["reason"]


@pytest.mark.parametrize("command_type", [Heartbeat, Checkpoint], ids=["heartbeat", "checkpoint"])
async def test_hung_store_call_loses_the_lease_within_its_window(context_factory, monkeypatch, command_type) -> None:
    store, create = context_factory
    context, claim = await create(lease_duration_ms=120)
    transition = store.transition

    async def hang(run_id, command):
        if isinstance(command, command_type):
            await asyncio.Event().wait()
        return await transition(run_id, command)

    monkeypatch.setattr(store, "transition", hang)
    if command_type is Heartbeat:
        await asyncio.wait_for(claim.lease_lost.wait(), 0.3)
    else:
        with pytest.raises(ExecutionLeaseLostError):
            await asyncio.wait_for(context.checkpoint(), 1)

    assert claim.lease_lost.is_set()
    assert not claim._transition_lock.locked()
    with pytest.raises(ExecutionLeaseLostError):
        await context.stream_chunk({"late": True})


async def test_heartbeat_drops_progress_the_store_already_holds(context_factory) -> None:
    store, create = context_factory
    context, claim = await create()
    await context.report_progress("one")
    await context.report_progress("two")
    _, events = claim.progress_snapshot()

    await store.transition(
        context.execution_id,
        Checkpoint(
            claim.control.fence,
            claim.worker_id,
            0,
            claim.lease_duration_ms,
            b"{}",
            events,
        ),
    )
    claim._confirmed_until -= 0.5
    await context.check_cancelled()

    assert claim.pending_progress == []
    await context.checkpoint()
    stored = await store.read(context.execution_id)
    assert stored is not None
    assert [event.sequence for event in stored.progress] == [1, 2]


async def test_checkpoint_replay_after_a_lost_reply_stores_progress_once(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()
    transition = store.transition
    reply_lost = False

    async def lose_first_reply(run_id, command):
        nonlocal reply_lost
        plan = await transition(run_id, command)
        if isinstance(command, Checkpoint) and not reply_lost:
            reply_lost = True
            message = "reply lost"
            raise ExecutionStoreError(message)
        return plan

    monkeypatch.setattr(store, "transition", lose_first_reply)
    await context.report_progress("one")
    await context.report_progress("two")
    await context.checkpoint()

    stored = await store.read(context.execution_id)
    assert stored is not None
    assert [event.sequence for event in stored.progress] == [1, 2]
    assert claim.pending_progress == []
    assert claim.control.progress_sequence == 2


async def test_commit_after_an_unconfirmed_checkpoint_does_not_duplicate_progress(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()
    transition = store.transition
    reply_lost = False

    async def corrupt_first_reply(run_id, command):
        nonlocal reply_lost
        plan = await transition(run_id, command)
        if isinstance(command, Checkpoint) and not reply_lost:
            reply_lost = True
            message = "bad reply"
            raise ExecutionStoreCorruptionError(message)
        return plan

    monkeypatch.setattr(store, "transition", corrupt_first_reply)
    await context.report_progress("one")
    await context.report_progress("two")
    with pytest.raises(ExecutionStoreCorruptionError, match="bad reply"):
        await context.checkpoint()

    first, events = claim.progress_snapshot()
    await claim.transition(
        Complete(
            claim.control.fence,
            claim.worker_id,
            0,
            b"null",
            progress_events=events,
            first_progress_sequence=first,
        )
    )
    stored = await store.read(context.execution_id)
    assert stored is not None and stored.control.status is ExecutionStatus.COMPLETED
    assert [event.sequence for event in stored.progress] == [1, 2]


async def test_unconfirmed_finishing_commit_ends_the_claim(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()
    transition = store.transition
    reply_lost = False

    async def lose_first_reply(run_id, command):
        nonlocal reply_lost
        plan = await transition(run_id, command)
        if isinstance(command, Complete) and not reply_lost:
            reply_lost = True
            message = "reply lost"
            raise ExecutionStoreError(message)
        return plan

    monkeypatch.setattr(store, "transition", lose_first_reply)
    with pytest.raises(ExecutionLeaseLostError, match="unconfirmed Complete"):
        await claim.transition(Complete(claim.control.fence, claim.worker_id, 0, b"null"))

    stored = await store.read(context.execution_id)
    assert stored is not None and stored.control.status is ExecutionStatus.COMPLETED
    assert claim.lease_lost.is_set()


async def test_heartbeat_marks_a_rejected_claim_lost(context_factory) -> None:
    store, create = context_factory
    context, claim = await create(lease_duration_ms=60)
    await store.transition(context.execution_id, ReleaseClaim(claim.control.fence, claim.worker_id))
    await asyncio.wait_for(claim.lease_lost.wait(), timeout=0.3)


@pytest.mark.parametrize("state", ["lost", "stopping"])
async def test_thread_start_rejects_lost_or_stopping_claim(context_factory, monkeypatch, state) -> None:
    _, create = context_factory
    context, claim = await create()
    start = Mock()
    monkeypatch.setattr("hayhooks.durable.context.start_daemon_thread", start)
    if state == "lost":
        claim.mark_lost()
    else:
        claim.stopping = True

    error = ExecutionLeaseLostError if state == "lost" else asyncio.CancelledError
    with pytest.raises(error):
        context._start_thread(lambda: None, name="test-rejected-thread")
    start.assert_not_called()


async def test_missing_execution_marks_claim_lost(context_factory) -> None:
    store, create = context_factory
    context, claim = await create()
    store._controls.pop(context.execution_id)
    with pytest.raises(ExecutionLeaseLostError, match="no longer exists"):
        await context.checkpoint()
    assert claim.lease_lost.is_set()


async def test_lease_lost_chunk_flush_surfaces_on_the_next_callback(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()
    monkeypatch.setattr(store, "append_chunks", AsyncMock(side_effect=ExecutionLeaseLostError))

    await context.stream_chunk({"chunk": 1})
    await claim.flush_chunks()

    assert claim.lease_lost.is_set()
    with pytest.raises(ExecutionLeaseLostError):
        await context.stream_chunk({"chunk": 2})


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(lambda fence: Complete(fence, "worker-run_1", 0, b"null"), id="complete"),
        pytest.param(lambda fence: Fail(fence, "worker-run_1", 0, b"{}"), id="fail"),
        pytest.param(lambda fence: Suspend(fence, "worker-run_1", 0, b"{}", b"{}"), id="suspend"),
        pytest.param(lambda fence: ScheduleRetry(fence, "worker-run_1", 0, 0, 1, b"{}"), id="retry"),
        pytest.param(lambda fence: ReleaseClaim(fence, "worker-run_1"), id="release"),
    ],
)
async def test_buffered_chunks_are_flushed_before_leaving_running(context_factory, command) -> None:
    store, create = context_factory
    context, claim = await create()
    await context.stream_chunk({"chunk": 1})
    assert await store.read_chunks(context.execution_id, CHUNK_CURSOR_START) == ()

    await claim.transition(command(claim.control.fence))

    chunks = await store.read_chunks(context.execution_id, CHUNK_CURSOR_START)
    assert decode_json(chunks[0].data, max_bytes=1_024) == {"chunk": 1}


async def test_chunk_buffer_stays_bounded_while_the_store_is_slow(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()
    release = asyncio.Event()
    sent: list[tuple[bytes, ...]] = []

    async def slow_append(*args) -> None:
        sent.append(tuple(args[-1]))
        await release.wait()

    monkeypatch.setattr(store, "append_chunks", slow_append)
    await context.stream_chunk({"chunk": 0})
    flush = asyncio.create_task(claim.flush_chunks())
    await asyncio.sleep(0)
    limit = store.config.max_stream_chunks
    for index in range(1, 3 * limit + 1):
        await context.stream_chunk({"chunk": index})

    assert len(sent) == 1 and len(claim._chunks) == limit
    assert decode_json(claim._chunks[0], max_bytes=1_024) == {"chunk": 2 * limit + 1}
    release.set()
    await flush


async def test_retry_request_keeps_buffered_progress_for_its_commit(context_factory) -> None:
    _, create = context_factory
    context, _ = await create()
    with pytest.raises(ValueError, match="finite non-negative"):
        await context.retry("later", delay=-1)
    await context.report_progress("retrying")
    with pytest.raises(_RetryRequestedError) as raised:
        await context.retry("later", delay=1.5)
    assert (str(raised.value), raised.value.delay) == ("later", 1.5)
    assert context._claim.progress_snapshot() == (1, (context._claim.pending_progress[0],))


async def test_release_rejects_a_write_waiting_behind_it(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()
    transition = store.transition
    commands: list[str] = []
    releasing, proceed = asyncio.Event(), asyncio.Event()

    async def slow_release(run_id: str, command):
        commands.append(type(command).__name__)
        if isinstance(command, ReleaseClaim):
            releasing.set()
            await proceed.wait()
        return await transition(run_id, command)

    monkeypatch.setattr(store, "transition", slow_release)
    release = asyncio.create_task(claim.release())
    await releasing.wait()
    # The checkpoint passes its ownership check, then waits for the transition lock the release holds.
    checkpoint = asyncio.create_task(context.checkpoint({"step": 1}))
    await asyncio.sleep(0.01)
    proceed.set()
    await release

    with pytest.raises(ExecutionLeaseLostError):
        await checkpoint
    assert commands == ["ReleaseClaim"]
    assert (await store.read(context.execution_id)).control.status is ExecutionStatus.QUEUED


async def test_stream_chunk_sync_hands_off_without_waiting_for_the_event_loop(context_factory) -> None:
    store, create = context_factory
    context, claim = await create()
    returned = threading.Event()
    thread = threading.Thread(
        target=lambda: (context.stream_chunk_sync({"chunk": 1}), returned.set()),
        daemon=True,
    )
    thread.start()
    assert returned.wait(timeout=5)
    await claim.transition(Complete(claim.control.fence, claim.worker_id, 0, b"null"))

    chunks = await store.read_chunks(context.execution_id, CHUNK_CURSOR_START)
    assert [chunk.data for chunk in chunks] == [b'{"chunk":1}', b""]


async def test_chunk_queued_before_lease_loss_is_never_sent(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()
    append = AsyncMock()
    monkeypatch.setattr(store, "append_chunks", append)
    returned = threading.Event()
    thread = threading.Thread(
        target=lambda: (context.stream_chunk_sync({"chunk": 1}), returned.set()),
        daemon=True,
    )
    thread.start()
    assert returned.wait(timeout=5)
    claim.mark_lost()
    await claim.flush_chunks()
    append.assert_not_called()
    with pytest.raises(ExecutionLeaseLostError):
        await asyncio.to_thread(context.stream_chunk_sync, {"chunk": 2})


async def test_stream_chunk_sync_on_the_event_loop_buffers_directly(context_factory) -> None:
    _, create = context_factory
    context, claim = await create()
    context.stream_chunk_sync({"chunk": 1})
    assert list(claim._chunks) == [b'{"chunk":1}']


async def test_idle_claim_does_not_wake_the_chunk_flusher(context_factory, monkeypatch) -> None:
    store, create = context_factory
    monkeypatch.setattr("hayhooks.durable.context._CHUNK_FLUSH_SECONDS", 0.001)
    context, claim = await create()
    flush, flushed = claim.flush_chunks, asyncio.Event()
    buffered: list[int] = []

    async def counted() -> None:
        buffered.append(len(claim._chunks))
        await flush()
        flushed.set()

    monkeypatch.setattr(claim, "flush_chunks", counted)
    await asyncio.sleep(0.05)
    assert buffered == []

    await context.stream_chunk({"chunk": 1})
    await asyncio.wait_for(flushed.wait(), timeout=1)
    assert buffered == [1]
    assert [chunk.data for chunk in await store.read_chunks(context.execution_id, CHUNK_CURSOR_START)] == [
        b'{"chunk":1}'
    ]


async def test_chunk_wake_is_bounded_while_the_event_loop_is_stalled(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()
    schedule = claim.event_loop.call_soon_threadsafe
    wake_callbacks = 0

    def counted_schedule(callback, *args, context=None):
        nonlocal wake_callbacks
        wake_callbacks += callback == claim._chunks_buffered.set
        return schedule(callback, *args, context=context)

    monkeypatch.setattr(claim.event_loop, "call_soon_threadsafe", counted_schedule)
    producers = 8
    chunks_per_producer = 1_250
    barrier = threading.Barrier(producers + 1)
    errors: list[BaseException] = []

    def produce(producer: int) -> None:
        try:
            barrier.wait()
            for index in range(chunks_per_producer):
                context.stream_chunk_sync({"producer": producer, "index": index})
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=produce, args=(producer,), daemon=True) for producer in range(producers)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert wake_callbacks == 1
    assert len(claim._chunks) == store.config.max_stream_chunks

    flush = claim.flush_chunks
    first_flushed = asyncio.Event()
    injected = False

    async def inject_during_clear_before_drain() -> None:
        nonlocal injected
        if not injected:
            injected = True
            assert not claim._chunk_wake_scheduled
            returned = threading.Event()
            thread = threading.Thread(
                target=lambda: (context.stream_chunk_sync({"chunk": "race"}), returned.set()),
                daemon=True,
            )
            thread.start()
            assert returned.wait(timeout=5)
        await flush()
        first_flushed.set()

    monkeypatch.setattr(claim, "flush_chunks", inject_during_clear_before_drain)
    await asyncio.wait_for(first_flushed.wait(), timeout=1)
    assert wake_callbacks == 2

    context.stream_chunk_sync({"chunk": "terminal"})
    await claim.transition(Complete(claim.control.fence, claim.worker_id, 0, b"null"))
    chunks, cursor = (), CHUNK_CURSOR_START
    while page := await store.read_chunks(context.execution_id, cursor):
        chunks += page
        cursor = page[-1].cursor
    assert [chunk.data for chunk in chunks[-3:]] == [b'{"chunk":"race"}', b'{"chunk":"terminal"}', b""]


async def test_first_heartbeat_is_due_one_interval_after_the_claim(context_factory, monkeypatch) -> None:
    store, create = context_factory
    _, claim = await create()
    await claim.stop()
    beat = asyncio.Event()
    transition = store.transition

    async def recorded(run_id, command):
        if isinstance(command, Heartbeat):
            beat.set()
        return await transition(run_id, command)

    monkeypatch.setattr(store, "transition", recorded)
    slow = _ClaimedExecution(
        store,
        claim.control,
        claim.worker_id,
        claim.lease_duration_ms,
        confirmed_at=time.monotonic() - 10,
    )
    await slow.start()
    try:
        await asyncio.wait_for(beat.wait(), timeout=1)
    finally:
        await slow.stop()


async def test_check_cancelled_reuses_a_recently_confirmed_control(context_factory, monkeypatch) -> None:
    store, create = context_factory
    context, claim = await create()
    transition = store.transition
    heartbeats = 0

    async def counted(run_id, command):
        nonlocal heartbeats
        heartbeats += isinstance(command, Heartbeat)
        return await transition(run_id, command)

    monkeypatch.setattr(store, "transition", counted)
    await store.transition(context.execution_id, RequestCancellation(0, "stop"))
    await context.check_cancelled()
    assert heartbeats == 0

    claim._confirmed_until -= 0.5
    with pytest.raises(DurableExecutionCancelledError):
        await context.check_cancelled()
    assert heartbeats == 1
