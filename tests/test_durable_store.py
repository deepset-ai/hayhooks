"""Contract checks for durable store implementations."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionPayloadSizeError,
    ExecutionStatus,
    InvalidExecutionTransitionError,
    PayloadKind,
    RequestCancellation,
    Resume,
    Suspend,
    initial_control,
)
from hayhooks.durable.store import (
    CHUNK_CURSOR_START,
    ChunkCursorExpiredError,
    ExecutionIdempotencyConflictError,
    MemoryExecutionStore,
    StoreConfig,
)


class Clock:
    now = 1_000

    def __call__(self) -> int:
        return self.now


def control(run_id: str = "run_1", *, idempotency: str = "idem", binding: str = "binding"):
    return initial_control(
        run_id=run_id,
        idempotency_digest=idempotency,
        idempotency_binding_digest=binding,
        deployment="jobs",
        definition_revision="v1",
        owner_id="owner",
        kind="pipeline",
        now_ms=0,
    )


async def test_memory_store_contract() -> None:
    clock = Clock()
    store = MemoryExecutionStore(
        "jobs",
        clock=clock,
        config=StoreConfig(lease_commit_safety_ms=10, max_progress_events=2, max_stream_chunks=3),
    )
    await store.initialize()

    with pytest.raises(InvalidExecutionTransitionError):
        await store.submit(replace(control(), version=2), b"input")
    submitted = await store.submit(control(), b"input")
    replayed = await store.submit(control("ignored"), b"input")
    assert submitted.created and not replayed.created
    assert replayed.control.run_id == "run_1"
    with pytest.raises(ExecutionIdempotencyConflictError):
        await store.submit(control("conflict", binding="other"), b"input")
    with pytest.raises(ExecutionIdempotencyConflictError, match="run ID"):
        await store.submit(control(idempotency="other", binding="other"), b"overwritten")
    snapshot = await store.read("run_1")
    assert snapshot is not None and snapshot.payloads[PayloadKind.INPUT] == b"input"
    assert await store.operational_counts() == {"nonterminal": 1, "runnable": 1, "lease_expiry": 0}

    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1"))
    assert claimed is not None and claimed.next_control.status is ExecutionStatus.RUNNING
    run_id = claimed.next_control.run_id
    await store.transition(
        run_id,
        Checkpoint(1, "worker", 0, 500, b"checkpoint", (b"one", b"two", b"three")),
    )
    snapshot = await store.read(run_id)
    assert snapshot is not None
    assert snapshot.payloads[PayloadKind.INPUT] == b"input"
    assert snapshot.payloads[PayloadKind.CHECKPOINT] == b"checkpoint"
    assert [event.sequence for event in snapshot.progress] == [2, 3]

    suspended = await store.transition(run_id, Suspend(1, "worker", 0, b"checkpoint", b"wait"))
    assert suspended.next_control.status is ExecutionStatus.WAITING
    resumed = await store.transition(run_id, Resume(0, "v1", b"resumed"))
    assert resumed.next_control.status is ExecutionStatus.QUEUED
    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1"))
    assert claimed is not None
    await store.transition(run_id, RequestCancellation(0, "stop"))
    terminal = await store.transition(run_id, Complete(claimed.next_control.fence, "worker", 0, b"ignored"))
    assert terminal.next_control.status is ExecutionStatus.CANCELED
    assert await store.operational_counts() == {"nonterminal": 0, "runnable": 0, "lease_expiry": 0}


async def test_chunks_are_bounded_and_outside_the_reducer() -> None:
    store = MemoryExecutionStore("jobs", clock=Clock(), config=StoreConfig(max_stream_chunks=3))
    await store.submit(control(), b"input")
    before = await store.read("run_1")
    for index in range(4):
        await store.append_chunk("run_1", 1, str(index).encode())
    after = await store.read("run_1")
    assert before is not None and after is not None
    assert after.control.version == before.control.version

    chunks = await store.read_chunks("run_1", CHUNK_CURSOR_START)
    assert [chunk.data for chunk in chunks] == [b"1", b"2", b"3"]
    assert await store.read_chunks("run_1", chunks[0].cursor) == chunks[1:]
    with pytest.raises(ChunkCursorExpiredError):
        await store.read_chunks("run_1", "0-1")


async def test_store_enforces_size_recovery_and_ttl() -> None:
    clock = Clock()
    store = MemoryExecutionStore(
        "jobs",
        clock=clock,
        config=StoreConfig(lease_commit_safety_ms=10, max_payload_bytes=8, terminal_ttl_seconds=1),
    )
    await store.submit(control(), b"input")
    claimed = await store.claim(Claim("worker", 0, 100, 2, "v1"))
    assert claimed is not None
    with pytest.raises(ExecutionPayloadSizeError):
        await store.transition("run_1", Checkpoint(1, "worker", 0, 100, b"too-large"))

    clock.now += 100
    assert await store.maintain(max_run_attempts=2, worker_revision="v1") == 1
    reclaimed = await store.claim(Claim("worker", 0, 100, 2, "v1"))
    assert reclaimed is not None
    await store.transition("run_1", Complete(2, "worker", 0, b"done"))
    clock.now += 1_000
    assert await store.read("run_1") is None


def test_importing_durable_does_not_load_the_server() -> None:
    subprocess.run(  # noqa: S603 - fixed interpreter and script
        [
            sys.executable,
            "-c",
            "import sys; import hayhooks.durable; "
            "assert not any(name == 'hayhooks.server' or name.startswith('hayhooks.server.') for name in sys.modules)",
        ],
        check=True,
    )
