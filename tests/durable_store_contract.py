"""Shared observable contract for durable store implementations."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import patch

import pytest

from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionCommand,
    ExecutionStatus,
    Heartbeat,
    InvalidExecutionTransitionError,
    PayloadKind,
    ReleaseClaim,
    RequestCancellation,
    Resume,
    Suspend,
    TransitionPlan,
    initial_control,
)
from hayhooks.durable.models import CheckpointEnvelope, decode_json
from hayhooks.durable.store import (
    CHUNK_CURSOR_START,
    ChunkCursorExpiredError,
    ExecutionIdempotencyConflictError,
    ExecutionStore,
    StoreConfig,
)

CONTRACT_CONFIG = StoreConfig(
    lease_commit_safety_ms=10,
    max_payload_bytes=64,
    max_progress_events=2,
    max_progress_event_bytes=32,
    max_stream_chunks=3,
    max_stream_chunk_bytes=64,
)
ATTEMPTS_ERROR = b"attempts"


def decode_checkpoint(payload: bytes) -> CheckpointEnvelope:
    return CheckpointEnvelope.model_validate(decode_json(payload, max_bytes=4_096))


def contract_control(
    deployment: str,
    run_id: str = "run_1",
    *,
    idempotency: str = "idem",
    binding: str = "binding",
    kind: str = "pipeline",
):
    return initial_control(
        run_id=run_id,
        idempotency_digest=idempotency,
        idempotency_binding_digest=binding,
        deployment=deployment,
        definition_revision="v1",
        owner_id="owner",
        kind=kind,
        now_ms=0,
    )


async def assert_store_contract(store: ExecutionStore) -> None:  # noqa: PLR0915
    """Exercise public store behavior without backend-specific access."""
    control = contract_control(store.deployment)
    with pytest.raises(InvalidExecutionTransitionError):
        await store.submit(replace(control, version=2), b"input")

    submitted = await store.submit(control, b"input")
    replayed = await store.submit(contract_control(store.deployment, "ignored"), b"input")
    assert submitted.created and not replayed.created
    assert replayed.control.run_id == control.run_id
    with pytest.raises(ExecutionIdempotencyConflictError):
        await store.submit(contract_control(store.deployment, "conflict", binding="other"), b"input")
    with pytest.raises(ExecutionIdempotencyConflictError, match="run ID"):
        await store.submit(contract_control(store.deployment, idempotency="other", binding="other"), b"changed")

    snapshot = await store.read(control.run_id)
    assert snapshot is not None and snapshot.payloads[PayloadKind.INPUT] == b"input"
    assert await store.read_public(control.run_id) == replace(snapshot, payloads={})
    assert await store.operational_counts(revision="v1") == {
        "nonterminal": 1,
        "revision_nonterminal": 1,
        "revision_runnable": 1,
        "lease_expiry": 0,
    }

    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None and claimed.next_control.status is ExecutionStatus.RUNNING
    released = await store.transition(control.run_id, ReleaseClaim(claimed.next_control.fence, "worker"))
    assert (released.next_control.run_attempt, released.next_control.lease_recoveries) == (1, 0)
    assert await store.operational_counts(revision="v1") == {
        "nonterminal": 1,
        "revision_nonterminal": 1,
        "revision_runnable": 1,
        "lease_expiry": 0,
    }

    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    before_heartbeat = await store.read(control.run_id)
    heartbeat = await store.transition(
        control.run_id,
        Heartbeat(claimed.next_control.fence, "worker", 0, 500),
    )
    after_heartbeat = await store.read(control.run_id)
    assert before_heartbeat is not None and after_heartbeat is not None
    assert heartbeat.next_control == after_heartbeat.control
    assert after_heartbeat == replace(
        before_heartbeat,
        control=replace(before_heartbeat.control, lease_expires_at_ms=heartbeat.next_control.lease_expires_at_ms),
    )
    assert await store.operational_counts(revision="v1") == {
        "nonterminal": 1,
        "revision_nonterminal": 1,
        "revision_runnable": 0,
        "lease_expiry": 1,
    }

    before_chunks = await store.read(control.run_id)
    await store.append_chunks(
        control.run_id,
        claimed.next_control.run_attempt,
        claimed.next_control.fence,
        "worker",
        [str(index).encode() for index in range(4)],
    )
    after_chunks = await store.read(control.run_id)
    assert before_chunks is not None and after_chunks is not None
    assert after_chunks.control.version == before_chunks.control.version
    chunks = await store.read_chunks(control.run_id, CHUNK_CURSOR_START)
    assert [chunk.data for chunk in chunks] == [b"1", b"2", b"3"]
    assert await store.read_chunks(control.run_id, chunks[0].cursor) == chunks[1:]
    assert await store.wait_chunks(control.run_id, chunks[0].cursor, 5) == chunks[1:]
    assert await store.wait_chunks(control.run_id, chunks[-1].cursor, 0.01) == ()
    for read in (store.read_chunks, lambda run_id, cursor: store.wait_chunks(run_id, cursor, 5)):
        with pytest.raises(ChunkCursorExpiredError):
            await read(control.run_id, "0-1")

    await store.transition(
        control.run_id,
        Checkpoint(claimed.next_control.fence, "worker", 0, 500, b"checkpoint", (b"one", b"two", b"three")),
    )
    snapshot = await store.read(control.run_id)
    assert snapshot is not None
    assert snapshot.payloads[PayloadKind.CHECKPOINT] == b"checkpoint"
    assert [event.sequence for event in snapshot.progress] == [2, 3]
    replayed = await store.transition(
        control.run_id,
        Checkpoint(
            claimed.next_control.fence,
            "worker",
            0,
            500,
            b"checkpoint",
            (b"two", b"three", b"four"),
            first_progress_sequence=2,
        ),
    )
    assert [(event.sequence, event.data) for event in replayed.progress_events] == [(4, b"four")]
    snapshot = await store.read(control.run_id)
    assert snapshot is not None and [(event.sequence, event.data) for event in snapshot.progress] == [
        (3, b"three"),
        (4, b"four"),
    ]

    suspended = await store.transition(
        control.run_id,
        Suspend(claimed.next_control.fence, "worker", 0, b"checkpoint", b"wait"),
    )
    assert suspended.next_control.status is ExecutionStatus.WAITING
    public = await store.read_public(control.run_id)
    assert public is not None and public.payloads == {PayloadKind.WAIT: b"wait"}
    resumed = await store.transition(control.run_id, Resume(0, "v1", b"resumed"))
    assert resumed.next_control.status is ExecutionStatus.QUEUED
    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    await store.transition(control.run_id, RequestCancellation(0, "stop"))
    terminal = await store.transition(
        control.run_id,
        Complete(claimed.next_control.fence, "worker", 0, b"x" * (store.config.max_payload_bytes + 1)),
    )
    assert terminal.next_control.status is ExecutionStatus.CANCELED
    snapshot = await store.read(control.run_id)
    assert snapshot is not None
    assert not ({PayloadKind.RESULT, PayloadKind.ERROR, PayloadKind.WAIT} & snapshot.payloads.keys())
    assert await store.operational_counts(revision="v1") == {
        "nonterminal": 0,
        "revision_nonterminal": 0,
        "revision_runnable": 0,
        "lease_expiry": 0,
    }


async def assert_revision_routing_contract(store: ExecutionStore) -> None:
    """Workers claim only executions for the revision they can run."""
    old = contract_control(store.deployment, "run_a_old", idempotency="old", binding="old")
    new = replace(
        contract_control(store.deployment, "run_b_new", idempotency="new", binding="new"),
        definition_revision="v2",
    )
    await store.submit(old, b"input")
    await store.submit(new, b"input")
    expected = {"nonterminal": 2, "revision_nonterminal": 1, "revision_runnable": 1, "lease_expiry": 0}
    assert await store.operational_counts(revision="v1") == expected
    assert await store.operational_counts(revision="v2") == expected

    new_claim = await store.claim(Claim("worker-v2", 0, 500, 3, "v2", ATTEMPTS_ERROR))
    assert new_claim is not None
    assert (new_claim.next_control.run_id, new_claim.next_control.status) == ("run_b_new", ExecutionStatus.RUNNING)
    assert await store.operational_counts(revision="v2") == {
        "nonterminal": 2,
        "revision_nonterminal": 1,
        "revision_runnable": 0,
        "lease_expiry": 1,
    }
    old_snapshot = await store.read(old.run_id)
    assert old_snapshot is not None and old_snapshot.control.status is ExecutionStatus.QUEUED

    old_claim = await store.claim(Claim("worker-v1", 0, 500, 3, "v1", ATTEMPTS_ERROR))
    assert old_claim is not None
    assert (old_claim.next_control.run_id, old_claim.next_control.status) == ("run_a_old", ExecutionStatus.RUNNING)


async def assert_terminal_markers_contract(store: ExecutionStore) -> None:
    """Every terminal path appends one marker, even when chunk persistence is disabled."""
    for index in range(4):
        run_id = f"run_{index}"
        await store.submit(contract_control(store.deployment, run_id, idempotency=run_id, binding=run_id), b"input")

    await store.transition("run_0", RequestCancellation(0, "queued"))
    claimed = [await store.claim(Claim("worker", 0, 50, 3, "v1", ATTEMPTS_ERROR)) for _ in range(3)]
    fences = {plan.next_control.run_id: plan.next_control.fence for plan in claimed if plan is not None}
    await store.transition("run_1", Suspend(fences["run_1"], "worker", 0, b"checkpoint", b"wait"))
    await store.transition("run_1", RequestCancellation(0, "waiting"))
    await store.transition("run_2", Complete(fences["run_2"], "worker", 0, b"done"))
    await asyncio.sleep(0.06)
    await store.maintain(max_run_attempts=1, attempts_error=ATTEMPTS_ERROR)

    for run_id, status in (
        ("run_0", ExecutionStatus.CANCELED),
        ("run_1", ExecutionStatus.CANCELED),
        ("run_2", ExecutionStatus.COMPLETED),
        ("run_3", ExecutionStatus.FAILED),
    ):
        stored = await store.read(run_id)
        chunks = await store.read_chunks(run_id, CHUNK_CURSOR_START)
        assert stored is not None and stored.control.status is status
        assert [(chunk.terminal, chunk.attempt) for chunk in chunks] == [(True, stored.control.run_attempt)]


async def assert_raced_recovery_contract(store: ExecutionStore) -> None:
    """A lease recovery that loses its race skips that entry and still recovers the rest of the batch."""
    for run_id in ("run_a", "run_b"):
        await store.submit(contract_control(store.deployment, run_id, idempotency=run_id, binding=run_id), b"input")
        assert await store.claim(Claim("worker", 0, 50, 3, "v1", ATTEMPTS_ERROR)) is not None
    await asyncio.sleep(0.06)
    transition = store.transition

    async def raced(run_id: str, command: ExecutionCommand) -> TransitionPlan:
        if run_id == "run_a":
            raise InvalidExecutionTransitionError("lease renewed after the index scan")
        return await transition(run_id, command)

    with patch.object(store, "transition", raced):
        await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR)

    statuses = [(await store.read(run_id)) for run_id in ("run_a", "run_b")]
    assert [stored.control.status for stored in statuses if stored is not None] == [
        ExecutionStatus.RUNNING,
        ExecutionStatus.QUEUED,
    ]


async def assert_lost_lease_budget_contract(store: ExecutionStore) -> None:
    """Each recovered lease is counted; the run fails when the last allowed lease is lost."""
    await store.submit(contract_control(store.deployment), b"input")
    for lost in range(1, 4):
        assert await store.claim(Claim("worker", 0, 50, 3, "v1", ATTEMPTS_ERROR)) is not None
        await asyncio.sleep(0.06)
        await store.maintain(max_run_attempts=3, attempts_error=ATTEMPTS_ERROR)
        stored = await store.read("run_1")
        assert stored is not None
        assert (stored.control.run_attempt, stored.control.lease_recoveries) == (lost, lost)
    assert stored.control.status is ExecutionStatus.FAILED
    assert stored.payloads[PayloadKind.ERROR] == ATTEMPTS_ERROR
