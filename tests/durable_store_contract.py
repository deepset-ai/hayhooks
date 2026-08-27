"""Shared observable contract for durable store implementations."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionStatus,
    Heartbeat,
    InvalidExecutionTransitionError,
    PayloadKind,
    ReleaseClaim,
    RequestCancellation,
    Resume,
    Suspend,
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
    assert await store.operational_counts() == {"nonterminal": 1, "runnable": 1, "lease_expiry": 0}

    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None and claimed.next_control.status is ExecutionStatus.RUNNING
    released = await store.transition(control.run_id, ReleaseClaim(claimed.next_control.fence, "worker"))
    assert released.next_control.run_attempt == 0
    assert await store.operational_counts() == {"nonterminal": 1, "runnable": 1, "lease_expiry": 0}

    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    heartbeat = await store.transition(
        control.run_id,
        Heartbeat(claimed.next_control.fence, "worker", 0, 500),
    )
    assert heartbeat.next_control.version == claimed.next_control.version
    assert await store.operational_counts() == {"nonterminal": 1, "runnable": 0, "lease_expiry": 1}

    before_chunks = await store.read(control.run_id)
    for index in range(4):
        await store.append_chunk(
            control.run_id,
            claimed.next_control.run_attempt,
            claimed.next_control.fence,
            "worker",
            str(index).encode(),
        )
    after_chunks = await store.read(control.run_id)
    assert before_chunks is not None and after_chunks is not None
    assert after_chunks.control.version == before_chunks.control.version
    chunks = await store.read_chunks(control.run_id, CHUNK_CURSOR_START)
    assert [chunk.data for chunk in chunks] == [b"1", b"2", b"3"]
    assert await store.read_chunks(control.run_id, chunks[0].cursor) == chunks[1:]
    with pytest.raises(ChunkCursorExpiredError):
        await store.read_chunks(control.run_id, "0-1")

    await store.transition(
        control.run_id,
        Checkpoint(claimed.next_control.fence, "worker", 0, 500, b"checkpoint", (b"one", b"two", b"three")),
    )
    snapshot = await store.read(control.run_id)
    assert snapshot is not None
    assert snapshot.payloads[PayloadKind.CHECKPOINT] == b"checkpoint"
    assert [event.sequence for event in snapshot.progress] == [2, 3]

    suspended = await store.transition(
        control.run_id,
        Suspend(claimed.next_control.fence, "worker", 0, b"checkpoint", b"wait"),
    )
    assert suspended.next_control.status is ExecutionStatus.WAITING
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
    assert await store.operational_counts() == {"nonterminal": 0, "runnable": 0, "lease_expiry": 0}


async def assert_revision_routing_contract(store: ExecutionStore) -> None:
    """Workers claim only executions for the revision they can run."""
    old = contract_control(store.deployment, "run_a_old", idempotency="old", binding="old")
    new = replace(
        contract_control(store.deployment, "run_b_new", idempotency="new", binding="new"),
        definition_revision="v2",
    )
    await store.submit(old, b"input")
    await store.submit(new, b"input")

    new_claim = await store.claim(Claim("worker-v2", 0, 500, 3, "v2", ATTEMPTS_ERROR))
    assert new_claim is not None
    assert (new_claim.next_control.run_id, new_claim.next_control.status) == ("run_b_new", ExecutionStatus.RUNNING)
    old_snapshot = await store.read(old.run_id)
    assert old_snapshot is not None and old_snapshot.control.status is ExecutionStatus.QUEUED

    old_claim = await store.claim(Claim("worker-v1", 0, 500, 3, "v1", ATTEMPTS_ERROR))
    assert old_claim is not None
    assert (old_claim.next_control.run_id, old_claim.next_control.status) == ("run_a_old", ExecutionStatus.RUNNING)
