"""Shared observable contract for the memory and Redis durable stores."""

from __future__ import annotations

from typing import Any

import pytest

from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionStatus,
    RequestCancellation,
    Resume,
    Suspend,
    initial_control,
)
from hayhooks.durable.redis import ExecutionIdempotencyConflictError


def control(run_id: str = "run_1", *, idempotency_digest: str = "a" * 64, binding_digest: str = "b" * 64):
    return initial_control(
        run_id=run_id,
        idempotency_digest=idempotency_digest,
        idempotency_binding_digest=binding_digest,
        deployment="integration",
        definition_revision="rev-1",
        owner_id="owner",
        kind="pipeline",
        now_ms=0,
    )


async def assert_store_contract(store: Any) -> None:
    """Exercise public store behavior without inspecting backend internals."""
    accepted = await store.submit(control(), b"{}", binding_digest="b" * 64)
    replay = await store.submit(control("run_2"), b"{}", binding_digest="b" * 64)
    assert accepted.created and not replay.created
    assert replay.control.run_id == "run_1"
    with pytest.raises(ExecutionIdempotencyConflictError):
        await store.submit(control("run_3", binding_digest="e" * 64), b"{}", binding_digest="e" * 64)

    run_id = await store.read_candidate()
    assert run_id is not None
    claimed = await store.transition(
        run_id,
        Claim("worker", 0, 1_000, 3, "rev-1"),
        candidate=True,
    )
    assert claimed.next_control.status is ExecutionStatus.RUNNING
    checkpointed = await store.transition(run_id, Checkpoint(1, "worker", 0, 1_000, b"checkpoint"))
    suspended = await store.transition(
        run_id,
        Suspend(checkpointed.next_control.fence, "worker", 0, b"checkpoint-2", b"wait"),
    )
    assert suspended.next_control.status is ExecutionStatus.WAITING
    resumed = await store.transition(run_id, Resume(0, "rev-1", b"checkpoint-3"))
    assert resumed.next_control.status is ExecutionStatus.QUEUED

    candidate = await store.read_candidate()
    assert candidate == run_id
    claimed_again = await store.transition(
        run_id,
        Claim("worker", 0, 1_000, 3, "rev-1"),
        candidate=True,
    )
    await store.transition(run_id, RequestCancellation(0, "stop"))
    terminal = await store.transition(
        run_id,
        Complete(claimed_again.next_control.fence, "worker", 0, b"ignored-result"),
    )
    assert terminal.next_control.status is ExecutionStatus.CANCELED
