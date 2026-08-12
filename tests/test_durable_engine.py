"""Reducer lifecycle invariants for the lean durable engine."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    Complete,
    ExecutionLeaseLostError,
    ExecutionStatus,
    Fail,
    Heartbeat,
    InvalidExecutionTransitionError,
    PayloadKind,
    RecoverExpiredLease,
    RequestCancellation,
    Resume,
    ScheduleRetry,
    Suspend,
    decide,
    initial_control,
    submission_plan,
)


def control(**changes: object):
    defaults = {
        "run_id": "run-1",
        "idempotency_digest": "idem",
        "idempotency_binding_digest": "binding",
        "deployment": "deployment",
        "definition_revision": "rev-1",
        "owner_id": "owner",
        "kind": "pipeline",
        "now_ms": 100,
    }
    defaults.update(changes)
    return initial_control(**defaults)


def claim(current, *, now_ms: int = 200):
    return decide(current, Claim("worker-a", now_ms, 500, 3, "rev-1"))


def test_submission_is_an_initial_control_and_input_only() -> None:
    plan = submission_plan(control(), b"input")
    assert plan.next_control.version == 1
    assert plan.payload_writes[0].kind is PayloadKind.INPUT
    assert plan.lease_index_update is None


def test_claim_fences_and_heartbeat_renews_without_business_version() -> None:
    claimed = claim(control()).next_control
    assert (claimed.status, claimed.version, claimed.fence, claimed.run_attempt) == (
        ExecutionStatus.RUNNING,
        2,
        1,
        1,
    )
    heartbeat = decide(claimed, Heartbeat(1, "worker-a", 300, 500)).next_control
    assert heartbeat.version == claimed.version
    assert heartbeat.lease_expires_at_ms == 800
    with pytest.raises(ExecutionLeaseLostError):
        decide(heartbeat, Heartbeat(0, "worker-a", 301, 500))
    with pytest.raises(ExecutionLeaseLostError):
        decide(heartbeat, Heartbeat(1, "worker-a", 750, 500, 50))


@pytest.mark.parametrize(
    "command",
    [
        lambda c: Checkpoint(c.fence, "stale", 300, 500, b"checkpoint"),
        lambda c: Complete(c.fence + 1, "worker-a", 300, b"result"),
        lambda c: ScheduleRetry(c.fence + 1, "worker-a", 300, 100, 2, b"error"),
        lambda c: Suspend(c.fence + 1, "worker-a", 300, b"checkpoint", b"wait"),
    ],
)
def test_owned_transitions_reject_stale_fences(command) -> None:
    with pytest.raises(ExecutionLeaseLostError):
        decide(claim(control()).next_control, command(claim(control()).next_control))


def test_cancellation_wins_completion_and_retry() -> None:
    claimed = claim(control()).next_control
    canceled = decide(claimed, RequestCancellation(250, "💥" * 2_000)).next_control
    assert canceled.cancel_reason == "💥" * 1_024
    assert len(canceled.cancel_reason.encode()) == 4_096
    terminal = decide(canceled, Complete(1, "worker-a", 300, b"result"))
    assert terminal.next_control.status is ExecutionStatus.CANCELED
    assert not terminal.payload_writes
    retried = decide(canceled, ScheduleRetry(1, "worker-a", 300, 100, 2, b"error"))
    assert retried.next_control.status is ExecutionStatus.CANCELED

def test_retry_and_lease_recovery_requeue_without_resetting_retry_count() -> None:
    claimed = claim(control()).next_control
    retry = decide(claimed, ScheduleRetry(1, "worker-a", 300, 100, 2, b"retry"))
    queued = retry.next_control
    assert (queued.status, queued.available_at_ms, queued.run_attempt, queued.application_retry_count) == (
        ExecutionStatus.QUEUED,
        400,
        1,
        1,
    )
    next_claim = claim(replace(queued, available_at_ms=None), now_ms=400).next_control
    recovered = decide(
        next_claim,
        RecoverExpiredLease(1_000, next_claim.fence, next_claim.lease_expires_at_ms or 0, 3, "rev-1"),
    )
    assert recovered.next_control.status is ExecutionStatus.QUEUED
    assert recovered.next_control.application_retry_count == 1
    assert recovered.next_control.run_attempt == 2


def test_wait_resume_and_progress_preserve_checkpoint_boundary() -> None:
    claimed = claim(control()).next_control
    waiting = decide(
        claimed,
        Suspend(1, "worker-a", 300, b"checkpoint", b"wait", progress_events=(b"waiting",)),
    )
    assert waiting.next_control.status is ExecutionStatus.WAITING
    assert waiting.lease_index_update and waiting.lease_index_update.deadline_ms is None
    resumed = decide(waiting.next_control, Resume(400, "rev-1", b"checkpoint", (b"resumed",)))
    assert resumed.next_control.status is ExecutionStatus.QUEUED
    assert [event.sequence for event in (*waiting.progress_events, *resumed.progress_events)] == [1, 2]


def test_terminal_state_is_irreversible_and_payloads_are_exclusive() -> None:
    completed = decide(claim(control()).next_control, Complete(1, "worker-a", 300, b"result"))
    assert completed.payload_deletes == (PayloadKind.ERROR, PayloadKind.WAIT)
    assert decide(completed.next_control, RequestCancellation(400, "late")).next_control == completed.next_control
    with pytest.raises(InvalidExecutionTransitionError):
        decide(completed.next_control, Claim("worker-a", 400, 500, 3, "rev-1"))
    failed = decide(claim(control()).next_control, Fail(1, "worker-a", 300, b"error"))
    assert failed.payload_deletes == (PayloadKind.RESULT, PayloadKind.WAIT)


def test_revision_mismatch_never_grants_a_fence() -> None:
    plan = decide(control(), Claim("worker-a", 200, 500, 3, "rev-2"))
    assert plan.next_control.status is ExecutionStatus.FAILED
    assert plan.next_control.fence == 0


def test_stale_lease_index_is_removed_without_changing_control() -> None:
    current = control()
    plan = decide(current, RecoverExpiredLease(200, 1, 100, 3, "rev-1"))
    assert plan.next_control == current
    assert plan.lease_index_update and plan.lease_index_update.deadline_ms is None
