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
    ReleaseClaim,
    RequestCancellation,
    Resume,
    ScheduleRetry,
    Suspend,
    decide,
    initial_control,
    submission_plan,
)

REVISION_ERROR = b"revision"
ATTEMPTS_ERROR = b"attempts"


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
    return decide(current, Claim("worker-a", now_ms, 500, 3, "rev-1", REVISION_ERROR, ATTEMPTS_ERROR))


@pytest.fixture
def claimed_control():
    return claim(control()).next_control


def test_submission_is_an_initial_control_and_input_only() -> None:
    plan = submission_plan(control(), b"input")
    assert plan.next_control.version == 1
    assert plan.payload_writes[0].kind is PayloadKind.INPUT
    assert plan.lease_index_update is None


def test_claim_fences_and_heartbeat_renews_without_business_version(claimed_control) -> None:
    claimed = claimed_control
    assert (claimed.status, claimed.version, claimed.fence, claimed.run_attempt) == (
        ExecutionStatus.RUNNING,
        2,
        1,
        1,
    )
    heartbeat_plan = decide(claimed, Heartbeat(1, "worker-a", 300, 500))
    heartbeat = heartbeat_plan.next_control
    assert heartbeat.version == claimed.version
    assert heartbeat.updated_at_ms == claimed.updated_at_ms
    assert heartbeat.lease_expires_at_ms == 800
    assert heartbeat_plan.lease_index_update and heartbeat_plan.lease_index_update.deadline_ms == 800
    assert not heartbeat_plan.payload_writes and not heartbeat_plan.progress_events
    with pytest.raises(ExecutionLeaseLostError):
        decide(heartbeat, Heartbeat(0, "worker-a", 301, 500))
    with pytest.raises(ExecutionLeaseLostError):
        decide(heartbeat, Heartbeat(1, "worker-a", 750, 500, 50))


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(Checkpoint(1, "stale", 300, 500, b"checkpoint"), id="checkpoint"),
        pytest.param(Complete(2, "worker-a", 300, b"result"), id="complete"),
        pytest.param(Fail(2, "worker-a", 300, b"error"), id="fail"),
        pytest.param(ReleaseClaim(1, "stale", 300), id="release"),
        pytest.param(ScheduleRetry(2, "worker-a", 300, 100, 2, b"error"), id="retry"),
        pytest.param(Suspend(2, "worker-a", 300, b"checkpoint", b"wait"), id="suspend"),
    ],
)
def test_owned_transitions_reject_stale_fences(command, claimed_control) -> None:
    with pytest.raises(ExecutionLeaseLostError):
        decide(claimed_control, command)


def test_cancellation_wins_every_owned_outcome(claimed_control) -> None:
    canceled = decide(claimed_control, RequestCancellation(250, "💥" * 2_000)).next_control
    assert canceled.cancel_reason == "💥" * 1_024
    assert len(canceled.cancel_reason.encode()) == 4_096
    commands = (
        Complete(1, "worker-a", 300, b"result", (b"progress",)),
        Fail(1, "worker-a", 300, b"error", (b"progress",)),
        ScheduleRetry(1, "worker-a", 300, 100, 2, b"error", (b"progress",)),
        Suspend(1, "worker-a", 300, b"checkpoint", b"wait", (b"progress",)),
    )
    for command in commands:
        terminal = decide(canceled, command)
        assert terminal.next_control.status is ExecutionStatus.CANCELED
        assert not terminal.payload_writes
        assert terminal.payload_deletes == (PayloadKind.RESULT, PayloadKind.ERROR, PayloadKind.WAIT)
        assert [event.data for event in terminal.progress_events] == [b"progress"]
    released = decide(canceled, ReleaseClaim(1, "worker-a", 300)).next_control
    reclaimed = decide(released, Claim("worker-b", 301, 500, 3, "rev-2", REVISION_ERROR, ATTEMPTS_ERROR))
    assert reclaimed.next_control.status is ExecutionStatus.CANCELED


def test_retry_and_lease_recovery_requeue_without_resetting_retry_count(claimed_control) -> None:
    retry = decide(
        claimed_control,
        ScheduleRetry(1, "worker-a", 300, 100, 2, b"retry", progress_events=(b"retrying",)),
    )
    queued = retry.next_control
    assert (
        queued.status,
        queued.available_at_ms,
        queued.run_attempt,
        queued.application_retry_count,
        queued.progress_sequence,
    ) == (
        ExecutionStatus.QUEUED,
        400,
        1,
        1,
        1,
    )
    assert retry.progress_events[0].data == b"retrying"
    next_claim = claim(replace(queued, available_at_ms=None), now_ms=400).next_control
    recovered = decide(
        next_claim,
        RecoverExpiredLease(
            1_000,
            next_claim.fence,
            next_claim.lease_expires_at_ms or 0,
            3,
            "rev-1",
            REVISION_ERROR,
            ATTEMPTS_ERROR,
        ),
    )
    assert recovered.next_control.status is ExecutionStatus.QUEUED
    assert recovered.next_control.application_retry_count == 1
    assert recovered.next_control.run_attempt == 2


def test_wait_resume_and_progress_preserve_checkpoint_boundary(claimed_control) -> None:
    waiting = decide(
        claimed_control,
        Suspend(1, "worker-a", 300, b"checkpoint", b"wait", progress_events=(b"waiting",)),
    )
    assert waiting.next_control.status is ExecutionStatus.WAITING
    assert waiting.lease_index_update and waiting.lease_index_update.deadline_ms is None
    resumed = decide(waiting.next_control, Resume(400, "rev-1", b"checkpoint", (b"resumed",)))
    assert resumed.next_control.status is ExecutionStatus.QUEUED
    assert [event.sequence for event in (*waiting.progress_events, *resumed.progress_events)] == [1, 2]


def test_terminal_state_is_irreversible_and_payloads_are_exclusive(claimed_control) -> None:
    completed = decide(claimed_control, Complete(1, "worker-a", 300, b"result"))
    assert completed.payload_deletes == (PayloadKind.ERROR, PayloadKind.WAIT)
    assert decide(completed.next_control, RequestCancellation(400, "late")).next_control == completed.next_control
    with pytest.raises(InvalidExecutionTransitionError):
        decide(
            completed.next_control,
            Claim("worker-a", 400, 500, 3, "rev-1", REVISION_ERROR, ATTEMPTS_ERROR),
        )
    failed = decide(claimed_control, Fail(1, "worker-a", 300, b"error"))
    assert failed.payload_deletes == (PayloadKind.RESULT, PayloadKind.WAIT)


def test_revision_mismatch_never_grants_a_fence() -> None:
    plan = decide(control(), Claim("worker-a", 200, 500, 3, "rev-2", REVISION_ERROR, ATTEMPTS_ERROR))
    assert plan.next_control.status is ExecutionStatus.FAILED
    assert plan.next_control.fence == 0


def test_stale_lease_index_is_removed_without_changing_control() -> None:
    current = control()
    plan = decide(current, RecoverExpiredLease(200, 1, 100, 3, "rev-1", REVISION_ERROR, ATTEMPTS_ERROR))
    assert plan.next_control == current
    assert plan.lease_index_update and plan.lease_index_update.deadline_ms is None


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        pytest.param({"cancel_requested_at_ms": 250}, ExecutionStatus.CANCELED, id="canceled"),
        pytest.param({"definition_revision": "rev-2"}, ExecutionStatus.FAILED, id="revision"),
        pytest.param({"run_attempt": 3}, ExecutionStatus.FAILED, id="attempts"),
    ],
)
def test_expired_lease_recovery_honors_cancellation_revision_and_attempt_budget(
    changes, expected, claimed_control
) -> None:
    current = replace(claimed_control, **changes)
    recovered = decide(
        current,
        RecoverExpiredLease(
            1_000,
            current.fence,
            current.lease_expires_at_ms or 0,
            3,
            "rev-1",
            REVISION_ERROR,
            ATTEMPTS_ERROR,
        ),
    )
    assert recovered.next_control.status is expected
