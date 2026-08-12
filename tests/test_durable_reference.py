"""Shared contract and lean-index checks for the reference backend."""

from __future__ import annotations

import pytest

from hayhooks.durable.backend import ExecutionStoreConfig
from hayhooks.durable.engine import (
    Checkpoint,
    Claim,
    ExecutionPayloadSizeError,
    RecoverExpiredLease,
    RequestCancellation,
    Suspend,
)
from hayhooks.durable.reference import InMemoryExecutionStore
from tests.durable_contract import assert_store_contract, control


async def test_reference_store_matches_contract() -> None:
    store = InMemoryExecutionStore(
        deployment="integration",
        config=ExecutionStoreConfig(
            max_input_bytes=64,
            max_checkpoint_bytes=64,
            max_result_bytes=64,
            max_error_bytes=64,
            max_wait_bytes=64,
            max_progress_events=2,
            max_progress_event_bytes=32,
        ),
    )
    await store.initialize()
    await assert_store_contract(store)


async def test_reference_rejects_oversized_payload_before_transition() -> None:
    store = InMemoryExecutionStore(
        deployment="integration",
        config=ExecutionStoreConfig(
            max_input_bytes=64,
            max_checkpoint_bytes=8,
            max_result_bytes=64,
            max_error_bytes=64,
            max_wait_bytes=64,
            max_progress_events=2,
            max_progress_event_bytes=32,
        ),
    )
    await store.submit(control(), b"{}", binding_digest="b" * 64)
    run_id = await store.read_candidate()
    assert run_id is not None
    claim = await store.transition(run_id, Claim("worker", 0, 1_000, 3, "rev-1"), candidate=True)
    with pytest.raises(ExecutionPayloadSizeError):
        await store.transition(run_id, Checkpoint(claim.next_control.fence, "worker", 0, 1_000, b"too-large"))
    with pytest.raises(ExecutionPayloadSizeError):
        await store.transition(run_id, Suspend(claim.next_control.fence, "worker", 0, b"ok", b"x" * 65))
    assert await store.get(run_id) == claim.next_control


async def test_maintenance_repairs_only_the_stale_lease_member() -> None:
    store = InMemoryExecutionStore(deployment="integration")
    await store.submit(control(), b"{}", binding_digest="b" * 64)
    store._lease_expiry["run_1|7"] = 0

    def recover(fence: int, deadline: int) -> RecoverExpiredLease:
        return RecoverExpiredLease(0, fence, deadline, 3, "rev-1")

    await store.maintain(recover)
    assert "run_1|7" not in store._lease_expiry

    run_id = await store.read_candidate()
    assert run_id is not None
    claim = await store.transition(run_id, Claim("worker", 0, 1_000_000, 3, "rev-1"), candidate=True)
    live_member = f"{run_id}|{claim.next_control.fence}"
    live_deadline = store._lease_expiry[live_member]
    store._lease_expiry[f"{run_id}|0"] = 0
    await store.maintain(recover)
    assert store._lease_expiry == {live_member: live_deadline}


async def test_candidate_read_is_non_destructive_and_cancel_removes_runnable() -> None:
    store = InMemoryExecutionStore(deployment="integration")
    first = control("run_a", idempotency_digest="a" * 64, binding_digest="b" * 64)
    second = control("run_b", idempotency_digest="c" * 64, binding_digest="d" * 64)
    await store.submit(first, b"{}", binding_digest="b" * 64)
    await store.submit(second, b"{}", binding_digest="d" * 64)
    assert await store.read_candidate() == "run_a"
    assert await store.read_candidate() == "run_a"
    await store.transition("run_a", RequestCancellation(0, "cancel"))
    await store.transition("run_b", RequestCancellation(0, "cancel"))
    assert await store.read_candidate() is None
    assert await store.operational_counts() == {"nonterminal": 0, "runnable": 0, "lease_expiry": 0}
