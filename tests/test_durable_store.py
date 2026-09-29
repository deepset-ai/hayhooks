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
    ExecutionLeaseLostError,
    ExecutionPayloadSizeError,
    ExecutionStatus,
    PayloadKind,
)
from hayhooks.durable.store import ExecutionStoreCorruptionError, MemoryExecutionStore, StoreConfig, chunk_read_count
from tests.durable_store_contract import (
    ATTEMPTS_ERROR,
    CONTRACT_CONFIG,
    assert_revision_routing_contract,
    assert_store_contract,
    contract_control,
)


class Clock:
    now = 1_000

    def __call__(self) -> int:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


async def test_memory_store_matches_contract(clock: Clock) -> None:
    store = MemoryExecutionStore("jobs", clock=clock, config=CONTRACT_CONFIG)
    await store.initialize()
    await assert_store_contract(store)


async def test_memory_store_routes_claims_by_revision(clock: Clock) -> None:
    store = MemoryExecutionStore("jobs", clock=clock, config=CONTRACT_CONFIG)
    await assert_revision_routing_contract(store)


def test_chunk_reads_are_bounded_by_bytes_entries_and_retention() -> None:
    assert chunk_read_count(StoreConfig(max_stream_chunks=10_000, max_stream_chunk_bytes=1)) == 1_000
    assert chunk_read_count(StoreConfig(max_stream_chunks=3, max_stream_chunk_bytes=1)) == 3
    assert chunk_read_count(StoreConfig(max_stream_chunks=0, max_stream_chunk_bytes=1)) == 1


def test_portable_store_defaults_bound_admission_and_stream_history() -> None:
    config = StoreConfig()
    assert config.max_nonterminal_executions == 1_000
    assert config.max_stream_chunks == 100


async def test_store_enforces_size_recovery_and_ttl(clock: Clock) -> None:
    store = MemoryExecutionStore(
        "jobs",
        clock=clock,
        config=StoreConfig(lease_commit_safety_ms=10, max_payload_bytes=8, terminal_ttl_seconds=1),
    )
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 100, 2, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    with pytest.raises(ExecutionPayloadSizeError):
        await store.transition("run_1", Checkpoint(1, "worker", 0, 100, b"too-large"))

    clock.now += 100
    await store.maintain(
        max_run_attempts=2,
        attempts_error=ATTEMPTS_ERROR,
    )
    reclaimed = await store.claim(Claim("worker", 0, 100, 2, "v1", ATTEMPTS_ERROR))
    assert reclaimed is not None
    await store.transition("run_1", Complete(2, "worker", 0, b"done"))
    clock.now += 1_000
    assert await store.read("run_1") is None


async def test_memory_store_repairs_only_the_stale_lease_member(clock: Clock) -> None:
    store = MemoryExecutionStore("jobs", clock=clock, config=StoreConfig(lease_commit_safety_ms=10))
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    live_member = ("run_1", claimed.next_control.fence)
    live_deadline = store._lease_expiry[live_member]
    store._lease_expiry[("run_1", 0)] = clock.now
    await store.maintain(
        max_run_attempts=3,
        attempts_error=ATTEMPTS_ERROR,
    )
    assert store._lease_expiry == {live_member: live_deadline}


@pytest.mark.parametrize(
    ("status", "payloads"),
    [
        pytest.param(ExecutionStatus.COMPLETED, {}, id="completed-result"),
        pytest.param(ExecutionStatus.FAILED, {}, id="failed-error"),
        pytest.param(ExecutionStatus.WAITING, {PayloadKind.CHECKPOINT: b"{}"}, id="waiting-wait"),
        pytest.param(ExecutionStatus.WAITING, {PayloadKind.WAIT: b"{}"}, id="waiting-checkpoint"),
    ],
)
async def test_memory_store_rejects_missing_lifecycle_payloads(
    clock: Clock,
    status: ExecutionStatus,
    payloads: dict[PayloadKind, bytes],
) -> None:
    store = MemoryExecutionStore("jobs", clock=clock, config=CONTRACT_CONFIG)
    await store.submit(contract_control("jobs"), b"input")
    store._controls["run_1"] = replace(store._controls["run_1"], status=status)
    store._payloads["run_1"] = {PayloadKind.INPUT: b"input", **payloads}

    with pytest.raises(ExecutionStoreCorruptionError, match="payload"):
        await store.read("run_1")


@pytest.mark.parametrize(
    ("fence", "worker_id", "terminal"),
    [
        pytest.param(0, "worker", False, id="stale-fence"),
        pytest.param(1, "stale", False, id="stale-owner"),
        pytest.param(1, "worker", True, id="terminal"),
    ],
)
async def test_memory_store_fences_stream_chunks(
    clock: Clock,
    fence: int,
    worker_id: str,
    terminal: bool,
) -> None:
    store = MemoryExecutionStore("jobs", clock=clock, config=CONTRACT_CONFIG)
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is not None
    if terminal:
        await store.transition("run_1", Complete(1, "worker", 0, b"done"))

    with pytest.raises(ExecutionLeaseLostError):
        await store.append_chunk("run_1", 1, fence, worker_id, b"stale")
    assert await store.read_chunks("run_1", "0-0") == ()


def test_importing_durable_loads_no_server_haystack_or_redis_modules() -> None:
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            (
                "import sys; import hayhooks.durable; "
                "assert not any(name == 'hayhooks.server' or name.startswith('hayhooks.server.') "
                "for name in sys.modules); "
                "assert not any(name == 'haystack' or name.startswith('haystack.') for name in sys.modules)"
                "; assert not any(name == 'redis' or name.startswith('redis.') for name in sys.modules)"
            ),
        ],
        check=True,
    )
