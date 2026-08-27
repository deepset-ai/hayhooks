"""Contract checks for durable store implementations."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hayhooks.durable.engine import Checkpoint, Claim, Complete, ExecutionPayloadSizeError
from hayhooks.durable.store import MemoryExecutionStore, StoreConfig, chunk_read_count
from tests.durable_store_contract import (
    ATTEMPTS_ERROR,
    CONTRACT_CONFIG,
    REVISION_ERROR,
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


def test_chunk_reads_are_bounded_by_bytes_entries_and_retention() -> None:
    assert chunk_read_count(StoreConfig(max_stream_chunk_bytes=1)) == 1_000
    assert chunk_read_count(StoreConfig(max_stream_chunks=3, max_stream_chunk_bytes=1)) == 3
    assert chunk_read_count(StoreConfig(max_stream_chunks=0, max_stream_chunk_bytes=1)) == 1


async def test_store_enforces_size_recovery_and_ttl(clock: Clock) -> None:
    store = MemoryExecutionStore(
        "jobs",
        clock=clock,
        config=StoreConfig(lease_commit_safety_ms=10, max_payload_bytes=8, terminal_ttl_seconds=1),
    )
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 100, 2, "v1", REVISION_ERROR, ATTEMPTS_ERROR))
    assert claimed is not None
    with pytest.raises(ExecutionPayloadSizeError):
        await store.transition("run_1", Checkpoint(1, "worker", 0, 100, b"too-large"))

    clock.now += 100
    await store.maintain(
        max_run_attempts=2,
        worker_revision="v1",
        revision_error=REVISION_ERROR,
        attempts_error=ATTEMPTS_ERROR,
    )
    reclaimed = await store.claim(Claim("worker", 0, 100, 2, "v1", REVISION_ERROR, ATTEMPTS_ERROR))
    assert reclaimed is not None
    await store.transition("run_1", Complete(2, "worker", 0, b"done"))
    clock.now += 1_000
    assert await store.read("run_1") is None


async def test_memory_store_repairs_only_the_stale_lease_member(clock: Clock) -> None:
    store = MemoryExecutionStore("jobs", clock=clock, config=StoreConfig(lease_commit_safety_ms=10))
    await store.submit(contract_control("jobs"), b"input")
    claimed = await store.claim(Claim("worker", 0, 500, 3, "v1", REVISION_ERROR, ATTEMPTS_ERROR))
    assert claimed is not None
    live_member = ("run_1", claimed.next_control.fence)
    live_deadline = store._lease_expiry[live_member]
    store._lease_expiry[("run_1", 0)] = clock.now
    await store.maintain(
        max_run_attempts=3,
        worker_revision="v1",
        revision_error=REVISION_ERROR,
        attempts_error=ATTEMPTS_ERROR,
    )
    assert store._lease_expiry == {live_member: live_deadline}


def test_importing_durable_loads_no_server_haystack_or_redis_modules() -> None:
    subprocess.run(
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
