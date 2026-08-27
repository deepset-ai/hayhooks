"""Redis codec and client-boundary checks."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from hayhooks.durable.engine import Claim, ExecutionStatus
from hayhooks.durable.redis import RedisExecutionStore, RedisKeys, decode_control, encode_control
from hayhooks.durable.store import ExecutionStoreCorruptionError, ExecutionStoreError
from tests.durable_store_contract import ATTEMPTS_ERROR, contract_control


def test_redis_keys_are_private_cluster_safe_and_strict() -> None:
    keys = RedisKeys("tenant:durable", "unsafe deployment/name")
    generated = (
        keys.runnable,
        keys.runnable_revision("v1"),
        keys.lease_expiry,
        keys.capacity,
        keys.control("run_1"),
        keys.idempotency("raw-client-material"),
    )
    hash_tags = {key[key.index("{") : key.index("}") + 1] for key in generated}
    assert len(hash_tags) == 1
    assert "unsafe" not in " ".join(generated)
    assert "raw-client-material" not in generated[-1]
    with pytest.raises(ValueError):
        RedisKeys("unsafe prefix", "jobs")
    with pytest.raises(ValueError):
        keys.control("unsafe:run")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda values: values.pop("version"), id="missing"),
        pytest.param(lambda values: values.__setitem__("unknown", "value"), id="unknown"),
        pytest.param(lambda values: values.__setitem__("status", "unknown"), id="status"),
        pytest.param(lambda values: values.__setitem__("fence", "-1"), id="negative"),
        pytest.param(lambda values: values.__setitem__("run_id", "x" * 5_000), id="oversized"),
        pytest.param(lambda values: values.__setitem__("lease_owner", "worker"), id="contradictory"),
        pytest.param(
            lambda values: values.update(status="completed", available_at_ms="1"),
            id="invalid-schedule",
        ),
    ],
)
def test_control_codec_rejects_corruption(mutate) -> None:
    encoded = encode_control(contract_control("jobs"))
    mutate(encoded)
    with pytest.raises(ExecutionStoreCorruptionError):
        decode_control(encoded, expected_run_id="run_1")


def test_control_codec_round_trip() -> None:
    control = contract_control("jobs")
    encoded = encode_control(control)
    assert encoded["schema_version"] == "1"
    assert decode_control(encoded, expected_run_id=control.run_id) == control


def test_control_codec_rejects_an_unsupported_schema_version() -> None:
    encoded = encode_control(contract_control("jobs"))
    encoded["schema_version"] = "2"
    with pytest.raises(ExecutionStoreCorruptionError, match="schema version"):
        decode_control(encoded, expected_run_id="run_1")


def test_redis_store_rejects_text_decoding_clients() -> None:
    client = Redis.from_url("redis://localhost", decode_responses=True)
    with pytest.raises(ValueError, match="decode_responses=False"):
        RedisExecutionStore(client, "jobs")


async def test_redis_client_errors_are_normalized() -> None:
    redis = AsyncMock()
    redis.connection_pool = None
    redis.info.side_effect = RedisConnectionError("secret endpoint")
    store = RedisExecutionStore(redis, "jobs")
    with pytest.raises(ExecutionStoreError, match="Redis durable store operation failed"):
        await store.initialize()


async def test_empty_scheduling_indexes_do_not_request_redis_time() -> None:
    redis = AsyncMock()
    redis.connection_pool = None
    redis.zrange.return_value = []
    store = RedisExecutionStore(redis, "jobs")

    claimed = await store.claim(Claim("worker", 0, 30_000, 3, "v1", ATTEMPTS_ERROR))
    await store.maintain(
        max_run_attempts=3,
        attempts_error=ATTEMPTS_ERROR,
    )

    assert claimed is None
    assert redis.zrange.await_count == 2
    redis.time.assert_not_awaited()


async def test_future_scheduling_scores_use_redis_time_and_are_not_processed() -> None:
    redis = AsyncMock()
    redis.connection_pool = None
    redis.time.return_value = (1, 0)
    store = RedisExecutionStore(redis, "jobs")
    store._transition = AsyncMock()  # type: ignore[method-assign]

    redis.zrange.return_value = [(b"run_1", 2_000.0)]
    claimed = await store.claim(Claim("worker", 0, 30_000, 3, "v1", ATTEMPTS_ERROR))
    assert claimed is None

    redis.zrange.return_value = [(b"run_1|1", 2_000.0)]
    await store.maintain(
        max_run_attempts=3,
        attempts_error=ATTEMPTS_ERROR,
    )

    assert redis.time.await_count == 2
    store._transition.assert_not_awaited()


@pytest.mark.parametrize("score", [float("inf"), -1.0, 1.5])
async def test_invalid_runnable_scores_report_corruption(score: float) -> None:
    redis = AsyncMock()
    redis.connection_pool = None
    redis.zrange.return_value = [(b"run_1", score)]
    store = RedisExecutionStore(redis, "jobs")

    with pytest.raises(ExecutionStoreCorruptionError, match="runnable index"):
        await store.claim(Claim("worker", 0, 30_000, 3, "v1", ATTEMPTS_ERROR))
    redis.time.assert_not_awaited()


async def test_invalid_lease_entries_are_removed_without_requesting_time() -> None:
    redis = AsyncMock()
    redis.connection_pool = None
    redis.zrange.return_value = [(b"run_1|1", float("inf"))]
    store = RedisExecutionStore(redis, "jobs")

    await store.maintain(
        max_run_attempts=3,
        attempts_error=ATTEMPTS_ERROR,
    )

    redis.zrem.assert_awaited_once_with(store.keys.lease_expiry, b"run_1|1")
    redis.time.assert_not_awaited()


async def test_chunk_append_is_bounded_and_sets_a_rolling_ttl() -> None:
    pipe = MagicMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=None)
    pipe.watch = AsyncMock()
    pipe.hgetall = AsyncMock(
        return_value=encode_control(
            replace(
                contract_control("jobs"),
                status=ExecutionStatus.RUNNING,
                fence=1,
                run_attempt=2,
                lease_owner="worker",
                lease_expires_at_ms=30_000,
            )
        )
    )
    pipe.time = AsyncMock(return_value=(1, 0))
    pipe.execute = AsyncMock(return_value=[])
    redis = MagicMock(connection_pool=None)
    redis.pipeline.return_value = pipe
    store = RedisExecutionStore(redis, "jobs")

    await store.append_chunk("run_1", 2, 1, "worker", b"chunk")

    pipe.xadd.assert_called_once_with(
        store.keys.chunks("run_1"),
        {"attempt": 2, "data": b"chunk"},
        maxlen=store.config.max_stream_chunks,
        approximate=False,
    )
    pipe.expire.assert_called_once_with(store.keys.chunks("run_1"), store.config.terminal_ttl_seconds)
    pipe.execute.assert_awaited_once()
