"""Redis codec and client-boundary checks."""

from __future__ import annotations

import re
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import ConnectionPool, Redis
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.retry import Retry

from hayhooks.durable.redis import RedisExecutionStore, RedisKeys, decode_control, encode_control
from hayhooks.durable.store import ExecutionStoreCorruptionError, ExecutionStoreError, StoreConfig, StreamChunk
from tests.durable_store_contract import contract_control


def test_redis_keys_are_private_cluster_safe_and_strict() -> None:
    keys = RedisKeys("tenant:durable", "unsafe deployment/name")
    generated = (
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
    field = RedisKeys.revision_nonterminal_field("v1")
    assert re.fullmatch(r"nonterminal:[0-9a-f]{64}", field)
    assert field.removeprefix("nonterminal:") == generated[0].rsplit(":", 1)[-1]
    with pytest.raises(ValueError):
        RedisKeys("unsafe prefix", "jobs")
    with pytest.raises(ValueError):
        keys.control("unsafe:run")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda values: values.pop("version"), id="missing"),
        pytest.param(lambda values: values.pop("lease_recoveries"), id="missing-lease-recoveries"),
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
    control = replace(contract_control("jobs"), run_attempt=3, lease_recoveries=2)
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


@pytest.mark.parametrize(
    "client",
    [
        pytest.param(Redis.from_url("redis://localhost", retry=Retry(NoBackoff(), 3)), id="retry-object"),
        pytest.param(Redis.from_url("redis://localhost?retry_on_timeout=true"), id="url-retry-on-timeout"),
        pytest.param(
            Redis(host="localhost", retry=None, retry_on_error=[RedisConnectionError]),
            id="retry-on-error",
        ),
    ],
)
def test_redis_store_rejects_clients_that_retry_commands(client: Redis) -> None:
    with pytest.raises(ValueError, match="does not retry commands"):
        RedisExecutionStore(client, "jobs")


def test_redis_store_rejects_a_viewer_that_retries_commands() -> None:
    worker = Redis.from_url("redis://localhost")
    viewer = Redis.from_url("redis://localhost", retry=Retry(NoBackoff(), 3))
    with pytest.raises(ValueError, match="does not retry commands"):
        RedisExecutionStore(worker, "jobs", viewer_client=viewer)


@pytest.mark.parametrize(
    "client",
    [
        pytest.param(Redis.from_url("redis://localhost"), id="from-url"),
        pytest.param(Redis(connection_pool=ConnectionPool.from_url("redis://localhost")), id="pool-from-url"),
        pytest.param(Redis(host="localhost", retry=None), id="retry-none"),
        pytest.param(Redis(host="localhost", retry=Retry(NoBackoff(), 0)), id="retry-zero"),
    ],
)
def test_redis_store_accepts_clients_without_command_retries(client: Redis) -> None:
    RedisExecutionStore(client, "jobs")


@pytest.mark.parametrize(
    "client",
    [
        pytest.param(Redis.from_url("redis://localhost", protocol=3), id="protocol-3"),
        pytest.param(Redis.from_url("redis://localhost?protocol=3"), id="url-protocol-3"),
        pytest.param(Redis.from_url("redis://localhost", legacy_responses=False), id="unified-replies"),
    ],
)
def test_redis_store_rejects_resp3_clients(client: Redis) -> None:
    with pytest.raises(ValueError, match="RESP2"):
        RedisExecutionStore(client, "jobs")


def mock_redis() -> AsyncMock:
    redis = AsyncMock(connection_pool=None)
    redis.register_script = MagicMock()
    return redis


async def test_redis_client_errors_are_normalized() -> None:
    redis = mock_redis()
    redis.info.side_effect = RedisConnectionError("secret endpoint")
    store = RedisExecutionStore(redis, "jobs")
    with pytest.raises(ExecutionStoreError) as raised:
        await store.initialize()
    assert str(raised.value) == "Redis durable store operation failed: ConnectionError"
    assert "secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ResponseError("WRONGTYPE Operation against a key holding the wrong kind of value"), "ResponseError WRONGTYPE"),
        (ResponseError("hash value is not an integer"), "ResponseError"),
    ],
)
async def test_redis_errors_name_the_response_code(error: ResponseError, message: str) -> None:
    redis = mock_redis()
    redis.info.side_effect = error
    store = RedisExecutionStore(redis, "jobs")
    with pytest.raises(ExecutionStoreError) as raised:
        await store.initialize()
    assert str(raised.value) == f"Redis durable store operation failed: {message}"


@pytest.mark.parametrize(
    ("info", "supported"),
    [
        pytest.param({"redis_version": "6.2.24"}, True, id="redis-floor"),
        pytest.param({b"redis_version": b"8.6.3"}, True, id="redis-latest"),
        pytest.param({"redis_version": "7.2.4", "valkey_version": "8.1.10"}, True, id="valkey"),
        pytest.param({"redis_version": "6.0.20"}, False, id="too-old"),
    ],
)
async def test_initialize_accepts_redis_6_2_and_valkey(info: dict, supported: bool) -> None:
    redis = mock_redis()
    redis.info.return_value = info
    store = RedisExecutionStore(redis, "jobs")
    if supported:
        await store.initialize()
    else:
        with pytest.raises(ExecutionStoreError, match=r"Redis 6\.2 or newer, or Valkey 7\.2 or newer"):
            await store.initialize()


def test_undecodable_stream_entries_are_skipped() -> None:
    store = RedisExecutionStore(mock_redis(), "jobs", config=StoreConfig(max_stream_chunk_bytes=1))
    chunks = store._decode_chunks(
        "run_1",
        [
            (b"1-0", {b"attempt": b"1", b"data": b"valid"}),
            (b"2-0", {b"bogus": b"1"}),
            (b"3-0", {b"attempt": b"x", b"data": b"bad"}),
            (b"4-0", {b"attempt": b"2", b"extra": b"y"}),
            (b"5-0", {b"attempt": b"2", b"terminal": b"completed"}),
            (b"6-0", {b"attempt": b"3", b"data": b"oversized"}),
        ],
    )
    assert chunks == (
        StreamChunk("1-0", 1, b"valid"),
        StreamChunk("2-0", 0, b"", skipped=True),
        StreamChunk("3-0", 0, b"", skipped=True),
        StreamChunk("4-0", 2, b"", skipped=True),
        StreamChunk("5-0", 2, b"", terminal=True),
        StreamChunk("6-0", 3, b"oversized"),
    )
