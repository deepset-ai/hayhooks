"""Redis layout tests that do not duplicate reducer decisions."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from hayhooks.durable.backend import MAINTENANCE_BATCH_SIZE, ExecutionStoreCorruptionError
from hayhooks.durable.engine import MAX_CONTROL_SCALAR_BYTES, initial_control
from hayhooks.durable.redis import RedisExecutionStore, RedisKeys, decode_control, digest, encode_control


def control(**changes: object):
    values = {
        "run_id": "run-1",
        "idempotency_digest": "a" * 64,
        "idempotency_binding_digest": "d" * 64,
        "deployment": "deployment",
        "definition_revision": "rev-1",
        "owner_id": "owner",
        "kind": "pipeline",
        "now_ms": 100,
    }
    values.update(changes)
    return initial_control(**values)


def test_key_builder_is_namespaced_hash_tagged_and_private() -> None:
    keys = RedisKeys("tenant:durable", "an unsafe deployment/name")
    idem = digest("idempotency", "raw-client-key")

    all_keys = (
        keys.runnable,
        keys.lease_expiry,
        keys.capacity,
        keys.control("run_1"),
        keys.idempotency(idem),
    )
    assert all("{" + keys.deployment_digest + "}" in key for key in all_keys)
    assert "unsafe" not in " ".join(all_keys)
    assert "raw-client-key" not in keys.idempotency(idem)
    assert RedisKeys.lease_member("run_1", 7) == "run_1|7"


def test_key_builder_rejects_user_influenced_unsafe_components() -> None:
    keys = RedisKeys("tenant:durable", "deployment")
    with pytest.raises(ValueError):
        keys.control("run:injection")
    with pytest.raises(ValueError):
        keys.idempotency("not-a-digest")


def test_control_hash_round_trip_omits_absent_optionals() -> None:
    original = control(owner_id=None)
    encoded = encode_control(original)
    restored = decode_control(encoded)

    assert restored == original
    assert "lease_owner" not in encoded
    assert "available_at_ms" not in encoded


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "idempotency_digest",
        "idempotency_binding_digest",
        "deployment",
        "definition_revision",
        "owner_id",
        "kind",
    ],
)
def test_control_rejects_scalars_redis_cannot_decode(field) -> None:
    with pytest.raises(ValueError, match=field):
        control(**{field: "x" * (MAX_CONTROL_SCALAR_BYTES + 1)})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values.pop("version"),
        lambda values: values.__setitem__("status", "unexpected"),
        lambda values: values.__setitem__("fence", "-1"),
        lambda values: values.__setitem__("run_id", "x" * 5_000),
        lambda values: values.__setitem__("lease_owner", "worker"),
    ],
)
def test_control_hash_corruption_is_rejected(mutate) -> None:
    values = encode_control(control())
    mutate(values)
    with pytest.raises(ExecutionStoreCorruptionError):
        decode_control(values)


async def test_maintenance_reads_a_fixed_batch_of_due_leases() -> None:
    redis = AsyncMock()
    redis.time.return_value = (123, 456_000)
    store = RedisExecutionStore(redis, deployment="deployment")

    await store.maintain(Mock())

    redis.time.assert_awaited_once()
    redis.zrangebyscore.assert_awaited_once_with(
        store.keys.lease_expiry,
        "-inf",
        123_456,
        start=0,
        num=MAINTENANCE_BATCH_SIZE,
        withscores=True,
    )
