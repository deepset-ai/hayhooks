"""Compatibility import path for Redis durable-execution storage."""

from hayhooks.durable import redis as _redis
from hayhooks.durable.redis import (
    EXECUTION_GROUP,
    RedisExecutionClaim,
    RedisExecutionStore,
    RedisExecutionStoreProvider,
)

# Kept as a module attribute for existing test and instrumentation patch points.
time = _redis.time

__all__ = ["EXECUTION_GROUP", "RedisExecutionClaim", "RedisExecutionStore", "RedisExecutionStoreProvider"]
