"""Polling helpers shared by the durable test modules."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any

import pytest

_ATTEMPTS = 200
_DELAY = 0.01


def wait_until(predicate: Callable[[], Any], message: str, *, attempts: int = _ATTEMPTS, delay: float = _DELAY) -> Any:
    """Poll *predicate* until it returns something truthy, then return it."""
    for _ in range(attempts):
        if value := predicate():
            return value
        time.sleep(delay)
    pytest.fail(message)


async def wait_until_async(
    predicate: Callable[[], Any], message: str, *, attempts: int = _ATTEMPTS, delay: float = _DELAY
) -> Any:
    """Async counterpart to :func:`wait_until`; the predicate may be awaitable."""
    for _ in range(attempts):
        value = predicate()
        if inspect.isawaitable(value):
            value = await value
        if value:
            return value
        await asyncio.sleep(delay)
    pytest.fail(message)


def wait_for_status(client: Any, url: str, status: str, **kwargs: Any) -> dict:
    """Poll a durable execution resource until it reports *status*."""

    def ready() -> dict | None:
        body = client.get(url).json()
        return body if body.get("status") == status else None

    return wait_until(ready, f"execution at {url} did not become {status}", **kwargs)


async def wait_for_record(
    source: Any,
    execution_id: str,
    predicate: Callable[[Any], bool] = lambda record: record.terminal,
    *,
    message: str = "durable execution did not reach its expected state",
    **kwargs: Any,
) -> Any:
    """Poll a store or deployment until its record satisfies *predicate*."""

    async def ready() -> Any:
        record = await source.get(execution_id)
        return record if record is not None and predicate(record) else None

    return await wait_until_async(ready, message, **kwargs)
