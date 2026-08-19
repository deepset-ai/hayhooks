"""Polling helpers shared by the durable test modules."""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import requests

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


def read_sse_events(client: Any, url: str, *, headers: dict | None = None, limit: int | None = None) -> list[dict]:
    """Collect SSE events from a durable execution stream, skipping heartbeats."""
    events: list[dict] = []
    current: dict[str, str] = {}
    with client.stream("GET", url, headers=headers) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith(":"):
                continue
            if line:
                field, _, value = line.partition(": ")
                current[field] = value
            elif current:
                events.append(current)
                current = {}
                if limit is not None and len(events) >= limit:
                    break
    return events


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


def start_server(port: int, environment: dict[str, str], factory: str = "hayhooks.cli.base:get_app") -> Any:
    """Start one real Hayhooks process; the caller owns stopping it."""
    return subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            factory,
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
            "--no-access-log",
        ],
        cwd=Path.cwd(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_server(server: Any) -> None:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=3)


def server_error(server: Any) -> str:
    output = server.stdout.read() if server.stdout is not None else ""
    return f"durable test server exited with {server.returncode}:\n{output}"


def wait_for_server(server: Any, base_url: str, *, attempts: int = 300, delay: float = 0.1) -> None:
    """Poll ``/status`` until the process serves, failing fast if it died."""
    # Booting a real server imports Haystack and loads a pipeline, measured at 1.9 s
    # on a developer machine -- the whole of wait_until's in-process default budget,
    # which is why CI failed here. A dead process fails immediately below rather than
    # waiting this out, so a generous ceiling only costs time when it is deserved.

    def ready() -> bool:
        if server.poll() is not None:
            pytest.fail(server_error(server))
        try:
            return requests.get(f"{base_url}/status", timeout=0.25).status_code == 200
        except requests.RequestException:
            return False

    wait_until(ready, "durable test server did not become ready", attempts=attempts, delay=delay)


def cleanup_redis(redis_url: str, prefix: str) -> None:
    """Delete only the keys one test created under its own prefix."""
    from redis import Redis

    redis = Redis.from_url(redis_url)
    try:
        if keys := list(redis.scan_iter(match=f"{prefix}:*")):
            redis.delete(*keys)
    finally:
        redis.close()
