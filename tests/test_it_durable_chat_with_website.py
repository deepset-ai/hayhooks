"""Three real chat-with-website executions, streamed concurrently from one server."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from tests.durable_helpers import cleanup_redis, start_server, stop_server, wait_for_server

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not importlib.metadata.version("haystack-ai").startswith("3."),
        reason="durable execution requires Haystack 3",
    ),
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is required"),
    pytest.mark.skipif(not os.environ.get("HAYHOOKS_TEST_REDIS_URL"), reason="HAYHOOKS_TEST_REDIS_URL is required"),
]

_PIPELINES_DIR = Path("examples/durable_chat_with_website/pipelines")
_URL = "https://haystack.deepset.ai"
# A real question each, plus a unique marker the answer must lead with. The marker
# is what makes a chunk landing on the wrong stream detectable.
_ASKS = {
    "ALPHA7451": "What is Haystack used for?",
    "BETA2298": "Which programming language is Haystack written in?",
    "GAMMA6013": "Name one thing you can build with Haystack.",
}


def _query(marker: str, question: str) -> str:
    return f"{question} Begin your reply with the exact token {marker} and then answer in one short sentence."


async def _ask(client: httpx.AsyncClient, marker: str, question: str) -> dict:
    """Submit, attach immediately, and collect the streamed answer and terminal body."""
    submitted = await client.post(
        "/chat_with_website/run-durable", json={"question": _query(marker, question), "urls": [_URL]}
    )
    assert submitted.status_code == 202, submitted.text

    chunks: list[str] = []
    terminal: dict | None = None
    fields: dict[str, str] = {}
    async with client.stream("GET", submitted.json()["links"]["stream"]) as response:
        assert response.status_code == 200, await response.aread()
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        async for line in response.aiter_lines():
            if line.startswith(":"):
                continue
            if line:
                name, _, value = line.partition(": ")
                fields[name] = value
            elif fields:
                if fields["event"] == "chunk":
                    chunks.append(json.loads(fields["data"])["payload"]["content"] or "")
                else:
                    terminal = json.loads(fields["data"])
                fields = {}
    assert terminal is not None, f"{marker} stream ended without a terminal event"
    # More than one chunk event proves the socket delivered the answer incrementally
    # rather than the whole body arriving at once.
    assert len(chunks) > 1, f"{marker} arrived in one piece, so nothing was really streamed"
    return {"text": "".join(chunks), "terminal": terminal}


def _processing_window(terminal: dict) -> tuple[datetime, datetime]:
    """The server-side span between an execution's first and last progress events."""
    stamps = [datetime.fromisoformat(event["timestamp"]) for event in terminal["progress"]]
    return min(stamps), max(stamps)


async def test_three_concurrent_real_executions_stream_independently(unused_tcp_port: int) -> None:
    key_prefix = f"hayhooks:test:{uuid.uuid4().hex}"
    redis_url = os.environ["HAYHOOKS_TEST_REDIS_URL"]
    base_url = f"http://127.0.0.1:{unused_tcp_port}"
    environment = os.environ | {
        "HAYHOOKS_PIPELINES_DIR": str(_PIPELINES_DIR),
        "HAYHOOKS_DURABLE_STORE": "redis",
        "HAYHOOKS_DURABLE_REDIS_URL": redis_url,
        "HAYHOOKS_DURABLE_REDIS_KEY_PREFIX": key_prefix,
        "HAYHOOKS_DURABLE_EXECUTION_CONCURRENCY": str(len(_ASKS)),
        # The production default polls every second, which would let three workers
        # claim up to a second apart and turn the overlap below into a coin flip.
        "HAYHOOKS_DURABLE_POLL_INTERVAL": "0.05",
    }

    server = start_server(unused_tcp_port, environment)
    try:
        wait_for_server(server, base_url)
        async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
            streams = await asyncio.gather(*(_ask(client, marker, question) for marker, question in _ASKS.items()))
    finally:
        stop_server(server)
        cleanup_redis(redis_url, key_prefix)

    for marker, stream in zip(_ASKS, streams, strict=True):
        terminal = stream["terminal"]
        assert terminal["status"] == "completed", terminal
        assert marker in stream["text"], f"{marker} never arrived on its own stream: {stream['text'][:200]}"
        # The load-bearing assertion: no stream carries another execution's tokens.
        assert not [other for other in _ASKS if other != marker and other in stream["text"]]
        # Streamed chunks must reconstruct exactly the durable result.
        assert stream["text"] == terminal["result"]["reply"]
        assert terminal["result"]["urls"] == [_URL]

    # The executions overlapped on the server, so the isolation above was proven
    # against three genuinely concurrent runs rather than three that queued up.
    windows = [_processing_window(stream["terminal"]) for stream in streams]
    assert max(start for start, _ in windows) < min(end for _, end in windows)
