import asyncio
from collections.abc import AsyncGenerator, Generator
from threading import get_ident

import pytest
from haystack.dataclasses import StreamingChunk

from hayhooks.server.routers.openai import _normalize_result


@pytest.mark.asyncio
async def test_non_streaming_sync_generator_runs_off_event_loop():
    event_loop_thread = get_ident()
    iterations = []

    def chunks() -> Generator[StreamingChunk, None, None]:
        iterations.append(get_ident())
        yield StreamingChunk(content="hello ")
        iterations.append(get_ident())
        yield StreamingChunk(content="world")

    assert await _normalize_result(chunks(), stream_requested=False) == "hello world"
    assert iterations and all(thread != event_loop_thread for thread in iterations)


@pytest.mark.asyncio
async def test_non_streaming_async_generator_stays_on_event_loop():
    loop = asyncio.get_running_loop()

    async def chunks() -> AsyncGenerator[StreamingChunk, None]:
        assert asyncio.get_running_loop() is loop
        yield StreamingChunk(content="hello")

    assert await _normalize_result(chunks(), stream_requested=False) == "hello"


@pytest.mark.asyncio
async def test_streaming_sync_generator_is_not_consumed():
    def chunks() -> Generator[str, None, None]:
        message = "The streaming consumer must drive this generator"
        raise AssertionError(message)
        yield "unreachable"

    stream = chunks()
    assert await _normalize_result(stream, stream_requested=True) is stream
    stream.close()
