import asyncio
import contextlib
from typing import Any

import pytest
from fastapi import FastAPI
from haystack import component
from haystack.dataclasses import StreamingChunk

from hayhooks.server.pipelines.streaming import _SHIELDED_PIPELINE_TASKS, async_streaming_generator
from hayhooks.server.utils.haystack_compat import AsyncPipeline
from hayhooks.server.utils.streaming_response_utils import _streaming_response_from_async_gen


@component
class _BlockingStreamingComponent:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.completed = asyncio.Event()

    @component.output_types(result=str)
    def run(self, streaming_callback: Any | None = None) -> dict[str, str]:
        raise AssertionError("The async pipeline must call run_async")

    @component.output_types(result=str)
    async def run_async(self, streaming_callback: Any | None = None) -> dict[str, str]:
        await streaming_callback(StreamingChunk(content="first", index=0))
        await self.release.wait()
        self.completed.set()
        return {"result": "done"}


async def _disconnect_after_first_chunk(app: FastAPI) -> None:
    incoming = asyncio.Queue()
    first_chunk = asyncio.Event()
    await incoming.put({"type": "http.request", "body": b"", "more_body": False})

    async def receive():
        return await incoming.get()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
        "root_path": "",
    }
    request_task = asyncio.create_task(app(scope, receive, send))

    try:
        await asyncio.wait_for(first_chunk.wait(), timeout=1.0)
        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(request_task, timeout=1.0)
    finally:
        if not request_task.done():
            request_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request_task


@pytest.mark.integration
@pytest.mark.parametrize("shield_pipeline_task", [False, True], ids=["cancel", "shield"])
async def test_http_disconnect_pipeline_task(shield_pipeline_task):
    component = _BlockingStreamingComponent()
    pipeline = AsyncPipeline()
    pipeline.add_component("blocking", component)
    app = FastAPI()

    @app.get("/stream")
    async def stream():
        generator = async_streaming_generator(pipeline, shield_pipeline_task=shield_pipeline_task)
        return _streaming_response_from_async_gen(generator)

    tasks_before = set(_SHIELDED_PIPELINE_TASKS)
    await _disconnect_after_first_chunk(app)
    detached_tasks = _SHIELDED_PIPELINE_TASKS - tasks_before

    assert bool(detached_tasks) is shield_pipeline_task
    assert all(not task.done() for task in detached_tasks)

    component.release.set()
    if shield_pipeline_task:
        # A shielded task survives the disconnect and completes once released.
        # The default (cancel) case makes no completion promise: whether the
        # cancelled task or the release wins the race varies across Haystack
        # versions, so only the shield contract is asserted here.
        await asyncio.wait_for(component.completed.wait(), timeout=1.0)
    if detached_tasks:
        await asyncio.wait_for(asyncio.gather(*detached_tasks), timeout=1.0)
        await asyncio.sleep(0)
        assert detached_tasks.isdisjoint(_SHIELDED_PIPELINE_TASKS)
