import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from haystack import AsyncPipeline, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import StreamingChunk

from hayhooks.server.pipelines.streaming import async_streaming_generator, streaming_generator

QUESTION = "Is Haystack a framework for developing AI applications? Answer Yes or No"


@component
class MockSlowGenerator:
    """Mock generator that takes a long time to complete (for testing early termination)."""

    @component.output_types(replies=list[str])
    def run(self, prompt: str, streaming_callback: Any | None = None) -> dict[str, Any]:
        if streaming_callback:
            streaming_callback(StreamingChunk(content="First chunk", index=0))
        time.sleep(5)
        if streaming_callback:
            streaming_callback(StreamingChunk(content="Second chunk", index=1))
        return {"replies": ["Done"]}


@component
class MockSyncOnlyGenerator:
    """Mock generator that only supports sync streaming callbacks (no run_async)."""

    @component.output_types(replies=list[str])
    def run(self, prompt: str, streaming_callback: Any | None = None) -> dict[str, Any]:
        chunks = ["This ", "is ", "a ", "test ", "response."]
        if streaming_callback:
            for i, chunk_text in enumerate(chunks):
                streaming_callback(StreamingChunk(content=chunk_text, index=i))
        return {"replies": ["This is a test response."]}


@component
class MockAsyncGenerator:
    """Mock generator that supports both sync and async streaming callbacks."""

    def __init__(self):
        # Set output types in constructor to satisfy Haystack's requirement
        component.set_output_types(self, replies=list[str])

    def run(self, prompt: str, streaming_callback: Any | None = None) -> dict[str, Any]:
        chunks = ["Async ", "test ", "response."]
        if streaming_callback:
            for i, chunk_text in enumerate(chunks):
                streaming_callback(StreamingChunk(content=chunk_text, index=i))
        return {"replies": ["Async test response."]}

    async def run_async(self, prompt: str, streaming_callback: Any | None = None) -> dict[str, Any]:
        chunks = ["Async ", "test ", "response."]
        if streaming_callback:
            for i, chunk_text in enumerate(chunks):
                await streaming_callback(StreamingChunk(content=chunk_text, index=i))
        return {"replies": ["Async test response."]}


@component
class MockFailingGenerator:
    """Mock generator that raises an exception during execution."""

    def __init__(self, fail_after_chunks: int = 0):
        self.fail_after_chunks = fail_after_chunks
        # Set output types in constructor to satisfy Haystack's requirement
        component.set_output_types(self, replies=list[str])

    def run(self, prompt: str, streaming_callback: Any | None = None) -> dict[str, Any]:
        if streaming_callback:
            for i in range(self.fail_after_chunks):
                streaming_callback(StreamingChunk(content=f"Chunk {i}", index=i))
        msg = "Pipeline execution failed"
        raise RuntimeError(msg)

    async def run_async(self, prompt: str, streaming_callback: Any | None = None) -> dict[str, Any]:
        if streaming_callback:
            for i in range(self.fail_after_chunks):
                await streaming_callback(StreamingChunk(content=f"Chunk {i}", index=i))
        msg = "Async pipeline execution failed"
        raise RuntimeError(msg)


@pytest.fixture
def pipeline_with_sync_only_generator():
    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", MockSyncOnlyGenerator())
    pipe.connect("prompt_builder.prompt", "llm.prompt")
    return pipe


@pytest.fixture
def async_pipeline_with_sync_only_generator():
    pipe = AsyncPipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", MockSyncOnlyGenerator())
    pipe.connect("prompt_builder.prompt", "llm.prompt")
    return pipe


@pytest.fixture
def pipeline_with_slow_generator():
    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", MockSlowGenerator())
    pipe.connect("prompt_builder.prompt", "llm.prompt")
    return pipe


@pytest.fixture
def create_mock_agent(mocker):  # noqa: C901
    def _factory(  # noqa: C901
        chunks: list[StreamingChunk] | None = None,
        result: dict[str, Any] | None = None,
        raise_exception: Exception | None = None,
    ):
        if chunks is None:
            chunks = [
                StreamingChunk(content="Agent ", index=0),
                StreamingChunk(content="response ", index=1),
                StreamingChunk(content="here.", index=2),
            ]
        if result is None:
            result = {"messages": ["Agent response here."]}

        mock_agent = mocker.Mock(spec=Agent)

        def mock_run(messages=None, streaming_callback=None, **kwargs):
            if raise_exception:
                raise raise_exception
            if streaming_callback:
                for chunk in chunks:
                    streaming_callback(chunk)
            return result

        async def mock_run_async(messages=None, streaming_callback=None, **kwargs):
            if raise_exception:
                raise raise_exception
            if streaming_callback:
                for chunk in chunks:
                    await streaming_callback(chunk)
            return result

        mock_agent.run.side_effect = mock_run
        mock_agent.run_async = mocker.AsyncMock(side_effect=mock_run_async)
        return mock_agent

    return _factory


def test_streaming_generator_with_sync_only_generator(pipeline_with_sync_only_generator):
    generator = streaming_generator(pipeline_with_sync_only_generator, pipeline_run_args={})

    assert isinstance(generator, Generator)
    chunks = list(generator)
    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]

    assert len(streaming_chunks) == 5
    assert streaming_chunks[0].content == "This "


async def test_async_streaming_rejects_sync_only_components(async_pipeline_with_sync_only_generator):
    with pytest.raises(ValueError, match="seems to not support async streaming callbacks"):
        async_gen = async_streaming_generator(async_pipeline_with_sync_only_generator, pipeline_run_args={})
        _ = [chunk async for chunk in async_gen]


async def test_async_streaming_hybrid_mode(async_pipeline_with_sync_only_generator):
    async_gen = async_streaming_generator(
        async_pipeline_with_sync_only_generator,
        pipeline_run_args={},
        allow_sync_streaming_callbacks=True,
    )

    assert isinstance(async_gen, AsyncGenerator)
    chunks = [chunk async for chunk in async_gen]
    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]

    assert len(streaming_chunks) == 5


async def test_async_streaming_does_not_mutate_args(async_pipeline_with_sync_only_generator):
    original_args: dict[str, Any] = {}

    async_gen = async_streaming_generator(
        async_pipeline_with_sync_only_generator,
        pipeline_run_args=original_args,
        allow_sync_streaming_callbacks=True,
    )
    _ = [chunk async for chunk in async_gen]

    assert original_args == {}


def test_streaming_generator_early_termination(pipeline_with_slow_generator):
    start_time = time.time()
    generator = streaming_generator(pipeline_with_slow_generator, pipeline_run_args={})

    for _chunk in generator:
        break  # Stop after first chunk

    generator.close()
    elapsed = time.time() - start_time

    assert elapsed < 2.0, f"Early termination took {elapsed:.2f}s, should be < 2s"


def test_streaming_generator_pipeline_exception():
    from haystack.core.errors import PipelineRuntimeError

    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", MockFailingGenerator())
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    generator = streaming_generator(pipe, pipeline_run_args={})

    with pytest.raises(PipelineRuntimeError, match="Pipeline execution failed"):
        list(generator)


async def test_async_streaming_generator_pipeline_exception():
    from haystack.core.errors import PipelineRuntimeError

    pipe = AsyncPipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", MockFailingGenerator())
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    async_gen = async_streaming_generator(pipe, pipeline_run_args={})

    with pytest.raises(PipelineRuntimeError, match="Async pipeline execution failed"):
        _ = [chunk async for chunk in async_gen]


async def test_async_streaming_exception_after_chunks():
    from haystack.core.errors import PipelineRuntimeError

    pipe = AsyncPipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", MockFailingGenerator(fail_after_chunks=2))
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    async_gen = async_streaming_generator(pipe, pipeline_run_args={})
    received_chunks = []

    with pytest.raises(PipelineRuntimeError, match="Async pipeline execution failed"):
        async for chunk in async_gen:
            if isinstance(chunk, StreamingChunk):
                received_chunks.append(chunk)

    # Should have received some chunks before the exception
    assert len(received_chunks) == 2


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_agent_streaming(create_mock_agent, use_async):
    mock_agent = create_mock_agent()

    if use_async:
        gen = async_streaming_generator(mock_agent, pipeline_run_args={"messages": []})
        chunks = [chunk async for chunk in gen]
    else:
        gen = streaming_generator(mock_agent, pipeline_run_args={"messages": []})
        chunks = list(gen)

    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]
    assert len(streaming_chunks) == 3
    assert streaming_chunks[0].content == "Agent "


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_agent_streaming_empty(create_mock_agent, use_async):
    mock_agent = create_mock_agent(chunks=[])

    if use_async:
        gen = async_streaming_generator(mock_agent, pipeline_run_args={"messages": []})
        chunks = [chunk async for chunk in gen]
    else:
        gen = streaming_generator(mock_agent, pipeline_run_args={"messages": []})
        chunks = list(gen)

    assert chunks == []


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_agent_on_pipeline_end(create_mock_agent, use_async):
    mock_agent = create_mock_agent()
    captured_result = {}

    def on_pipeline_end(result: dict[str, Any]) -> str:
        captured_result.update(result)
        return "Final output"

    if use_async:
        gen = async_streaming_generator(mock_agent, pipeline_run_args={"messages": []}, on_pipeline_end=on_pipeline_end)
        chunks = [chunk async for chunk in gen]
    else:
        gen = streaming_generator(mock_agent, pipeline_run_args={"messages": []}, on_pipeline_end=on_pipeline_end)
        chunks = list(gen)

    assert len(chunks) == 4
    assert chunks[-1].content == "Final output"
    assert captured_result == {"messages": ["Agent response here."]}


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_agent_does_not_mutate_args(create_mock_agent, use_async):
    mock_agent = create_mock_agent()
    original_args = {"messages": [{"role": "user", "content": "Hello"}]}
    expected = {"messages": [{"role": "user", "content": "Hello"}]}

    if use_async:
        gen = async_streaming_generator(mock_agent, pipeline_run_args=original_args)
        _ = [chunk async for chunk in gen]
    else:
        gen = streaming_generator(mock_agent, pipeline_run_args=original_args)
        list(gen)

    assert original_args == expected


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_agent_exception(create_mock_agent, use_async):
    mock_agent = create_mock_agent(raise_exception=RuntimeError("Agent failed"))

    if use_async:
        gen = async_streaming_generator(mock_agent, pipeline_run_args={"messages": []})
        with pytest.raises(RuntimeError, match="Agent failed"):
            _ = [chunk async for chunk in gen]
    else:
        gen = streaming_generator(mock_agent, pipeline_run_args={"messages": []})
        with pytest.raises(RuntimeError, match="Agent failed"):
            list(gen)


async def test_async_streaming_cancellation():
    pipe = AsyncPipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", MockAsyncGenerator())
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    async def consume_with_cancel():
        gen = async_streaming_generator(pipe, pipeline_run_args={})
        chunks_received = []
        async for chunk in gen:
            chunks_received.append(chunk)
            if len(chunks_received) >= 1:
                # Cancel after receiving first chunk
                break
        return chunks_received

    chunks = await consume_with_cancel()
    assert len(chunks) >= 1
