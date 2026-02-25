import contextvars
import os
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from haystack import AsyncPipeline, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.dataclasses.streaming_chunk import ToolCallDelta
from haystack.utils import Secret

from hayhooks.events import PipelineEvent
from hayhooks.server.pipelines.streaming import _process_tool_call_start, async_streaming_generator, streaming_generator

QUESTION = "Is Haystack a framework for developing AI applications? Answer Yes or No"

# skip decorator for tests requiring OpenAI API key
requires_openai_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)


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
    """
    Pipeline with OpenAIGenerator - a sync-only component (no run_async method).
    This is a real integration test fixture requiring OPENAI_API_KEY.
    """
    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini"))
    pipe.connect("prompt_builder.prompt", "llm.prompt")
    return pipe


@pytest.fixture
def async_pipeline_with_sync_only_generator():
    """
    AsyncPipeline with OpenAIGenerator - a sync-only component (no run_async method).
    This demonstrates the case where async pipeline contains sync-only streaming components.
    """
    pipe = AsyncPipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini"))
    pipe.connect("prompt_builder.prompt", "llm.prompt")
    return pipe


@pytest.fixture
def pipeline_with_slow_generator():
    """Pipeline with MockSlowGenerator for testing early termination behavior."""
    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", MockSlowGenerator())
    pipe.connect("prompt_builder.prompt", "llm.prompt")
    return pipe


@pytest.fixture
def async_pipeline_with_async_capable_generator():
    """
    AsyncPipeline with OpenAIChatGenerator - supports both sync and async streaming callbacks.
    This is a real integration test fixture requiring OPENAI_API_KEY.
    """
    pipe = AsyncPipeline()
    pipe.add_component("prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user(QUESTION)]))
    pipe.add_component("llm", OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini"))
    pipe.connect("prompt_builder.prompt", "llm.messages")
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


@requires_openai_api_key
@pytest.mark.integration
def test_streaming_generator_with_sync_only_generator(pipeline_with_sync_only_generator):
    generator = streaming_generator(pipeline_with_sync_only_generator, pipeline_run_args={})

    assert isinstance(generator, Generator)
    chunks = list(generator)
    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]

    # Real OpenAI response will have multiple chunks
    assert len(streaming_chunks) > 0
    # Content should contain Yes or No as per the question
    full_content = "".join(c.content for c in streaming_chunks)
    assert "Yes" in full_content or "No" in full_content


@requires_openai_api_key
@pytest.mark.integration
async def test_async_streaming_rejects_sync_only_components(async_pipeline_with_sync_only_generator):
    with pytest.raises(ValueError, match="seems to not support async streaming callbacks"):
        async_gen = async_streaming_generator(async_pipeline_with_sync_only_generator, pipeline_run_args={})
        _ = [chunk async for chunk in async_gen]


@requires_openai_api_key
@pytest.mark.integration
async def test_async_streaming_hybrid_mode(async_pipeline_with_sync_only_generator):
    async_gen = async_streaming_generator(
        async_pipeline_with_sync_only_generator,
        pipeline_run_args={},
        allow_sync_streaming_callbacks=True,
    )

    assert isinstance(async_gen, AsyncGenerator)
    chunks = [chunk async for chunk in async_gen]
    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]

    # Real OpenAI response will have multiple chunks
    assert len(streaming_chunks) > 0


@requires_openai_api_key
@pytest.mark.integration
async def test_async_streaming_async_pipeline_emits_chunks(async_pipeline_with_async_capable_generator):
    async_gen = async_streaming_generator(async_pipeline_with_async_capable_generator, pipeline_run_args={})

    chunks = [chunk async for chunk in async_gen]
    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]

    # Real OpenAI response will have multiple chunks
    assert len(streaming_chunks) > 0
    # Content should contain Yes or No as per the question
    full_content = "".join(c.content for c in streaming_chunks)
    assert "Yes" in full_content or "No" in full_content


@requires_openai_api_key
@pytest.mark.integration
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
                received_chunks.append(chunk)  # noqa: PERF401

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


@requires_openai_api_key
@pytest.mark.integration
async def test_async_streaming_cancellation(async_pipeline_with_async_capable_generator):
    async def consume_with_cancel():
        gen = async_streaming_generator(async_pipeline_with_async_capable_generator, pipeline_run_args={})
        chunks_received = []
        async for chunk in gen:
            chunks_received.append(chunk)
            if len(chunks_received) >= 1:
                # Cancel after receiving first chunk
                break
        return chunks_received

    chunks = await consume_with_cancel()
    assert len(chunks) >= 1


# ContextVar used to verify context propagation into the streaming thread
_test_context_var: contextvars.ContextVar[str] = contextvars.ContextVar("_test_context_var")


@component
class MockContextAwareGenerator:
    """Mock generator that reads a ContextVar during execution to verify propagation."""

    @component.output_types(replies=list[str])
    def run(self, prompt: str, streaming_callback: Any | None = None) -> dict[str, Any]:
        value = _test_context_var.get("NOT_SET")
        if streaming_callback:
            streaming_callback(StreamingChunk(content=value, index=0))
        return {"replies": [value]}


def test_streaming_generator_propagates_context_vars():
    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipe.add_component("llm", MockContextAwareGenerator())
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    _test_context_var.set("propagated_value")

    gen = streaming_generator(pipe, pipeline_run_args={})
    chunks = [c for c in gen if isinstance(c, StreamingChunk)]

    assert len(chunks) > 0
    assert chunks[0].content == "propagated_value"


def _make_tool_call_chunk(
    tool_name: str | None = "get_weather",
    arguments: str | None = '{"location": "Berlin"}',
    call_id: str | None = "call_1",
) -> StreamingChunk:
    return StreamingChunk(
        content="",
        index=0,
        tool_calls=[ToolCallDelta(index=0, tool_name=tool_name, arguments=arguments, id=call_id)],
    )


class TestProcessToolCallStart:
    def test_parses_json_arguments_to_dict(self):
        captured: list[tuple] = []

        def callback(name: str, args: dict[str, Any], call_id: str | None) -> None:
            captured.append((name, args, call_id))

        chunk = _make_tool_call_chunk(arguments='{"location": "Berlin"}')
        list(_process_tool_call_start(chunk, callback))

        assert len(captured) == 1
        assert captured[0] == ("get_weather", {"location": "Berlin"}, "call_1")

    def test_none_arguments_become_empty_dict(self):
        captured: list[tuple] = []

        def callback(name: str, args: dict[str, Any], call_id: str | None) -> None:
            captured.append((name, args, call_id))

        chunk = _make_tool_call_chunk(arguments=None)
        list(_process_tool_call_start(chunk, callback))

        assert len(captured) == 1
        assert captured[0] == ("get_weather", {}, "call_1")

    def test_empty_string_arguments_become_empty_dict(self):
        captured: list[tuple] = []

        def callback(name: str, args: dict[str, Any], call_id: str | None) -> None:
            captured.append((name, args, call_id))

        chunk = _make_tool_call_chunk(arguments="")
        list(_process_tool_call_start(chunk, callback))

        assert len(captured) == 1
        assert captured[0] == ("get_weather", {}, "call_1")

    def test_invalid_json_arguments_become_empty_dict(self):
        captured: list[tuple] = []

        def callback(name: str, args: dict[str, Any], call_id: str | None) -> None:
            captured.append((name, args, call_id))

        chunk = _make_tool_call_chunk(arguments='{"location": ')
        list(_process_tool_call_start(chunk, callback))

        assert len(captured) == 1
        assert captured[0] == ("get_weather", {}, "call_1")

    def test_skips_tool_calls_without_tool_name(self):
        captured: list[tuple] = []

        def callback(name: str, args: dict[str, Any], call_id: str | None) -> None:
            captured.append((name, args, call_id))

        chunk = _make_tool_call_chunk(tool_name=None)
        list(_process_tool_call_start(chunk, callback))

        assert len(captured) == 0

    def test_callback_returning_events(self):
        def callback(name: str, args: dict[str, Any], call_id: str | None) -> list[PipelineEvent]:
            return [
                PipelineEvent(type="status", data={"description": f"Calling {name}"}),
            ]

        chunk = _make_tool_call_chunk()
        results = list(_process_tool_call_start(chunk, callback))

        assert len(results) == 1
        assert results[0].type == "status"

    def test_callback_error_is_swallowed(self):
        def callback(name: str, args: dict[str, Any], call_id: str | None) -> None:
            raise ValueError("callback boom")

        chunk = _make_tool_call_chunk()
        results = list(_process_tool_call_start(chunk, callback))
        assert results == []

    def test_no_callback_yields_nothing(self):
        chunk = _make_tool_call_chunk()
        results = list(_process_tool_call_start(chunk, None))
        assert results == []
