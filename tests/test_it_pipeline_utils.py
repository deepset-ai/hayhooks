import os
import pytest
import asyncio
from haystack import Pipeline, AsyncPipeline
from haystack.dataclasses import ChatMessage
from typing import Generator, AsyncGenerator, Dict, Any
from hayhooks.server.pipelines.utils import async_streaming_generator, streaming_generator, find_streaming_component
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

QUESTION = "Is Haystack a framework for developing AI applications? Answer Yes or No"


@pytest.fixture
def sync_pipeline_with_sync_streaming_callback_support():
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component(
        "llm", OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
    )
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


@pytest.fixture
def async_pipeline_with_async_streaming_callback_support():
    """
    NOTE: `OpenAIChatGenerator` supports both _async_ and _sync_ `streaming_callback`.
    """
    pipeline = AsyncPipeline()
    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component(
        "llm", OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
    )
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


@pytest.fixture
def async_pipeline_without_async_streaming_callback_support():
    pipeline = AsyncPipeline()
    pipeline.add_component("prompt_builder", PromptBuilder(template=QUESTION))
    pipeline.add_component("llm", OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini"))
    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    return pipeline


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.integration
def test_sync_pipeline_sync_streaming_callback_streaming_generator(sync_pipeline_with_sync_streaming_callback_support):
    pipeline = sync_pipeline_with_sync_streaming_callback_support

    generator = streaming_generator(
        pipeline,
        {
            "prompt_builder": {"template": [ChatMessage.from_user(QUESTION)]},
        },
    )
    assert isinstance(generator, Generator)

    chunks = [chunk for chunk in generator]
    assert len(chunks) > 0
    assert isinstance(chunks[0], StreamingChunk)
    assert any("Yes" in chunk.content for chunk in chunks)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.integration
async def test_async_pipeline_async_streaming_callback_async_streaming_generator(
    async_pipeline_with_async_streaming_callback_support,
):
    pipeline = async_pipeline_with_async_streaming_callback_support

    # Here streaming_generator will call the .run() method of the AsyncPipeline,
    # which will wrap the call to .run_async() with asyncio.run().
    async_generator = async_streaming_generator(
        pipeline,
        {
            "prompt_builder": {"template": [ChatMessage.from_user(QUESTION)]},
        },
    )
    assert isinstance(async_generator, AsyncGenerator)

    chunks = [chunk async for chunk in async_generator]
    assert len(chunks) > 0
    assert isinstance(chunks[0], StreamingChunk)
    assert any("Yes" in chunk.content for chunk in chunks)


async def test_async_pipeline_without_async_streaming_callback_support_should_raise_exception(
    async_pipeline_without_async_streaming_callback_support,
):
    pipeline = async_pipeline_without_async_streaming_callback_support

    with pytest.raises(ValueError, match="seems to not support async streaming callbacks"):
        async_generator = async_streaming_generator(pipeline, {})

        # Try to consume the generator to trigger the exception
        _ = [chunk async for chunk in async_generator]


class MockComponent:
    """Mock component for testing"""

    def __init__(self, has_streaming=True, has_async_support=True):
        if has_streaming:
            self.streaming_callback = None
        if has_async_support:
            self.run_async = lambda: None


class MockChunk:
    """Mock chunk object with content"""

    def __init__(self, content: str):
        self.content = content

    def __eq__(self, other):
        if isinstance(other, MockChunk):
            return self.content == other.content
        return False

    def __repr__(self):
        return f"MockChunk({self.content!r})"


@pytest.fixture
def mocked_pipeline_with_streaming_component(mocker):
    streaming_component = MockComponent(has_streaming=True)

    pipeline = mocker.Mock(spec=AsyncPipeline)
    pipeline._spec_class = AsyncPipeline
    pipeline.walk.return_value = [("streaming_component", streaming_component)]
    pipeline.get_component.return_value = streaming_component

    return streaming_component, pipeline


def test_find_streaming_component_no_streaming_component():
    pipeline = Pipeline()

    with pytest.raises(ValueError, match="No streaming-capable component found in the pipeline"):
        find_streaming_component(pipeline)


def test_find_streaming_component_finds_streaming_component(mocker):
    streaming_component = MockComponent(has_streaming=True)
    non_streaming_component = MockComponent(has_streaming=False)

    pipeline = mocker.Mock(spec=Pipeline)
    pipeline.walk.return_value = [
        ("component1", non_streaming_component),
        ("streaming_component", streaming_component),
        ("component2", non_streaming_component),
    ]

    component, name = find_streaming_component(pipeline)
    assert component == streaming_component
    assert name == "streaming_component"


def test_streaming_generator_no_streaming_component():
    pipeline = Pipeline()

    with pytest.raises(ValueError, match="No streaming-capable component found in the pipeline"):
        list(streaming_generator(pipeline, {}))


def test_streaming_generator_with_existing_component_args(mocked_pipeline_with_streaming_component):
    streaming_component, pipeline = mocked_pipeline_with_streaming_component

    # Mock the run method to simulate streaming
    def mock_run(data):
        # Simulate calling the streaming callback
        if streaming_component.streaming_callback:
            streaming_component.streaming_callback(MockChunk("chunk1"))
            streaming_component.streaming_callback(MockChunk("chunk2"))

    pipeline.run.side_effect = mock_run

    pipeline_run_args = {"streaming_component": {"existing": "args"}}

    generator = streaming_generator(pipeline, pipeline_run_args)
    chunks = list(generator)

    assert chunks == [MockChunk("chunk1"), MockChunk("chunk2")]
    # Verify original args were preserved and copied
    assert pipeline_run_args == {"streaming_component": {"existing": "args"}}


def test_streaming_generator_pipeline_exception(mocked_pipeline_with_streaming_component):
    _, pipeline = mocked_pipeline_with_streaming_component

    # Mock the run method to raise an exception
    expected_error = RuntimeError("Pipeline execution failed")
    pipeline.run.side_effect = expected_error

    generator = streaming_generator(pipeline, {})

    with pytest.raises(RuntimeError, match="Pipeline execution failed"):
        list(generator)


def test_streaming_generator_empty_output(mocked_pipeline_with_streaming_component):
    _, pipeline = mocked_pipeline_with_streaming_component

    # Mock the run method without calling streaming callback
    pipeline.run.return_value = None

    generator = streaming_generator(pipeline, {})
    chunks = list(generator)

    assert chunks == []


@pytest.mark.asyncio
async def test_async_streaming_generator_no_streaming_component():
    pipeline = Pipeline()

    with pytest.raises(ValueError, match="No streaming-capable component found in the pipeline"):
        _ = [chunk async for chunk in async_streaming_generator(pipeline, {})]


@pytest.mark.asyncio
async def test_async_streaming_generator_with_existing_component_args(mocker, mocked_pipeline_with_streaming_component):
    streaming_component, pipeline = mocked_pipeline_with_streaming_component
    mock_chunks = [MockChunk("async_chunk1"), MockChunk("async_chunk2")]

    # Mock the run_async method to simulate streaming
    async def mock_run_async(data):
        # Simulate calling the streaming callback
        if streaming_component.streaming_callback:
            await streaming_component.streaming_callback(mock_chunks[0])
            await streaming_component.streaming_callback(mock_chunks[1])

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)
    pipeline_run_args = {"streaming_component": {"existing": "args"}}

    chunks = [chunk async for chunk in async_streaming_generator(pipeline, pipeline_run_args)]
    assert chunks == mock_chunks

    # Verify original args were preserved and copied
    assert pipeline_run_args == {"streaming_component": {"existing": "args"}}


@pytest.mark.asyncio
async def test_async_streaming_generator_pipeline_exception(mocker, mocked_pipeline_with_streaming_component):
    _, pipeline = mocked_pipeline_with_streaming_component

    # Mock the run_async method to raise an exception
    expected_error = Exception("Async pipeline execution failed")
    pipeline.run_async = mocker.AsyncMock(side_effect=expected_error)

    with pytest.raises(Exception, match="Async pipeline execution failed"):
        _ = [chunk async for chunk in async_streaming_generator(pipeline, {})]


@pytest.mark.asyncio
async def test_async_streaming_generator_empty_output(mocker, mocked_pipeline_with_streaming_component):
    _, pipeline = mocked_pipeline_with_streaming_component

    # Mock the run_async method without calling streaming callback
    pipeline.run_async = mocker.AsyncMock(return_value=None)

    chunks = [chunk async for chunk in async_streaming_generator(pipeline, {})]

    assert chunks == []


@pytest.mark.asyncio
async def test_async_streaming_generator_cancellation(mocker, mocked_pipeline_with_streaming_component):
    _, pipeline = mocked_pipeline_with_streaming_component

    # Mock the run_async method to simulate long-running task
    async def mock_long_running_task(data):
        await asyncio.sleep(4)  # Simulate long-running task

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_long_running_task)

    async def run_and_cancel():
        generator = async_streaming_generator(pipeline, {})
        task = asyncio.create_task(generator.__anext__())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, StopAsyncIteration):
            # Both are acceptable - CancelledError if cancelled during wait,
            # StopAsyncIteration if generator ends without yielding
            pass
        except Exception as e:
            # Any other exception should cause the test to fail
            raise AssertionError(f"Unexpected exception during cancellation: {e}")

    # Should not raise any unexpected exceptions
    await run_and_cancel()


@pytest.mark.asyncio
async def test_async_streaming_generator_timeout_scenarios(mocker, mocked_pipeline_with_streaming_component):
    streaming_component, pipeline = mocked_pipeline_with_streaming_component
    mock_chunks = [MockChunk("delayed_chunk")]

    # Mock the run_async method to simulate delayed completion
    async def mock_delayed_task(data):
        await asyncio.sleep(0.5)  # Longer than the timeout in the implementation
        if streaming_component.streaming_callback:
            await streaming_component.streaming_callback(mock_chunks[0])

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_delayed_task)

    chunks = [chunk async for chunk in async_streaming_generator(pipeline, {})]

    assert chunks == mock_chunks


def test_streaming_generator_modifies_args_copy(mocked_pipeline_with_streaming_component) -> None:
    _, pipeline = mocked_pipeline_with_streaming_component

    # Mock the run method
    pipeline.run.return_value = None
    original_args = {"other_component": {"param": "value"}}

    # Consume the generator
    generator = streaming_generator(pipeline, original_args)
    list(generator)

    # Original args should be unchanged
    assert original_args == {"other_component": {"param": "value"}}

    # Args passed to pipeline.run should include the streaming component
    pipeline.run.assert_called_once()
    call_kwargs: Dict[str, Any] = pipeline.run.call_args.kwargs
    assert "streaming_component" in call_kwargs["data"]
    assert "other_component" in call_kwargs["data"]


@pytest.mark.asyncio
async def test_async_streaming_generator_modifies_args_copy(mocker, mocked_pipeline_with_streaming_component) -> None:
    _, pipeline = mocked_pipeline_with_streaming_component
    pipeline._spec_class = AsyncPipeline

    # Mock the run_async method
    pipeline.run_async = mocker.AsyncMock(return_value=None)
    original_args = {"other_component": {"param": "value"}}

    # Consume the generator
    async_generator = async_streaming_generator(pipeline, original_args)
    _ = [chunk async for chunk in async_generator]

    # Original args should be unchanged
    assert original_args == {"other_component": {"param": "value"}}

    # Args passed to pipeline.run_async should include the streaming component
    pipeline.run_async.assert_called_once()
    call_kwargs: Dict[str, Any] = pipeline.run_async.call_args.kwargs
    assert "streaming_component" in call_kwargs["data"]
    assert "other_component" in call_kwargs["data"]
