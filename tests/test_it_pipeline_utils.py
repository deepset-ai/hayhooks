import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from haystack import AsyncPipeline, Pipeline
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall, ToolCallDelta, ToolCallResult
from haystack.utils import Secret
from loguru import logger

from hayhooks import callbacks
from hayhooks.open_webui import OpenWebUIEvent, create_notification_event
from hayhooks.server.pipelines.utils import (
    async_streaming_generator,
    find_all_streaming_components,
    streaming_generator,
)
from hayhooks.settings import AppSettings

QUESTION = "Is Haystack a framework for developing AI applications? Answer Yes or No"


@pytest.fixture
def sync_pipeline_with_sync_streaming_callback_support():
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", ChatPromptBuilder(required_variables="*"))
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
    pipeline.add_component("prompt_builder", ChatPromptBuilder(required_variables="*"))
    pipeline.add_component(
        "llm", OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
    )
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


@pytest.fixture
def async_pipeline_without_async_streaming_callback_support():
    pipeline = AsyncPipeline()
    pipeline.add_component("prompt_builder", PromptBuilder(template=QUESTION, required_variables="*"))
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
        pipeline_run_args={
            "prompt_builder": {"template": [ChatMessage.from_user(QUESTION)]},
        },
    )
    assert isinstance(generator, Generator)

    chunks = list(generator)
    assert len(chunks) > 0

    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]
    assert len(streaming_chunks) > 0
    assert isinstance(streaming_chunks[0], StreamingChunk)
    assert any("Yes" in chunk.content for chunk in streaming_chunks)


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
        pipeline_run_args={
            "prompt_builder": {"template": [ChatMessage.from_user(QUESTION)]},
        },
    )
    assert isinstance(async_generator, AsyncGenerator)

    chunks = [chunk async for chunk in async_generator]
    assert len(chunks) > 0

    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]
    assert len(streaming_chunks) > 0
    assert isinstance(streaming_chunks[0], StreamingChunk)
    assert any("Yes" in chunk.content for chunk in streaming_chunks)


async def test_async_pipeline_without_async_streaming_callback_support_should_raise_exception(
    async_pipeline_without_async_streaming_callback_support,
):
    pipeline = async_pipeline_without_async_streaming_callback_support

    with pytest.raises(ValueError, match="seems to not support async streaming callbacks"):
        async_generator = async_streaming_generator(pipeline)

        # Try to consume the generator to trigger the exception
        _ = [chunk async for chunk in async_generator]


class MockComponent:
    """Mock component for testing"""

    def __init__(self, has_streaming=True, has_async_support=True):
        if has_streaming:

            def run_with_streaming(streaming_callback=None):
                pass

            self.run = run_with_streaming
        else:

            def run_without_streaming():
                pass

            self.run = run_without_streaming

        if has_async_support:
            self.run_async = lambda: None


@pytest.fixture
def mocked_pipeline_with_streaming_component(mocker):
    streaming_component = MockComponent(has_streaming=True)

    pipeline = mocker.Mock(spec=AsyncPipeline)
    pipeline._spec_class = AsyncPipeline
    pipeline.walk.return_value = [("streaming_component", streaming_component)]
    pipeline.get_component.return_value = streaming_component

    return pipeline


def test_streaming_generator_no_streaming_component():
    pipeline = Pipeline()

    with pytest.raises(ValueError, match="No streaming-capable components found in the pipeline"):
        list(streaming_generator(pipeline))


def test_streaming_generator_with_existing_component_args(mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    # Mock the run method to simulate streaming
    def mock_run(data):
        # Simulate calling the streaming callback
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            callback(StreamingChunk(content="chunk1"))
            callback(StreamingChunk(content="chunk2"))

    pipeline.run.side_effect = mock_run

    pipeline_run_args = {"streaming_component": {"existing": "args"}}

    generator = streaming_generator(pipeline, pipeline_run_args=pipeline_run_args)
    chunks = list(generator)

    assert chunks == [StreamingChunk(content="chunk1"), StreamingChunk(content="chunk2")]
    # Verify original args were preserved and copied
    assert pipeline_run_args == {"streaming_component": {"existing": "args"}}


def test_streaming_generator_pipeline_exception(mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    # Mock the run method to raise an exception
    expected_error = RuntimeError("Pipeline execution failed")
    pipeline.run.side_effect = expected_error

    generator = streaming_generator(pipeline)

    with pytest.raises(RuntimeError, match="Pipeline execution failed"):
        list(generator)


def test_streaming_generator_empty_output(mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    generator = streaming_generator(pipeline)
    chunks = list(generator)

    assert chunks == []


@pytest.mark.asyncio
async def test_async_streaming_generator_no_streaming_component():
    pipeline = Pipeline()

    with pytest.raises(ValueError, match="No streaming-capable components found in the pipeline"):
        _ = [chunk async for chunk in async_streaming_generator(pipeline)]


@pytest.mark.asyncio
async def test_async_streaming_generator_with_existing_component_args(mocker, mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component
    mock_chunks = [StreamingChunk(content="async_chunk1"), StreamingChunk(content="async_chunk2")]

    # Mock the run_async method to simulate streaming
    async def mock_run_async(data):
        # Simulate calling the streaming callback
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            await callback(mock_chunks[0])
            await callback(mock_chunks[1])

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)
    pipeline_run_args = {"streaming_component": {"existing": "args"}}

    chunks = [chunk async for chunk in async_streaming_generator(pipeline, pipeline_run_args=pipeline_run_args)]
    assert chunks == mock_chunks

    # Verify original args were preserved and copied
    assert pipeline_run_args == {"streaming_component": {"existing": "args"}}


@pytest.mark.asyncio
async def test_async_streaming_generator_pipeline_exception(mocker, mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    # Mock the run_async method to raise an exception
    expected_error = Exception("Async pipeline execution failed")
    pipeline.run_async = mocker.AsyncMock(side_effect=expected_error)

    with pytest.raises(Exception, match="Async pipeline execution failed"):
        _ = [chunk async for chunk in async_streaming_generator(pipeline)]


@pytest.mark.asyncio
async def test_async_streaming_generator_empty_output(mocker, mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    # Mock the run_async method without calling streaming callback
    pipeline.run_async = mocker.AsyncMock(return_value=None)

    chunks = [chunk async for chunk in async_streaming_generator(pipeline)]

    assert chunks == []


@pytest.mark.asyncio
async def test_async_streaming_generator_cancellation(mocker, mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    # Mock the run_async method to simulate long-running task
    async def mock_long_running_task(data):
        await asyncio.sleep(4)  # Simulate long-running task

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_long_running_task)

    async def run_and_cancel():
        generator = async_streaming_generator(pipeline)
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
            msg = f"Unexpected exception during cancellation: {e}"
            raise AssertionError(msg) from e

    # Should not raise any unexpected exceptions
    await run_and_cancel()


@pytest.mark.asyncio
async def test_async_streaming_generator_timeout_scenarios(mocker, mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component
    mock_chunks = [StreamingChunk(content="delayed_chunk")]

    # Mock the run_async method to simulate delayed completion
    async def mock_delayed_task(data):
        await asyncio.sleep(0.5)  # Longer than the timeout in the implementation
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            await callback(mock_chunks[0])

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_delayed_task)

    chunks = [chunk async for chunk in async_streaming_generator(pipeline)]

    assert chunks == mock_chunks


def test_streaming_generator_modifies_args_copy(mocked_pipeline_with_streaming_component) -> None:
    pipeline = mocked_pipeline_with_streaming_component

    # Mock the run method
    pipeline.run.return_value = None
    original_args = {"other_component": {"param": "value"}}

    # Consume the generator
    generator = streaming_generator(pipeline, pipeline_run_args=original_args)
    list(generator)

    # Original args should be unchanged
    assert original_args == {"other_component": {"param": "value"}}

    # Args passed to pipeline.run should include the streaming component
    pipeline.run.assert_called_once()
    call_kwargs: dict[str, Any] = pipeline.run.call_args.kwargs
    assert "streaming_component" in call_kwargs["data"]
    assert "other_component" in call_kwargs["data"]


@pytest.mark.asyncio
async def test_async_streaming_generator_modifies_args_copy(mocker, mocked_pipeline_with_streaming_component) -> None:
    pipeline = mocked_pipeline_with_streaming_component
    pipeline._spec_class = AsyncPipeline

    # Mock the run_async method
    pipeline.run_async = mocker.AsyncMock(return_value=None)
    original_args = {"other_component": {"param": "value"}}

    # Consume the generator
    async_generator = async_streaming_generator(pipeline, pipeline_run_args=original_args)
    _ = [chunk async for chunk in async_generator]

    # Original args should be unchanged
    assert original_args == {"other_component": {"param": "value"}}

    # Args passed to pipeline.run_async should include the streaming component
    pipeline.run_async.assert_called_once()
    call_kwargs: dict[str, Any] = pipeline.run_async.call_args.kwargs
    assert "streaming_component" in call_kwargs["data"]
    assert "other_component" in call_kwargs["data"]


def test_streaming_generator_with_tool_calls_and_default_callbacks(mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    tool_call_start = ToolCallDelta(index=0, tool_name="test_tool", arguments="")
    tool_call_end = ToolCallResult(
        origin=ToolCall(tool_name="test_tool", arguments={"city": "Berlin"}), result="sunny", error=None
    )
    mock_chunks_from_pipeline = [
        StreamingChunk(content="", index=0, tool_calls=[tool_call_start]),
        StreamingChunk(content=" some content "),
        StreamingChunk(content="", index=0, tool_call_result=tool_call_end),
    ]

    def mock_run(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                callback(chunk)
        return {"result": "Final result"}

    pipeline.run.side_effect = mock_run

    generator = streaming_generator(
        pipeline,
        on_tool_call_start=callbacks.default_on_tool_call_start,
        on_tool_call_end=callbacks.default_on_tool_call_end,
        on_pipeline_end=callbacks.default_on_pipeline_end,
    )
    chunks = list(generator)

    assert len(chunks) == 7
    assert isinstance(chunks[0], OpenWebUIEvent)
    assert chunks[0].type == "status"
    assert "Calling 'test_tool' tool..." in chunks[0].data.description
    assert chunks[1] == mock_chunks_from_pipeline[0]

    assert chunks[2] == mock_chunks_from_pipeline[1]

    assert isinstance(chunks[3], OpenWebUIEvent)
    assert chunks[3].type == "status"
    assert "Called 'test_tool' tool" in chunks[3].data.description
    assert isinstance(chunks[4], str)
    assert "Tool call result for 'test_tool'" in chunks[4]
    assert chunks[5] == mock_chunks_from_pipeline[2]


def test_streaming_generator_with_custom_callbacks(mocker, mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    tool_call_start = ToolCallDelta(index=0, tool_name="test_tool", arguments="")
    tool_call_end = ToolCallResult(
        origin=ToolCall(tool_name="test_tool", arguments={"city": "Berlin"}), result="sunny", error=None
    )
    mock_chunks_from_pipeline = [
        StreamingChunk(content="", index=0, tool_calls=[tool_call_start]),
        StreamingChunk(content="", index=0, tool_call_result=tool_call_end),
    ]

    def mock_run(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                callback(chunk)
        return {"result": "Final result"}

    pipeline.run.side_effect = mock_run

    # Use real functions and spy on them
    on_tool_call_start_spy = mocker.spy(callbacks, "default_on_tool_call_start")
    on_tool_call_end_spy = mocker.spy(callbacks, "default_on_tool_call_end")
    on_pipeline_end_spy = mocker.spy(callbacks, "default_on_pipeline_end")

    generator = streaming_generator(
        pipeline,
        on_tool_call_start=on_tool_call_start_spy,
        on_tool_call_end=on_tool_call_end_spy,
        on_pipeline_end=on_pipeline_end_spy,
    )
    chunks = list(generator)

    # Verify the chunks contain the expected results from the real callbacks
    assert len(chunks) == 6

    # First chunk should be the result from default_on_tool_call_start
    assert isinstance(chunks[0], OpenWebUIEvent)
    assert chunks[0].type == "status"
    assert "Calling 'test_tool' tool..." in chunks[0].data.description

    # Second chunk is the original streaming chunk
    assert chunks[1] == mock_chunks_from_pipeline[0]

    # Third chunk should be the status event from default_on_tool_call_end
    assert isinstance(chunks[2], OpenWebUIEvent)
    assert chunks[2].type == "status"
    assert "Called 'test_tool' tool" in chunks[2].data.description

    # Fourth chunk should be the details tag from default_on_tool_call_end
    assert isinstance(chunks[3], str)
    assert "Tool call result for 'test_tool'" in chunks[3]

    # Fifth chunk is the original streaming chunk
    assert chunks[4] == mock_chunks_from_pipeline[1]

    # Sixth chunk is the final result from pipeline.run
    assert chunks[5] == StreamingChunk(content='{"result": "Final result"}')

    # Verify the spies were called correctly
    on_tool_call_start_spy.assert_called_once_with("test_tool", "", None)
    on_tool_call_end_spy.assert_called_once_with("test_tool", {"city": "Berlin"}, "sunny", False)
    on_pipeline_end_spy.assert_called_once_with({"result": "Final result"})


@pytest.mark.asyncio
async def test_async_streaming_generator_with_tool_calls_and_default_callbacks(
    mocker, mocked_pipeline_with_streaming_component
):
    pipeline = mocked_pipeline_with_streaming_component

    tool_call_start = ToolCallDelta(index=0, tool_name="test_tool", arguments="")
    tool_call_end = ToolCallResult(
        origin=ToolCall(tool_name="test_tool", arguments={"city": "Berlin"}), result="sunny", error=None
    )
    mock_chunks_from_pipeline = [
        StreamingChunk(content="", index=0, tool_calls=[tool_call_start]),
        StreamingChunk(content=" some content "),
        StreamingChunk(content="", index=0, tool_call_result=tool_call_end),
    ]

    async def mock_run_async(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                await callback(chunk)
        return {"result": "Final result"}

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    generator = async_streaming_generator(
        pipeline,
        on_tool_call_start=callbacks.default_on_tool_call_start,
        on_tool_call_end=callbacks.default_on_tool_call_end,
        on_pipeline_end=callbacks.default_on_pipeline_end,
    )
    chunks = [chunk async for chunk in generator]

    assert len(chunks) == 7

    assert isinstance(chunks[0], OpenWebUIEvent)
    assert chunks[0].type == "status"
    assert "Calling 'test_tool' tool..." in chunks[0].data.description

    assert chunks[1] == mock_chunks_from_pipeline[0]
    assert chunks[2] == mock_chunks_from_pipeline[1]

    assert isinstance(chunks[3], OpenWebUIEvent)
    assert chunks[3].type == "status"
    assert "Called 'test_tool' tool" in chunks[3].data.description
    assert isinstance(chunks[4], str)
    assert "Tool call result for 'test_tool'" in chunks[4]
    assert chunks[5] == mock_chunks_from_pipeline[2]
    # Final result chunk
    assert chunks[6] == StreamingChunk(content='{"result": "Final result"}')


@pytest.mark.asyncio
async def test_async_streaming_generator_with_custom_callbacks(mocker, mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    tool_call_start = ToolCallDelta(index=0, tool_name="test_tool", arguments="")
    tool_call_end = ToolCallResult(
        origin=ToolCall(tool_name="test_tool", arguments={"city": "Berlin"}), result="sunny", error=None
    )
    mock_chunks_from_pipeline = [
        StreamingChunk(content="", index=0, tool_calls=[tool_call_start]),
        StreamingChunk(content="", index=0, tool_call_result=tool_call_end),
    ]

    async def mock_run_async(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                await callback(chunk)
        return {"result": "Final result"}

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    # Use real functions and spy on them
    on_tool_call_start_spy = mocker.spy(callbacks, "default_on_tool_call_start")
    on_tool_call_end_spy = mocker.spy(callbacks, "default_on_tool_call_end")
    on_pipeline_end_spy = mocker.spy(callbacks, "default_on_pipeline_end")

    generator = async_streaming_generator(
        pipeline,
        on_tool_call_start=on_tool_call_start_spy,
        on_tool_call_end=on_tool_call_end_spy,
        on_pipeline_end=on_pipeline_end_spy,
    )
    chunks = [chunk async for chunk in generator]

    # Verify the chunks contain the expected results from the real callbacks
    assert len(chunks) == 6

    # First chunk should be the result from default_on_tool_call_start
    assert isinstance(chunks[0], OpenWebUIEvent)
    assert chunks[0].type == "status"
    assert "Calling 'test_tool' tool..." in chunks[0].data.description

    # Second chunk is the original streaming chunk
    assert chunks[1] == mock_chunks_from_pipeline[0]

    # Third chunk should be the status event from default_on_tool_call_end
    assert isinstance(chunks[2], OpenWebUIEvent)
    assert chunks[2].type == "status"
    assert "Called 'test_tool' tool" in chunks[2].data.description

    # Fourth chunk should be the details tag from default_on_tool_call_end
    assert isinstance(chunks[3], str)
    assert "Tool call result for 'test_tool'" in chunks[3]

    # Fifth chunk is the original streaming chunk
    assert chunks[4] == mock_chunks_from_pipeline[1]

    # Sixth chunk is the final result from pipeline.run
    assert chunks[5] == StreamingChunk(content='{"result": "Final result"}')

    # Verify the spies were called correctly
    on_tool_call_start_spy.assert_called_once_with("test_tool", "", None)
    on_tool_call_end_spy.assert_called_once_with("test_tool", {"city": "Berlin"}, "sunny", False)
    on_pipeline_end_spy.assert_called_once_with({"result": "Final result"})


def test_streaming_generator_with_custom_callbacks_returning_list(mocker, mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    tool_call_start = ToolCallDelta(index=0, tool_name="test_tool", arguments="")
    tool_call_end = ToolCallResult(
        origin=ToolCall(tool_name="test_tool", arguments={"city": "Berlin"}), result="sunny", error=None
    )
    mock_chunks_from_pipeline = [
        StreamingChunk(content="", index=0, tool_calls=[tool_call_start]),
        StreamingChunk(content="", index=0, tool_call_result=tool_call_end),
    ]

    def mock_run(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                callback(chunk)

    pipeline.run.side_effect = mock_run

    # Create custom callbacks that return lists
    def custom_on_tool_call_start(tool_name, arguments, tool_call_id):
        return [
            create_notification_event(content=f"Calling '{tool_name}' tool... (1/2)", notification_type="info"),
            create_notification_event(content=f"Calling '{tool_name}' tool... (2/2)", notification_type="info"),
        ]

    def custom_on_tool_call_end(tool_name, arguments, result, error):
        return [
            create_notification_event(content=f"Called '{tool_name}' tool (1/2)", notification_type="success"),
            create_notification_event(content=f"Called '{tool_name}' tool (2/2)", notification_type="success"),
        ]

    # Use real functions and spy on them
    on_tool_call_start_spy = mocker.Mock(side_effect=custom_on_tool_call_start)
    on_tool_call_end_spy = mocker.Mock(side_effect=custom_on_tool_call_end)

    generator = streaming_generator(
        pipeline, on_tool_call_start=on_tool_call_start_spy, on_tool_call_end=on_tool_call_end_spy
    )
    chunks = list(generator)

    # Verify the chunks contain the expected results from the real callbacks
    assert len(chunks) == 6

    # First two chunks should be the results from custom_on_tool_call_start
    assert isinstance(chunks[0], OpenWebUIEvent)
    assert chunks[0].type == "notification"
    assert "Calling 'test_tool' tool... (1/2)" in chunks[0].data.content

    assert isinstance(chunks[1], OpenWebUIEvent)
    assert chunks[1].type == "notification"
    assert "Calling 'test_tool' tool... (2/2)" in chunks[1].data.content

    # Third chunk is the original streaming chunk
    assert chunks[2] == mock_chunks_from_pipeline[0]

    # Fourth and fifth chunks should be the results from custom_on_tool_call_end
    assert isinstance(chunks[3], OpenWebUIEvent)
    assert chunks[3].type == "notification"
    assert "Called 'test_tool' tool (1/2)" in chunks[3].data.content

    assert isinstance(chunks[4], OpenWebUIEvent)
    assert chunks[4].type == "notification"
    assert "Called 'test_tool' tool (2/2)" in chunks[4].data.content

    # Sixth chunk is the original streaming chunk
    assert chunks[5] == mock_chunks_from_pipeline[1]

    # Verify the spies were called correctly
    on_tool_call_start_spy.assert_called_once_with("test_tool", "", None)
    on_tool_call_end_spy.assert_called_once_with("test_tool", {"city": "Berlin"}, "sunny", False)


@pytest.mark.asyncio
async def test_async_streaming_generator_with_custom_callbacks_returning_list(
    mocker, mocked_pipeline_with_streaming_component
):
    pipeline = mocked_pipeline_with_streaming_component

    tool_call_start = ToolCallDelta(index=0, tool_name="test_tool", arguments="")
    tool_call_end = ToolCallResult(
        origin=ToolCall(tool_name="test_tool", arguments={"city": "Berlin"}), result="sunny", error=None
    )
    mock_chunks_from_pipeline = [
        StreamingChunk(content="", index=0, tool_calls=[tool_call_start]),
        StreamingChunk(content="", index=0, tool_call_result=tool_call_end),
    ]

    async def mock_run_async(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                await callback(chunk)

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    # Create custom callbacks that return lists
    def custom_on_tool_call_start(tool_name, arguments, tool_call_id):
        return [
            create_notification_event(content=f"Calling '{tool_name}' tool... (1/2)", notification_type="info"),
            create_notification_event(content=f"Calling '{tool_name}' tool... (2/2)", notification_type="info"),
        ]

    def custom_on_tool_call_end(tool_name, arguments, result, error):
        return [
            create_notification_event(content=f"Called '{tool_name}' tool (1/2)", notification_type="success"),
            create_notification_event(content=f"Called '{tool_name}' tool (2/2)", notification_type="success"),
        ]

    # Use real functions and spy on them
    on_tool_call_start_spy = mocker.Mock(side_effect=custom_on_tool_call_start)
    on_tool_call_end_spy = mocker.Mock(side_effect=custom_on_tool_call_end)

    generator = async_streaming_generator(
        pipeline, on_tool_call_start=on_tool_call_start_spy, on_tool_call_end=on_tool_call_end_spy
    )
    chunks = [chunk async for chunk in generator]

    # Verify the chunks contain the expected results from the real callbacks
    assert len(chunks) == 6

    # First two chunks should be the results from custom_on_tool_call_start
    assert isinstance(chunks[0], OpenWebUIEvent)
    assert chunks[0].type == "notification"
    assert "Calling 'test_tool' tool... (1/2)" in chunks[0].data.content

    assert isinstance(chunks[1], OpenWebUIEvent)
    assert chunks[1].type == "notification"
    assert "Calling 'test_tool' tool... (2/2)" in chunks[1].data.content

    # Third chunk is the original streaming chunk
    assert chunks[2] == mock_chunks_from_pipeline[0]

    # Fourth and fifth chunks should be the results from custom_on_tool_call_end
    assert isinstance(chunks[3], OpenWebUIEvent)
    assert chunks[3].type == "notification"
    assert "Called 'test_tool' tool (1/2)" in chunks[3].data.content

    assert isinstance(chunks[4], OpenWebUIEvent)
    assert chunks[4].type == "notification"
    assert "Called 'test_tool' tool (2/2)" in chunks[4].data.content

    # Sixth chunk is the original streaming chunk
    assert chunks[5] == mock_chunks_from_pipeline[1]

    # Verify the spies were called correctly
    on_tool_call_start_spy.assert_called_once_with("test_tool", "", None)
    on_tool_call_end_spy.assert_called_once_with("test_tool", {"city": "Berlin"}, "sunny", False)


def test_streaming_generator_with_tool_calls_and_no_callbacks(mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    tool_call_start = ToolCallDelta(index=0, tool_name="test_tool", arguments="")
    tool_call_end = ToolCallResult(
        origin=ToolCall(tool_name="test_tool", arguments={"city": "Berlin"}), result="sunny", error=None
    )
    mock_chunks_from_pipeline = [
        StreamingChunk(content="", index=0, tool_calls=[tool_call_start]),
        StreamingChunk(content=" some content "),
        StreamingChunk(content="", index=0, tool_call_result=tool_call_end),
    ]

    def mock_run(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                callback(chunk)

    pipeline.run.side_effect = mock_run

    generator = streaming_generator(pipeline)
    chunks = list(generator)

    assert chunks == mock_chunks_from_pipeline


@pytest.mark.asyncio
async def test_async_streaming_generator_with_tool_calls_and_no_callbacks(
    mocker, mocked_pipeline_with_streaming_component
):
    pipeline = mocked_pipeline_with_streaming_component

    tool_call_start = ToolCallDelta(index=0, tool_name="test_tool", arguments="")
    tool_call_end = ToolCallResult(
        origin=ToolCall(tool_name="test_tool", arguments={"city": "Berlin"}), result="sunny", error=None
    )
    mock_chunks_from_pipeline = [
        StreamingChunk(content="", index=0, tool_calls=[tool_call_start]),
        StreamingChunk(content=" some content "),
        StreamingChunk(content="", index=0, tool_call_result=tool_call_end),
    ]

    async def mock_run_async(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                await callback(chunk)

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    generator = async_streaming_generator(pipeline)
    chunks = [chunk async for chunk in generator]

    assert chunks == mock_chunks_from_pipeline


def test_sync_streaming_generator_on_pipeline_end_callback(mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    mock_chunks_from_pipeline = [
        StreamingChunk(content="Chunk 1", index=0),
        StreamingChunk(content="Chunk 2", index=0),
    ]

    def mock_run(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                callback(chunk)
        return {"result": "Final result"}

    pipeline.run.side_effect = mock_run

    generator = streaming_generator(pipeline, on_pipeline_end=callbacks.default_on_pipeline_end)
    chunks = list(generator)

    assert len(chunks) == 3
    assert chunks[0] == mock_chunks_from_pipeline[0]
    assert chunks[1] == mock_chunks_from_pipeline[1]
    assert chunks[2] == StreamingChunk(content='{"result": "Final result"}')


@pytest.mark.asyncio
async def test_async_streaming_generator_on_pipeline_end_callback(mocker, mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    mock_chunks_from_pipeline = [
        StreamingChunk(content="Chunk 1", index=0),
        StreamingChunk(content="Chunk 2", index=0),
    ]

    async def mock_run_async(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                await callback(chunk)
        return {"result": "Final result"}

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    generator = async_streaming_generator(pipeline, on_pipeline_end=callbacks.default_on_pipeline_end)
    chunks = [chunk async for chunk in generator]

    assert len(chunks) == 3
    assert chunks[0] == mock_chunks_from_pipeline[0]
    assert chunks[1] == mock_chunks_from_pipeline[1]
    assert chunks[2] == StreamingChunk(content='{"result": "Final result"}')


def test_sync_streaming_generator_on_pipeline_end_callback_no_return(mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    mock_chunks_from_pipeline = [
        StreamingChunk(content="Chunk 1", index=0),
        StreamingChunk(content="Chunk 2", index=0),
    ]

    def mock_run(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                callback(chunk)
        return {"result": "Final result"}

    pipeline.run.side_effect = mock_run

    # Custom callback that returns None
    def custom_on_pipeline_end(output):
        pass

    generator = streaming_generator(pipeline, on_pipeline_end=custom_on_pipeline_end)
    chunks = list(generator)

    assert len(chunks) == 2
    assert chunks[0] == mock_chunks_from_pipeline[0]
    assert chunks[1] == mock_chunks_from_pipeline[1]


@pytest.mark.asyncio
async def test_async_streaming_generator_on_pipeline_end_callback_no_return(
    mocker, mocked_pipeline_with_streaming_component
):
    pipeline = mocked_pipeline_with_streaming_component

    mock_chunks_from_pipeline = [
        StreamingChunk(content="Chunk 1", index=0),
        StreamingChunk(content="Chunk 2", index=0),
    ]

    async def mock_run_async(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                await callback(chunk)
        return {"result": "Final result"}

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    # Custom callback that returns None
    def custom_on_pipeline_end(output):
        pass

    generator = async_streaming_generator(pipeline, on_pipeline_end=custom_on_pipeline_end)
    chunks = [chunk async for chunk in generator]

    assert len(chunks) == 2
    assert chunks[0] == mock_chunks_from_pipeline[0]
    assert chunks[1] == mock_chunks_from_pipeline[1]


def test_sync_streaming_generator_on_pipeline_end_callback_raises(mocked_pipeline_with_streaming_component):
    pipeline = mocked_pipeline_with_streaming_component

    mock_chunks_from_pipeline = [
        StreamingChunk(content="Chunk 1", index=0),
        StreamingChunk(content="Chunk 2", index=0),
    ]

    def mock_run(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                callback(chunk)
        return {"result": "Final result"}

    pipeline.run.side_effect = mock_run

    # Custom callback that raises an exception
    def custom_on_pipeline_end(output):
        msg = "Callback error"
        raise ValueError(msg)

    generator = streaming_generator(pipeline, on_pipeline_end=custom_on_pipeline_end)

    messages = []
    logger.add(lambda msg: messages.append(msg), level="ERROR")
    _ = list(generator)
    assert "Error in on_pipeline_end callback" in messages[0]


@pytest.mark.asyncio
async def test_async_streaming_generator_on_pipeline_end_callback_raises(
    mocker, mocked_pipeline_with_streaming_component
):
    pipeline = mocked_pipeline_with_streaming_component

    mock_chunks_from_pipeline = [
        StreamingChunk(content="Chunk 1", index=0),
        StreamingChunk(content="Chunk 2", index=0),
    ]

    async def mock_run_async(data):
        callback = data.get("streaming_component", {}).get("streaming_callback")
        if callback:
            for chunk in mock_chunks_from_pipeline:
                await callback(chunk)
        return {"result": "Final result"}

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    # Custom callback that raises an exception
    def custom_on_pipeline_end(output):
        msg = "Callback error"
        raise ValueError(msg)

    generator = async_streaming_generator(pipeline, on_pipeline_end=custom_on_pipeline_end)

    messages = []
    logger.add(lambda msg: messages.append(msg), level="ERROR")
    _ = [chunk async for chunk in generator]
    assert "Error in on_pipeline_end callback" in messages[0]


def test_find_all_streaming_components_finds_multiple(mocker):
    streaming_component1 = MockComponent(has_streaming=True)
    streaming_component2 = MockComponent(has_streaming=True)
    non_streaming_component = MockComponent(has_streaming=False)

    pipeline = mocker.Mock(spec=Pipeline)
    pipeline.walk.return_value = [
        ("component1", streaming_component1),
        ("non_streaming", non_streaming_component),
        ("component2", streaming_component2),
    ]

    components = find_all_streaming_components(pipeline)
    assert len(components) == 2
    assert components[0] == (streaming_component1, "component1")
    assert components[1] == (streaming_component2, "component2")


def test_find_all_streaming_components_raises_when_none_found():
    pipeline = Pipeline()

    with pytest.raises(ValueError, match="No streaming-capable components found in the pipeline"):
        find_all_streaming_components(pipeline)


def test_find_all_streaming_components_ignores_attribute_only(mocker):
    """
    Given a component with streaming_callback as attribute but not in run()/run_async() method signatures
    The component should be ignored by find_all_streaming_components(), because it doesn't have
    streaming_callback in run/run_async method signatures, and should raise ValueError.

    The main reason is that we want to ensure that all streaming components support async streaming callbacks.
    If a component has streaming_callback as attribute but not in run/run_async method signatures,
    it means that the component is not streaming capable, and should be ignored.
    """

    class ComponentWithAttributeOnly:
        """Component with streaming_callback as attribute but not in run() signature."""

        def __init__(self):
            self.streaming_callback = None  # Attribute only

        def run(self):
            """Run method without streaming_callback parameter."""
            pass

    component_with_attr = ComponentWithAttributeOnly()

    pipeline = mocker.Mock(spec=Pipeline)
    pipeline.walk.return_value = [
        ("component_with_attribute", component_with_attr),
    ]

    # Should raise ValueError because the component doesn't have streaming_callback in method signature
    with pytest.raises(ValueError, match="No streaming-capable components found in the pipeline"):
        find_all_streaming_components(pipeline)


@pytest.fixture
def pipeline_with_multiple_streaming_components(mocker):
    streaming_component1 = MockComponent(has_streaming=True)
    streaming_component2 = MockComponent(has_streaming=True)
    non_streaming_component = MockComponent(has_streaming=False)

    pipeline = mocker.Mock(spec=AsyncPipeline)
    pipeline._spec_class = AsyncPipeline
    pipeline.walk.return_value = [
        ("component1", streaming_component1),
        ("non_streaming", non_streaming_component),
        ("component2", streaming_component2),
    ]

    def mock_get_component(name):
        if name == "component1":
            return streaming_component1
        elif name == "component2":
            return streaming_component2
        return non_streaming_component

    pipeline.get_component.side_effect = mock_get_component

    return pipeline


def test_streaming_generator_with_multiple_components_default_behavior(pipeline_with_multiple_streaming_components):
    pipeline = pipeline_with_multiple_streaming_components

    mock_chunks = [
        StreamingChunk(content="chunk1_from_component2"),
        StreamingChunk(content="chunk2_from_component2"),
    ]

    def mock_run(data):
        # Only component2 should stream (it's the last one)
        callback = data.get("component2", {}).get("streaming_callback")
        if callback:
            callback(mock_chunks[0])
            callback(mock_chunks[1])

    pipeline.run.side_effect = mock_run

    generator = streaming_generator(pipeline)
    chunks = list(generator)

    assert chunks == mock_chunks
    # Verify the callback was passed in the pipeline run args
    # (we can't easily verify this in the test after the fact, but the mock_run will fail if it's not there)


@pytest.mark.parametrize(
    "streaming_components",
    [
        ["component1", "component2"],  # Explicit list
        "all",  # "all" keyword
    ],
    ids=["list_both", "all_keyword"],
)
def test_streaming_generator_with_all_components_enabled(
    pipeline_with_multiple_streaming_components, streaming_components
):
    pipeline = pipeline_with_multiple_streaming_components

    mock_chunks = [
        StreamingChunk(content="chunk1_from_component1"),
        StreamingChunk(content="chunk2_from_component1"),
        StreamingChunk(content="chunk1_from_component2"),
        StreamingChunk(content="chunk2_from_component2"),
    ]

    def mock_run(data):
        callback1 = data.get("component1", {}).get("streaming_callback")
        if callback1:
            callback1(mock_chunks[0])
            callback1(mock_chunks[1])
        callback2 = data.get("component2", {}).get("streaming_callback")
        if callback2:
            callback2(mock_chunks[2])
            callback2(mock_chunks[3])

    pipeline.run.side_effect = mock_run

    generator = streaming_generator(pipeline, streaming_components=streaming_components)
    chunks = list(generator)

    assert chunks == mock_chunks
    # Verify the callbacks were passed in the pipeline run args
    # (we can't easily verify this in the test after the fact, but the mock_run will fail if they're not there)


def test_streaming_generator_with_multiple_components_selective(pipeline_with_multiple_streaming_components):
    pipeline = pipeline_with_multiple_streaming_components

    mock_chunks = [
        StreamingChunk(content="chunk1_from_component1"),
        StreamingChunk(content="chunk2_from_component1"),
    ]

    def mock_run(data):
        # Only component1 should stream based on config
        callback = data.get("component1", {}).get("streaming_callback")
        if callback:
            callback(mock_chunks[0])
            callback(mock_chunks[1])

    pipeline.run.side_effect = mock_run

    generator = streaming_generator(pipeline, streaming_components=["component1"])
    chunks = list(generator)

    assert chunks == mock_chunks


@pytest.mark.asyncio
async def test_async_streaming_generator_with_multiple_components_default_behavior(
    mocker, pipeline_with_multiple_streaming_components
):
    pipeline = pipeline_with_multiple_streaming_components

    mock_chunks = [
        StreamingChunk(content="async_chunk1_from_component2"),
        StreamingChunk(content="async_chunk2_from_component2"),
    ]

    async def mock_run_async(data):
        # Only component2 should stream (it's the last one)
        callback = data.get("component2", {}).get("streaming_callback")
        if callback:
            await callback(mock_chunks[0])
            await callback(mock_chunks[1])

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    chunks = [chunk async for chunk in async_streaming_generator(pipeline)]

    assert chunks == mock_chunks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "streaming_components",
    [
        ["component1", "component2"],  # Explicit list
        "all",  # "all" keyword
    ],
    ids=["list_both", "all_keyword"],
)
async def test_async_streaming_generator_with_all_components_enabled(
    mocker, pipeline_with_multiple_streaming_components, streaming_components
):
    pipeline = pipeline_with_multiple_streaming_components

    mock_chunks = [
        StreamingChunk(content="async_chunk1_from_component1"),
        StreamingChunk(content="async_chunk2_from_component1"),
        StreamingChunk(content="async_chunk1_from_component2"),
        StreamingChunk(content="async_chunk2_from_component2"),
    ]

    async def mock_run_async(data):
        # Both components should stream
        callback1 = data.get("component1", {}).get("streaming_callback")
        if callback1:
            await callback1(mock_chunks[0])
            await callback1(mock_chunks[1])
        callback2 = data.get("component2", {}).get("streaming_callback")
        if callback2:
            await callback2(mock_chunks[2])
            await callback2(mock_chunks[3])

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    chunks = [chunk async for chunk in async_streaming_generator(pipeline, streaming_components=streaming_components)]

    assert chunks == mock_chunks


@pytest.mark.asyncio
async def test_async_streaming_generator_with_multiple_components_selective(
    mocker, pipeline_with_multiple_streaming_components
):
    pipeline = pipeline_with_multiple_streaming_components

    mock_chunks = [
        StreamingChunk(content="async_chunk1_from_component1"),
        StreamingChunk(content="async_chunk2_from_component1"),
    ]

    async def mock_run_async(data):
        # Only component1 should stream based on config
        callback = data.get("component1", {}).get("streaming_callback")
        if callback:
            await callback(mock_chunks[0])
            await callback(mock_chunks[1])

    pipeline.run_async = mocker.AsyncMock(side_effect=mock_run_async)

    chunks = [chunk async for chunk in async_streaming_generator(pipeline, streaming_components=["component1"])]

    assert chunks == mock_chunks


@pytest.mark.parametrize(
    "env_var_value",
    [
        "all",  # "all" keyword
        "component1,component2",  # Comma-separated list
    ],
    ids=["env_all_keyword", "env_comma_separated"],
)
def test_streaming_generator_with_env_var_all_components(
    monkeypatch, pipeline_with_multiple_streaming_components, env_var_value
):
    pipeline = pipeline_with_multiple_streaming_components

    # Set environment variable and reload settings
    monkeypatch.setenv("HAYHOOKS_STREAMING_COMPONENTS", env_var_value)
    monkeypatch.setattr("hayhooks.server.pipelines.utils.settings", AppSettings())

    mock_chunks = [
        StreamingChunk(content="chunk1_from_component1"),
        StreamingChunk(content="chunk2_from_component1"),
        StreamingChunk(content="chunk1_from_component2"),
        StreamingChunk(content="chunk2_from_component2"),
    ]

    def mock_run(data):
        # Both components should stream
        callback1 = data.get("component1", {}).get("streaming_callback")
        if callback1:
            callback1(mock_chunks[0])
            callback1(mock_chunks[1])
        callback2 = data.get("component2", {}).get("streaming_callback")
        if callback2:
            callback2(mock_chunks[2])
            callback2(mock_chunks[3])

    pipeline.run.side_effect = mock_run

    # Don't pass streaming_components - should use env var
    generator = streaming_generator(pipeline)
    chunks = list(generator)

    assert chunks == mock_chunks


def test_streaming_generator_param_overrides_env_var(monkeypatch, pipeline_with_multiple_streaming_components):
    pipeline = pipeline_with_multiple_streaming_components

    # Set environment variable to "all"
    monkeypatch.setenv("HAYHOOKS_STREAMING_COMPONENTS", "all")
    monkeypatch.setattr("hayhooks.server.pipelines.utils.settings", AppSettings())

    mock_chunks = [
        StreamingChunk(content="chunk1_from_component1"),
        StreamingChunk(content="chunk2_from_component1"),
    ]

    def mock_run(data):
        # Only component1 should stream (explicit param overrides env var)
        callback = data.get("component1", {}).get("streaming_callback")
        if callback:
            callback(mock_chunks[0])
            callback(mock_chunks[1])

    pipeline.run.side_effect = mock_run

    # Explicit parameter should override env var
    generator = streaming_generator(pipeline, streaming_components=["component1"])
    chunks = list(generator)

    assert chunks == mock_chunks


def test_streaming_generator_with_empty_list(pipeline_with_multiple_streaming_components):
    pipeline = pipeline_with_multiple_streaming_components

    def mock_run(data):
        # Neither component should stream
        pass

    pipeline.run.side_effect = mock_run

    generator = streaming_generator(pipeline, streaming_components=[])
    chunks = list(generator)

    assert chunks == []


def test_streaming_generator_with_nonexistent_component_name(pipeline_with_multiple_streaming_components):
    pipeline = pipeline_with_multiple_streaming_components

    mock_chunks = [
        StreamingChunk(content="chunk1_from_component1"),
        StreamingChunk(content="chunk2_from_component1"),
    ]

    def mock_run(data):
        # Only component1 should stream (component3 doesn't exist)
        callback = data.get("component1", {}).get("streaming_callback")
        if callback:
            callback(mock_chunks[0])
            callback(mock_chunks[1])

    pipeline.run.side_effect = mock_run

    # Include non-existent component3
    generator = streaming_generator(pipeline, streaming_components=["component1", "component3"])
    chunks = list(generator)

    assert chunks == mock_chunks


def test_streaming_generator_with_single_component_comma_separated(
    monkeypatch, pipeline_with_multiple_streaming_components
):
    pipeline = pipeline_with_multiple_streaming_components

    # Set environment variable with single component
    monkeypatch.setenv("HAYHOOKS_STREAMING_COMPONENTS", "component1")
    monkeypatch.setattr("hayhooks.server.pipelines.utils.settings", AppSettings())

    mock_chunks = [
        StreamingChunk(content="chunk1_from_component1"),
        StreamingChunk(content="chunk2_from_component1"),
    ]

    def mock_run(data):
        callback = data.get("component1", {}).get("streaming_callback")
        if callback:
            callback(mock_chunks[0])
            callback(mock_chunks[1])

    pipeline.run.side_effect = mock_run

    generator = streaming_generator(pipeline)
    chunks = list(generator)

    assert chunks == mock_chunks


def test_parse_streaming_components_with_empty_string(monkeypatch, pipeline_with_multiple_streaming_components):
    pipeline = pipeline_with_multiple_streaming_components

    # Set environment variable to empty string (default)
    monkeypatch.setenv("HAYHOOKS_STREAMING_COMPONENTS", "")
    monkeypatch.setattr("hayhooks.server.pipelines.utils.settings", AppSettings())

    mock_chunks = [
        StreamingChunk(content="chunk1_from_component2"),
        StreamingChunk(content="chunk2_from_component2"),
    ]

    def mock_run(data):
        # Should use default (last component only)
        callback = data.get("component2", {}).get("streaming_callback")
        if callback:
            callback(mock_chunks[0])
            callback(mock_chunks[1])

    pipeline.run.side_effect = mock_run

    generator = streaming_generator(pipeline)
    chunks = list(generator)

    assert chunks == mock_chunks


def test_parse_streaming_components_setting_with_all():
    from hayhooks.server.pipelines.utils import _parse_streaming_components_setting

    assert _parse_streaming_components_setting("all") == "all"
    assert _parse_streaming_components_setting("ALL") == "all"
    assert _parse_streaming_components_setting("  all  ") == "all"


def test_parse_streaming_components_setting_with_comma_list():
    from hayhooks.server.pipelines.utils import _parse_streaming_components_setting

    result = _parse_streaming_components_setting("llm_1,llm_2,llm_3")
    assert result == ["llm_1", "llm_2", "llm_3"]

    # Test with spaces
    result = _parse_streaming_components_setting("llm_1, llm_2 , llm_3")
    assert result == ["llm_1", "llm_2", "llm_3"]


def test_parse_streaming_components_setting_with_empty():
    from hayhooks.server.pipelines.utils import _parse_streaming_components_setting

    assert _parse_streaming_components_setting("") is None
    assert _parse_streaming_components_setting("   ") is None
