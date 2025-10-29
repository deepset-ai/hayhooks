from collections.abc import AsyncGenerator, Generator
from typing import Any, Optional

import pytest
from haystack import AsyncPipeline, Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import StreamingChunk

from hayhooks.server.pipelines.utils import async_streaming_generator, streaming_generator

QUESTION = "Is Haystack a framework for developing AI applications? Answer Yes or No"


@component
class MockSyncOnlyGenerator:
    """
    Mock generator component that only supports sync streaming callbacks.
    Simulates components like OpenAIGenerator that don't have run_async() support.
    """

    @component.output_types(replies=list[str])
    def run(self, prompt: str, streaming_callback: Optional[Any] = None) -> dict[str, Any]:
        """Run method with streaming_callback parameter (sync only, no run_async)."""
        # Simulate streaming output
        chunks = ["This ", "is ", "a ", "test ", "response."]

        if streaming_callback:
            for i, chunk_text in enumerate(chunks):
                chunk = StreamingChunk(content=chunk_text, index=i)
                streaming_callback(chunk)

        return {"replies": ["This is a test response."]}


@pytest.fixture
def pipeline_with_sync_only_generator():
    prompt_builder = PromptBuilder(template=QUESTION)
    llm = MockSyncOnlyGenerator()

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    return pipe


@pytest.fixture
def async_pipeline_with_sync_only_generator():
    prompt_builder = PromptBuilder(template=QUESTION)
    llm = MockSyncOnlyGenerator()

    pipe = AsyncPipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    return pipe


def test_streaming_generator_with_sync_only_generator(pipeline_with_sync_only_generator):
    """Test that sync streaming works with sync-only generator components."""
    pipeline = pipeline_with_sync_only_generator

    generator = streaming_generator(
        pipeline,
        pipeline_run_args={},
    )

    assert isinstance(generator, Generator)

    chunks = list(generator)
    assert len(chunks) > 0

    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]
    assert len(streaming_chunks) > 0
    assert isinstance(streaming_chunks[0], StreamingChunk)


async def test_async_streaming_generator_with_sync_only_generator_false_or_default_mode(
    async_pipeline_with_sync_only_generator,
):
    """
    Test that async streaming fails without hybrid mode for sync-only components.
    This is the default mode when allow_sync_streaming_callbacks is not provided.
    """
    pipeline = async_pipeline_with_sync_only_generator

    with pytest.raises(ValueError, match="seems to not support async streaming callbacks"):
        async_generator = async_streaming_generator(
            pipeline,
            pipeline_run_args={},
        )
        # Try to consume the generator
        _ = [chunk async for chunk in async_generator]


async def test_async_streaming_generator_with_sync_only_generator_true_mode(
    async_pipeline_with_sync_only_generator,
):
    """
    Test that allow_sync_streaming_callbacks=True correctly detects and enables
    hybrid mode for sync-only components.
    """
    pipeline = async_pipeline_with_sync_only_generator

    async_generator = async_streaming_generator(
        pipeline,
        pipeline_run_args={},
        allow_sync_streaming_callbacks=True,
    )

    assert isinstance(async_generator, AsyncGenerator)

    chunks = [chunk async for chunk in async_generator]
    assert len(chunks) > 0

    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]
    assert len(streaming_chunks) > 0
    assert isinstance(streaming_chunks[0], StreamingChunk)
