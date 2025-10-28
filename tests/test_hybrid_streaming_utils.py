from collections.abc import AsyncGenerator, Generator

import pytest
from haystack import AsyncPipeline, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.dataclasses import StreamingChunk

from hayhooks.server.pipelines.utils import async_streaming_generator, streaming_generator

QUESTION = "Is Haystack a framework for developing AI applications? Answer Yes or No"


@pytest.fixture
def pipeline_with_huggingface_local_generator():
    prompt_builder = PromptBuilder(template=QUESTION)
    llm = HuggingFaceLocalGenerator(
        model="hf-internal-testing/tiny-random-T5ForConditionalGeneration",
        task="text2text-generation",
        generation_kwargs={"max_new_tokens": 10, "num_beams": 1},  # num_beams=1 required for streaming
    )

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    return pipe


@pytest.fixture
def async_pipeline_with_huggingface_local_generator():
    prompt_builder = PromptBuilder(template=QUESTION)
    llm = HuggingFaceLocalGenerator(
        model="hf-internal-testing/tiny-random-T5ForConditionalGeneration",
        task="text2text-generation",
        generation_kwargs={"max_new_tokens": 10, "num_beams": 1},  # num_beams=1 required for streaming
    )

    pipe = AsyncPipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    return pipe


def test_streaming_generator_with_huggingface_local_generator(pipeline_with_huggingface_local_generator):
    pipeline = pipeline_with_huggingface_local_generator

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


async def test_async_streaming_generator_with_huggingface_local_generator_should_fail(
    async_pipeline_with_huggingface_local_generator,
):
    pipeline = async_pipeline_with_huggingface_local_generator

    with pytest.raises(ValueError, match="seems to not support async streaming callbacks"):
        async_generator = async_streaming_generator(
            pipeline,
            pipeline_run_args={},
        )
        # Try to consume the generator
        _ = [chunk async for chunk in async_generator]


async def test_async_streaming_generator_with_huggingface_local_generator_hybrid_mode(
    async_pipeline_with_huggingface_local_generator,
):
    pipeline = async_pipeline_with_huggingface_local_generator

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


async def test_async_streaming_generator_with_huggingface_local_generator_auto_mode(
    async_pipeline_with_huggingface_local_generator,
):
    pipeline = async_pipeline_with_huggingface_local_generator

    async_generator = async_streaming_generator(
        pipeline,
        pipeline_run_args={},
        allow_sync_streaming_callbacks="auto",
    )

    assert isinstance(async_generator, AsyncGenerator)

    chunks = [chunk async for chunk in async_generator]
    assert len(chunks) > 0

    streaming_chunks = [c for c in chunks if isinstance(c, StreamingChunk)]
    assert len(streaming_chunks) > 0
    assert isinstance(streaming_chunks[0], StreamingChunk)
