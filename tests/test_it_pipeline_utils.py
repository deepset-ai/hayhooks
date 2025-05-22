import os
import pytest
from pathlib import Path
from haystack import Pipeline, AsyncPipeline
from haystack.dataclasses import ChatMessage
from typing import Generator, AsyncGenerator
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.deploy_utils import add_pipeline_to_registry
from hayhooks.server.pipelines.utils import async_streaming_generator, streaming_generator

TEST_FILES_DIR_STREAMING = Path(__file__).parent / "test_files/files/chat_with_website_streaming"
SAMPLE_PIPELINE_FILES_STREAMING = {
    "pipeline_wrapper.py": (TEST_FILES_DIR_STREAMING / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (TEST_FILES_DIR_STREAMING / "chat_with_website.yml").read_text(),
}

TEST_FILES_DIR_ASYNC_STREAMING = Path(__file__).parent / "test_files/files/async_question_answer"
SAMPLE_PIPELINE_FILES_ASYNC_STREAMING = {
    "pipeline_wrapper.py": (TEST_FILES_DIR_ASYNC_STREAMING / "pipeline_wrapper.py").read_text(),
    "question_answer.yml": (TEST_FILES_DIR_ASYNC_STREAMING / "question_answer.yml").read_text(),
}

QUESTION = "Is Haystack a framework for developing AI applications? Answer Yes or No"


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.integration
def test_streaming_generator():
    pipeline_name = "test_streaming_generator"
    pipeline_files = SAMPLE_PIPELINE_FILES_STREAMING

    add_pipeline_to_registry(pipeline_name, pipeline_files)

    pipeline_wrapper = registry.get(pipeline_name)
    assert pipeline_wrapper is not None
    assert isinstance(pipeline_wrapper.pipeline, Pipeline)

    generator = streaming_generator(
        pipeline_wrapper.pipeline,
        {
            "fetcher": {"urls": ["https://haystack.deepset.ai/"]},
            "prompt": {"query": QUESTION},
        },
    )
    assert isinstance(generator, Generator)

    chunks = [chunk for chunk in generator]
    assert len(chunks) > 0
    assert isinstance(chunks[0], str)
    assert "Yes" in chunks


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.integration
async def test_async_streaming_generator():
    pipeline_name = "test_async_streaming_generator"
    pipeline_files = SAMPLE_PIPELINE_FILES_ASYNC_STREAMING

    add_pipeline_to_registry(pipeline_name, pipeline_files)

    pipeline_wrapper = registry.get(pipeline_name)
    assert pipeline_wrapper is not None
    assert isinstance(pipeline_wrapper.pipeline, AsyncPipeline)

    messages = [ChatMessage.from_user(QUESTION)]

    async_generator = async_streaming_generator(
        pipeline_wrapper.pipeline,
        {"prompt_builder": {"template": messages}},
    )
    assert isinstance(async_generator, AsyncGenerator)

    chunks = [chunk async for chunk in async_generator]
    assert len(chunks) > 0
    assert isinstance(chunks[0], str)
    assert "Yes" in chunks
