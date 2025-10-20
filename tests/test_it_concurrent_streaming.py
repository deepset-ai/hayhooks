import asyncio
import concurrent.futures
import os

import pytest
from haystack import AsyncPipeline, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

from hayhooks.server.pipelines.utils import async_streaming_generator, streaming_generator

# Test configuration
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_KEY_SECRET = Secret.from_env_var("OPENAI_API_KEY") if os.environ.get("OPENAI_API_KEY") else None

NUM_STRESS_TEST_REQUESTS = 10
NUM_SYNC_TEST_REQUESTS = 5

# Skip tests if OpenAI API key is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable required for integration tests",
)


@pytest.fixture(scope="module")
def async_streaming_pipeline():
    pipeline = AsyncPipeline()
    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            api_key=OPENAI_API_KEY_SECRET,
            model=OPENAI_MODEL,
        ),
    )
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


@pytest.fixture(scope="module")
def sync_streaming_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            api_key=OPENAI_API_KEY_SECRET,
            model=OPENAI_MODEL,
        ),
    )
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


def _create_test_message(request_id: str) -> list[ChatMessage]:
    return [ChatMessage.from_user(f"Say only 'Response for request {request_id}' and nothing else.")]


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by removing spaces, underscores and converting to uppercase."""
    return text.replace(" ", "").replace("_", "").upper()


def _verify_chunks_belong_to_request(chunks: list[str], request_id: str) -> None:
    assert chunks, f"Request {request_id} received no chunks"

    # Reconstruct the full response from chunks
    # The LLM response will contain the request_id but may be tokenized across chunks
    full_response = "".join(chunks)

    assert _normalize_text(request_id) in _normalize_text(full_response), (
        f"Request {request_id} did not receive its own response. "
        f"Expected to find '{request_id}' in response. Got: {full_response[:200]}..."
    )


async def _consume_async_stream(pipeline, request_name: str) -> list[str]:
    pipeline_args = {"prompt_builder": {"template": _create_test_message(request_name)}}

    chunks = [
        chunk.content
        async for chunk in async_streaming_generator(
            pipeline,
            pipeline_run_args=pipeline_args,
        )
        if isinstance(chunk, StreamingChunk) and chunk.content
    ]

    return chunks


def _consume_sync_stream(pipeline, request_name: str) -> list[str]:
    pipeline_args = {"prompt_builder": {"template": _create_test_message(request_name)}}

    chunks = [
        chunk.content
        for chunk in streaming_generator(
            pipeline,
            pipeline_run_args=pipeline_args,
        )
        if isinstance(chunk, StreamingChunk) and chunk.content
    ]

    return chunks


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_concurrent_streaming_no_interference(async_streaming_pipeline):
    """
    Test that concurrent async streaming requests don't interfere with each other.

    This test simulates two clients making simultaneous streaming requests to the same
    pipeline instance and verifies that:
    1. Each request receives chunks
    2. Chunks are not cross-contaminated between requests
    3. Requests complete successfully
    """
    # Run both requests concurrently
    chunks_1, chunks_2 = await asyncio.gather(
        _consume_async_stream(async_streaming_pipeline, "REQUEST_1"),
        _consume_async_stream(async_streaming_pipeline, "REQUEST_2"),
    )

    # Verify both requests completed successfully without interference
    _verify_chunks_belong_to_request(chunks_1, "REQUEST_1")
    _verify_chunks_belong_to_request(chunks_2, "REQUEST_2")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_concurrent_streaming_stress_test(async_streaming_pipeline):
    """
    Stress test with many concurrent async streaming requests.

    This test simulates realistic load with multiple concurrent users all streaming
    from the same pipeline instance. It verifies that:
    1. All requests complete successfully
    2. No chunks are lost or cross-contaminated
    3. The system handles high concurrency gracefully
    """
    # Run all requests concurrently
    results = await asyncio.gather(
        *[_consume_async_stream(async_streaming_pipeline, f"REQ_{i}") for i in range(NUM_STRESS_TEST_REQUESTS)]
    )

    # Verify all requests completed successfully
    assert len(results) == NUM_STRESS_TEST_REQUESTS, f"Expected {NUM_STRESS_TEST_REQUESTS} results, got {len(results)}"

    # Verify each request received its own response
    for request_id, chunks in enumerate(results):
        _verify_chunks_belong_to_request(chunks, f"REQ_{request_id}")


@pytest.mark.integration
def test_sync_concurrent_streaming_with_threads(sync_streaming_pipeline):
    """
    Test concurrent sync streaming requests using threading.

    This test verifies the fix works with the synchronous streaming generator,
    which uses threading internally. It simulates multiple threads making
    simultaneous requests and verifies proper isolation.
    """
    # Run concurrent requests using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_SYNC_TEST_REQUESTS) as executor:
        futures = [
            executor.submit(_consume_sync_stream, sync_streaming_pipeline, f"SYNC_REQ_{i}")
            for i in range(NUM_SYNC_TEST_REQUESTS)
        ]
        results = [future.result() for future in futures]

    # Verify all requests completed successfully
    assert len(results) == NUM_SYNC_TEST_REQUESTS, f"Expected {NUM_SYNC_TEST_REQUESTS} results, got {len(results)}"

    # Verify each request received its own response
    for request_id, chunks in enumerate(results):
        _verify_chunks_belong_to_request(chunks, f"SYNC_REQ_{request_id}")
