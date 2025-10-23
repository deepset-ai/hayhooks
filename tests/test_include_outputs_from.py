"""Tests for include_outputs_from parameter in streaming generators."""

import os
from typing import Any

import pytest
from haystack import AsyncPipeline, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

from hayhooks.server.pipelines.utils import async_streaming_generator, streaming_generator

QUESTION = "What is the capital of France?"


@pytest.fixture
def document_store():
    store = InMemoryDocumentStore()
    store.write_documents(
        [
            Document(content="Paris is the capital of France."),
            Document(content="London is the capital of the United Kingdom."),
            Document(content="Berlin is the capital of Germany."),
        ]
    )
    return store


@pytest.fixture
def sync_pipeline_with_retriever(document_store):
    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))

    # Create a template that uses documents
    template = [
        ChatMessage.from_user(
            """Answer the question based on the given context.

            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Question: {{ query }}
            Answer:
            """
        )
    ]

    pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template, required_variables="*"))
    pipeline.add_component(
        "llm", OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
    )
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


@pytest.fixture
def async_pipeline_with_retriever(document_store):
    pipeline = AsyncPipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))

    # Create a template that uses documents
    template = [
        ChatMessage.from_user(
            """Answer the question based on the given context.

            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Question: {{ query }}
            Answer:
            """
        )
    ]

    pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template, required_variables="*"))
    pipeline.add_component(
        "llm", OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
    )
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.integration
@pytest.mark.parametrize(
    "include_outputs_from,should_include_retriever",
    [
        (None, False),  # Without include_outputs_from, retriever is NOT included
        ({"retriever"}, True),  # With include_outputs_from, retriever IS included
    ],
    ids=["without_include_outputs_from", "with_include_outputs_from"],
)
def test_streaming_generator_include_outputs_from(
    sync_pipeline_with_retriever, include_outputs_from, should_include_retriever
):
    pipeline = sync_pipeline_with_retriever

    captured_result = {}

    def on_pipeline_end(result: dict[str, Any]) -> None:
        captured_result.update(result)

    generator = streaming_generator(
        pipeline,
        pipeline_run_args={
            "retriever": {"query": QUESTION},
            "prompt_builder": {"query": QUESTION},
        },
        include_outputs_from=include_outputs_from,
        on_pipeline_end=on_pipeline_end,
    )

    # Consume the generator
    list(generator)

    # llm is always included (leaf component)
    assert "llm" in captured_result
    assert "replies" in captured_result["llm"]

    # retriever inclusion depends on include_outputs_from parameter
    if should_include_retriever:
        assert "retriever" in captured_result
        assert "documents" in captured_result["retriever"]
        assert len(captured_result["retriever"]["documents"]) > 0
    else:
        assert "retriever" not in captured_result

    # prompt_builder is never included (intermediate, not in include_outputs_from)
    assert "prompt_builder" not in captured_result


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.integration
@pytest.mark.parametrize(
    "include_outputs_from,should_include_retriever",
    [
        (None, False),  # Without include_outputs_from, retriever is NOT included
        ({"retriever"}, True),  # With include_outputs_from, retriever IS included
    ],
    ids=["without_include_outputs_from", "with_include_outputs_from"],
)
async def test_async_streaming_generator_include_outputs_from(
    async_pipeline_with_retriever, include_outputs_from, should_include_retriever
):
    pipeline = async_pipeline_with_retriever

    captured_result = {}

    def on_pipeline_end(result: dict[str, Any]) -> None:
        captured_result.update(result)

    async_gen = async_streaming_generator(
        pipeline,
        pipeline_run_args={
            "retriever": {"query": QUESTION},
            "prompt_builder": {"query": QUESTION},
        },
        include_outputs_from=include_outputs_from,
        on_pipeline_end=on_pipeline_end,
    )

    # Consume the generator
    async for _ in async_gen:
        pass

    # llm is always included (leaf component)
    assert "llm" in captured_result
    assert "replies" in captured_result["llm"]

    # retriever inclusion depends on include_outputs_from parameter
    if should_include_retriever:
        assert "retriever" in captured_result
        assert "documents" in captured_result["retriever"]
        assert len(captured_result["retriever"]["documents"]) > 0
    else:
        assert "retriever" not in captured_result

    # prompt_builder is never included (intermediate, not in include_outputs_from)
    assert "prompt_builder" not in captured_result
