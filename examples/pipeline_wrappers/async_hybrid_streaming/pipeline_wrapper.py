from collections.abc import AsyncGenerator
from typing import Any

from haystack import Pipeline
from haystack.core.component import component
from haystack.dataclasses import StreamingChunk

from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message, log


@component
class SyncOnlyGenerator:
    """
    A minimal sync-only generator.

    It implements ``run`` with a ``streaming_callback`` but intentionally does NOT
    implement ``run_async``. Haystack v3 removed the legacy ``OpenAIGenerator``, but
    sync-only components (custom or third-party) still exist. This component stands in
    for one so the example can demonstrate hybrid streaming without any external service.
    """

    @component.output_types(replies=list[str])
    def run(self, query: str, streaming_callback: Any | None = None) -> dict[str, Any]:
        reply = f"You asked: {query}"
        if streaming_callback:
            # Emit the reply token-by-token via a *sync* callback.
            for index, word in enumerate(reply.split()):
                streaming_callback(StreamingChunk(content=word + " ", index=index))
        return {"replies": [reply]}


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """
        Build a Pipeline containing a sync-only component.

        Because ``SyncOnlyGenerator`` has no ``run_async()``, streaming it through
        ``async_streaming_generator`` requires hybrid mode
        (``allow_sync_streaming_callbacks=True``), which bridges the sync streaming
        callback onto the async event loop.
        """
        self.pipeline = Pipeline()
        self.pipeline.add_component("llm", SyncOnlyGenerator())

    async def run_api_async(self, question: str) -> str:
        """
        Simple async API endpoint that returns the final answer.

        Args:
            question: The user's question

        Returns:
            The generator's reply as a string
        """
        log.trace("Running pipeline with question: {}", question)

        result = await self.pipeline.run_async({"llm": {"query": question}})
        return result["llm"]["replies"][0]

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        """
        OpenAI-compatible chat completion endpoint with streaming support.

        This demonstrates using allow_sync_streaming_callbacks=True to enable hybrid mode,
        which allows the sync-only component to work with async_streaming_generator.

        Args:
            model: The model name (ignored in this example)
            messages: Chat messages in OpenAI format
            body: Additional request parameters

        Yields:
            Streaming chunks from the pipeline execution
        """
        log.trace("Running pipeline with model: {}, messages: {}, body: {}", model, messages, body)

        question = get_last_user_message(messages)
        log.trace("Question: {}", question)

        # ✅ Enable hybrid mode with allow_sync_streaming_callbacks=True
        # This is required because SyncOnlyGenerator only supports sync streaming
        # callbacks (no run_async). Hybrid mode automatically detects this and bridges
        # the sync callback to work with the async event loop.
        #
        # If all components supported async, this would use pure async mode with no overhead.
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"llm": {"query": question}},
            allow_sync_streaming_callbacks=True,
        )
