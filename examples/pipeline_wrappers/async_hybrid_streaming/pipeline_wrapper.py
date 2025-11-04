from collections.abc import AsyncGenerator
from pathlib import Path

from haystack import AsyncPipeline

from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message, log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """
        Setup an AsyncPipeline with a legacy OpenAIGenerator component.

        OpenAIGenerator only supports sync streaming callbacks (no run_async() method).
        To use it with AsyncPipeline and async_streaming_generator, we need to enable
        hybrid mode with allow_sync_streaming_callbacks=True.
        """
        pipeline_yaml = (Path(__file__).parent / "hybrid_streaming.yml").read_text()
        self.pipeline = AsyncPipeline.loads(pipeline_yaml)

    async def run_api_async(self, question: str) -> str:
        """
        Simple async API endpoint that returns the final answer.

        Args:
            question: The user's question

        Returns:
            The LLM's answer as a string
        """
        log.trace("Running pipeline with question: {}", question)

        result = await self.pipeline.run_async({"prompt": {"query": question}})
        return result["llm"]["replies"][0]

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        """
        OpenAI-compatible chat completion endpoint with streaming support.

        This demonstrates using allow_sync_streaming_callbacks=True to enable hybrid mode,
        which allows the sync-only OpenAIGenerator to work with async_streaming_generator.

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
        # This is required because OpenAIGenerator (legacy component) only supports
        # sync streaming callbacks. The hybrid mode automatically detects this and
        # bridges the sync callback to work with the async event loop.
        #
        # If all components supported async, this would use pure async mode with no overhead.
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt": {"query": question}},
            allow_sync_streaming_callbacks=True,
        )
