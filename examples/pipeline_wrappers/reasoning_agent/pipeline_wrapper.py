from collections.abc import AsyncGenerator

from haystack import AsyncPipeline
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage

from hayhooks import (
    BasePipelineWrapper,
    async_streaming_generator,
    chat_messages_from_openai_response,
    log,
)

MODEL_NAME = "gpt-5.4-mini"

# Request reasoning summaries so Open WebUI can render "Thinking" blocks.
DEFAULT_GENERATION_KWARGS = {
    "reasoning": {"effort": "high", "summary": "auto"},
}


class PipelineWrapper(BasePipelineWrapper):
    pipeline: AsyncPipeline

    def setup(self) -> None:
        self.pipeline = AsyncPipeline()
        self.pipeline.add_component(
            "llm",
            OpenAIResponsesChatGenerator(
                model=MODEL_NAME,
                generation_kwargs=DEFAULT_GENERATION_KWARGS,
            ),
        )

    async def run_chat_completion_async(
        self,
        model: str,
        messages: list[dict],
        body: dict,  # noqa: ARG002
    ) -> AsyncGenerator:
        log.trace("Running reasoning agent (request model='{}', backend model='{}')", model, MODEL_NAME)

        chat_messages = [ChatMessage.from_openai_dict_format(message) for message in messages]

        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "llm": {
                    "messages": chat_messages,
                }
            },
        )

    async def run_response_async(
        self,
        model: str,
        input_items: list[dict],
        body: dict,  # noqa: ARG002
    ) -> AsyncGenerator:
        log.trace("Running reasoning agent Responses API (request model='{}', backend model='{}')", model, MODEL_NAME)

        messages = chat_messages_from_openai_response(input_items)

        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "llm": {
                    "messages": messages,
                }
            },
        )
