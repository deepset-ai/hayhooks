from collections.abc import AsyncGenerator
from pathlib import Path

from haystack import AsyncPipeline
from haystack.dataclasses import ChatMessage

from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message, log

SYSTEM_MESSAGE = "You are a helpful assistant that can answer questions about the world."


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "question_answer.yml").read_text()
        self.pipeline = AsyncPipeline.loads(pipeline_yaml)

    async def run_api_async(self, question: str) -> str:
        log.trace(f"Running pipeline with question: {question}")

        result = await self.pipeline.run_async(
            {
                "prompt_builder": {
                    "template": [
                        ChatMessage.from_system(SYSTEM_MESSAGE),
                        ChatMessage.from_user(question),
                    ]
                }
            }
        )
        return result["llm"]["replies"][0].text

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "prompt_builder": {
                    "template": [
                        ChatMessage.from_system(SYSTEM_MESSAGE),
                        ChatMessage.from_user(question),
                    ]
                },
            },
        )
