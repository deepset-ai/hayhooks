from collections.abc import AsyncGenerator
from pathlib import Path

from haystack import AsyncPipeline

from hayhooks import BasePipelineWrapper, log

SYSTEM_MESSAGE = "You are a helpful assistant that can answer questions about the world."


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "question_answer.yml").read_text()
        self.pipeline = AsyncPipeline.loads(pipeline_yaml)

    async def run_api_async(self, question: str) -> str:
        log.trace(f"Running pipeline with question: {question}")

        return "This is a mock response from the pipeline"

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        mock_response = "This is a mock response from the pipeline"

        async def mock_generator():
            for word in mock_response.split():
                yield word + " "

        return mock_generator()
