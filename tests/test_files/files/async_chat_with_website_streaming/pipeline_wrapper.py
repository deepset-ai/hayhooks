from pathlib import Path
from typing import List, AsyncGenerator
from haystack import Pipeline
from hayhooks import get_last_user_message, BasePipelineWrapper, log


URLS = ["https://haystack.deepset.ai", "https://www.redis.io", "https://ssi.inc"]


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    async def run_chat_completion_async(
        self, model: str, messages: List[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

        # Mock streaming pipeline run, will return a fixed string
        # NOTE: This is used in tests, please don't change it
        if "Redis" in question:
            mock_response = "Redis is an in-memory data structure store, used as a database, cache and message broker."
        else:
            mock_response = "This is a mock response from the pipeline"

        async def mock_generator():
            for word in mock_response.split():
                yield word + " "

        return mock_generator()
