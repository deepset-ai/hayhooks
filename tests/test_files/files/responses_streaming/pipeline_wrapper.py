from collections.abc import Generator

from haystack import Pipeline
from haystack.dataclasses import StreamingChunk

from hayhooks import BasePipelineWrapper, get_last_user_input_text, log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_response(self, model: str, input_items: list[dict], body: dict) -> Generator[StreamingChunk, None, None]:
        log.trace(
            "Running pipeline with model: {}, input_items: {}, body: {}",
            model,
            input_items,
            body,
        )

        question = get_last_user_input_text(input_items)
        log.trace("Question: {}", question)

        # NOTE: This is used in tests, please don't change it
        if question and "Redis" in question:
            mock_response = "Redis is an in-memory data structure store, used as a database, cache and message broker."
        else:
            mock_response = "This is a mock response from the pipeline"

        def mock_generator():
            for word in mock_response.split():
                yield StreamingChunk(content=word + " ")

        return mock_generator()
