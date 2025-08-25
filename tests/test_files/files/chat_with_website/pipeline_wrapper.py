from collections.abc import Generator
from pathlib import Path
from typing import Union

from haystack import Pipeline

from hayhooks import BasePipelineWrapper, get_last_user_message, log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: list[str], question: str) -> str:
        log.trace(f"Running pipeline with urls: {urls} and question: {question}")

        # NOTE: This is used in tests, please don't change it
        return "This is a mock response from the pipeline"

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> Union[str, Generator]:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

        # Mock streaming pipeline run, will return a fixed string
        # NOTE: This is used in tests, please don't change it
        return "This is a mock response from the pipeline"
