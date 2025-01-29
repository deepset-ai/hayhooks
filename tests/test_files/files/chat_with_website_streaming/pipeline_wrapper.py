from pathlib import Path
from pprint import pprint
from typing import Generator, List, Union
from haystack import Pipeline
from hayhooks.server.pipelines.utils import get_last_user_message, streaming_generator
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.logger import log


URLS = ["https://haystack.deepset.ai", "https://www.redis.io", "https://ssi.inc"]


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: List[str], question: str) -> str:
        log.trace(f"Running pipeline with urls: {urls} and question: {question}")
        result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
        return result["llm"]["replies"][0]

    def run_chat(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

        # Real pipeline run with streaming
        # return streaming_generator(
        #     pipeline=self.pipeline,
        #     pipeline_run_args={"fetcher": {"urls": URLS}, "prompt": {"query": question}},
        # )

        # Mock streaming pipeline run, will return a fixed string
        # NOTE: This is used in tests, please don't change it
        if "Redis" in question:
            mock_response = "Redis is an in-memory data structure store, used as a database, cache and message broker."
        else:
            mock_response = "This is a mock response from the pipeline"

        def mock_generator():
            for word in mock_response.split():
                yield word + " "

        return mock_generator()
