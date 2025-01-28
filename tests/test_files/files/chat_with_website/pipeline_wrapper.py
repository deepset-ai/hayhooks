from pathlib import Path
from typing import Generator, List, Union
from haystack import Pipeline
from hayhooks.server.pipelines.utils import get_last_user_message, streaming_generator
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.logger import log


URLS = ["https://haystack.deepset.ai", "https://www.redis.io"]


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

        # Plain pipeline run, will return a string
        # result = self.pipeline.run({"fetcher": {"urls": URLS}, "prompt": {"query": question}})
        # return result["llm"]["replies"][0]

        # Streaming pipeline run, will return a generator
        # def pipeline_runner():
        #     self.pipeline.run({"fetcher": {"urls": URLS}, "prompt": {"query": question}})

        # return streaming_generator(self.pipeline, pipeline_runner)

        # Mock streaming pipeline run, will return a fixed string
        # NOTE: This is used in tests, please don't change it
        return "This is a mock response from the pipeline"
