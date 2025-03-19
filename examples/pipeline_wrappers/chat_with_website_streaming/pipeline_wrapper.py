from pathlib import Path
from typing import Generator, List, Union
from haystack import Pipeline
from hayhooks import get_last_user_message, BasePipelineWrapper, log, streaming_generator


URLS = ["https://haystack.deepset.ai", "https://www.redis.io", "https://ssi.inc"]


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: List[str], question: str) -> str:
        log.trace(f"Running pipeline with urls: {urls} and question: {question}")
        result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
        return result["llm"]["replies"][0]

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

        # Streaming pipeline run, will return a generator
        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"fetcher": {"urls": URLS}, "prompt": {"query": question}},
        )
