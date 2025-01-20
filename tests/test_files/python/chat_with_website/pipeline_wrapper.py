from pathlib import Path
from typing import List
from haystack import Pipeline
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


URLS = ["https://haystack.deepset.ai", "https://www.redis.io"]


class PipelineWrapper(BasePipelineWrapper):
    def __init__(self) -> None:
        self.pipeline: Pipeline = None

    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: List[str], question: str) -> dict:
        return self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})

    def run_chat(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> dict:
        question = user_message
        return self.pipeline.run({"fetcher": {"urls": URLS}, "prompt": {"query": question}})
