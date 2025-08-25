from pathlib import Path

from haystack import Pipeline

from hayhooks import BasePipelineWrapper, log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    async def run_api_async(self, urls: list[str], question: str) -> str:
        log.trace(f"Running pipeline with urls: {urls} and question: {question}")

        # NOTE: This is used in tests, please don't change it
        return "This is a mock response from the pipeline"
