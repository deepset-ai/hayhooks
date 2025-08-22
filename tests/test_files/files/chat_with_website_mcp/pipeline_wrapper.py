from pathlib import Path

from haystack import Pipeline

from hayhooks import BasePipelineWrapper, log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: list[str], question: str) -> str:
        """
        Ask a question about one or more websites using a Haystack pipeline.
        """
        log.trace(f"Running pipeline with urls: {urls} and question: {question}")
        return "This is a mock response from the pipeline"
