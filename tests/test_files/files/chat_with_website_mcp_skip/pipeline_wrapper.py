from haystack import Pipeline

from hayhooks import BasePipelineWrapper, log


class PipelineWrapper(BasePipelineWrapper):
    skip_mcp = True

    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_api(self, urls: list[str], question: str) -> str:
        """
        Ask a question about one or more websites using a Haystack pipeline.
        """
        log.trace(f"Running pipeline with urls: {urls} and question: {question}")
        return "This is a mock response from the pipeline"
