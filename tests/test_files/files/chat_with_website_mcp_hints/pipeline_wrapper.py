from haystack import Pipeline

from hayhooks import BasePipelineWrapper, log


class PipelineWrapper(BasePipelineWrapper):
    tool_hints = {
        "title": "Website Q&A",
        "readOnly": True,
        "destructive": False,
        "idempotent": True,
        "openWorld": False,
    }

    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_api(self, urls: list[str], question: str) -> str:
        """
        Ask a question about one or more websites using a Haystack pipeline.
        """
        log.trace("Running pipeline with urls: {} and question: {}", urls, question)
        return "This is a mock response from the pipeline"
