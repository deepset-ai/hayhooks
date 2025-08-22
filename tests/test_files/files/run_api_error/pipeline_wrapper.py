from haystack import Pipeline

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_api(self, test_param: str) -> str:
        msg = "This is a test error"
        raise ValueError(msg)
