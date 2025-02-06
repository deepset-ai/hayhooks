from haystack import Pipeline
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_api(self, test_param: str) -> str:
        raise ValueError("This is a test error")
