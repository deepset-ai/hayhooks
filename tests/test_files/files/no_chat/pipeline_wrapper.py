from haystack import Pipeline

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = Pipeline()

    def run_api(self, test_param: str) -> str:
        return f"Dummy result with {test_param}"
