"""Pipeline wrapper with a broken import to test error handling."""

from haystack import Pipeline
from hayhooks import BasePipelineWrapper

# This import will fail - module doesn't exist
from .nonexistent_module import some_function


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_api(self, value: int) -> int:
        return value
