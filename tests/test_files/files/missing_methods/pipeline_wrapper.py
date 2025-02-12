from hayhooks import BasePipelineWrapper
from haystack import Pipeline


class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = Pipeline()
