from haystack import Pipeline

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = Pipeline()
