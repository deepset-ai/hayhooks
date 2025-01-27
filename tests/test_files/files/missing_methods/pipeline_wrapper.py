from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from haystack import Pipeline

class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = Pipeline()
