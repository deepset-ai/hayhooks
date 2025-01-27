from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        raise ValueError("Setup failed!")

    def run_api(self):
        return {"result": "This should never be reached"}
