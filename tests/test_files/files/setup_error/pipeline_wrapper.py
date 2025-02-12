from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        raise ValueError("Setup failed!")

    def run_api(self) -> dict:
        return {"result": "This should never be reached"}
