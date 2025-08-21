from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        msg = "Setup failed!"
        raise ValueError(msg)

    def run_api(self) -> dict:
        return {"result": "This should never be reached"}
