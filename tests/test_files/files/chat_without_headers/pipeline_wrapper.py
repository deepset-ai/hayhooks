from haystack import Pipeline

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    """Keeps the pre-existing signature, so Hayhooks must not forward request headers."""

    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str:
        # NOTE: This is used in tests, please don't change it
        return "no headers parameter declared"
