from haystack import Pipeline

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    """Declares the optional `headers` parameter, so Hayhooks forwards the request headers."""

    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_chat_completion(self, model: str, messages: list[dict], body: dict, headers: dict[str, str]) -> str:
        # NOTE: This is used in tests, please don't change it
        return f"authorization={headers.get('authorization', 'missing')}"
