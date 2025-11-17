from collections.abc import Generator

from hayhooks import BasePipelineWrapper
from hayhooks.open_webui import create_message_event


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    def run_api(self, query: str) -> Generator:
        def stream() -> Generator:
            yield create_message_event(content=f"event: {query}")

        return stream()
