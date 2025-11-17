from collections.abc import Generator

from haystack.dataclasses import StreamingChunk

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    def run_api(self, query: str) -> Generator[StreamingChunk, None, None]:
        def stream() -> Generator[StreamingChunk, None, None]:
            for word in query.split():
                yield StreamingChunk(content=f"{word} ")

        return stream()
