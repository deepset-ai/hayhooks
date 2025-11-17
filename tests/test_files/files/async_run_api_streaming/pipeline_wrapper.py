import asyncio
from collections.abc import AsyncGenerator

from haystack.dataclasses import StreamingChunk

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    async def run_api_async(self, query: str) -> AsyncGenerator[StreamingChunk, None]:
        async def stream() -> AsyncGenerator[StreamingChunk, None]:
            for word in query.split():
                await asyncio.sleep(0)
                yield StreamingChunk(content=f"{word} ")

        return stream()
