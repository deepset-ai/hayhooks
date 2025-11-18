from collections.abc import Generator
from pathlib import Path
from typing import Optional

from haystack import Pipeline
from haystack.dataclasses import StreamingChunk

from hayhooks import BasePipelineWrapper, log, streaming_generator

DEFAULT_URLS = [
    "https://haystack.deepset.ai",
    "https://www.redis.io",
    "https://ssi.inc",
]


class PipelineWrapper(BasePipelineWrapper):
    """
    Pipeline wrapper that streams responses directly from the /run endpoint.

    The wrapper uses ``streaming_generator`` so `/run` returns a streaming response instead
    of waiting for the pipeline to finish.
    """

    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(
        self,
        question: str,
        urls: Optional[list[str]] = None,
    ) -> Generator[StreamingChunk, None, None]:
        """
        Execute the pipeline and stream tokens back to the caller.

        Args:
            question: User question about the target websites.
            urls: Optional list of URLs to crawl. Defaults to a curated list.
        """
        target_urls = urls or DEFAULT_URLS
        log.info("Streaming pipeline run for question='{}' urls={}", question, target_urls)

        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "fetcher": {"urls": target_urls},
                "prompt": {"query": question},
            },
        )
