"""The familiar chat-with-website Pipeline with durable REST execution."""

from pathlib import Path

from haystack import Pipeline
from haystack.core.errors import PipelineRuntimeError
from pydantic import BaseModel, Field, HttpUrl

from hayhooks import BasePipelineWrapper, DurableContext


class WebsiteQuestionRequest(BaseModel):
    """Websites and question accepted by the durable endpoint."""

    urls: list[HttpUrl] = Field(min_length=1, max_length=3)
    question: str = Field(min_length=1, max_length=2_000)


class WebsiteAnswer(BaseModel):
    """Client-safe answer returned after the Pipeline completes."""

    answer: str
    sources: list[str]


class PipelineWrapper(BasePipelineWrapper):
    """Fetch websites once, checkpoint their processing, and answer a question."""

    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: list[str], question: str) -> str:
        """Keep the ordinary synchronous endpoint for side-by-side comparison."""
        outputs = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
        return outputs["llm"]["replies"][0].text

    async def run_durable_async(
        self,
        context: DurableContext,
        request: WebsiteQuestionRequest,
    ) -> WebsiteAnswer:
        sources = [str(url) for url in request.urls]
        await context.report_progress("Website question accepted", kind="accepted")
        try:
            outputs = await context.run_pipeline_async(
                {
                    "fetcher": {"urls": sources},
                    "prompt": {"query": request.question},
                },
                # Preserve the expensive network fetch without copying the
                # converted page and rendered prompt into later snapshots.
                # Conversion, prompting, and the LLM remain replayable.
                checkpoint_at=["converter"],
            )
        except PipelineRuntimeError as error:
            await context.retry(f"Website question Pipeline failed: {error}", delay=2)
            raise

        reply = outputs["llm"]["replies"][0]
        answer = reply.text
        if not isinstance(answer, str) or not answer:
            raise ValueError("The chat generator returned an empty answer")
        await context.report_progress("Website answer completed", kind="completed")
        return WebsiteAnswer(answer=answer, sources=sources)
