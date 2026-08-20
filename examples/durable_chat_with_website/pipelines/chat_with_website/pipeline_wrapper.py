"""A durable chat-with-website Pipeline that streams its answer while it runs."""

from pathlib import Path

from haystack import Pipeline
from haystack.core.errors import PipelineRuntimeError
from pydantic import BaseModel, Field

from hayhooks import BasePipelineWrapper, DurableContext, durable_streaming_callback

DEFAULT_URLS = ["https://haystack.deepset.ai", "https://www.redis.io"]


class ChatRequest(BaseModel):
    """A question to answer from the live contents of a few web pages."""

    question: str = Field(min_length=1, max_length=2_000)
    urls: list[str] = Field(default=DEFAULT_URLS, min_length=1, max_length=5)


class ChatAnswer(BaseModel):
    """The finished answer, already delivered token by token over the stream."""

    reply: str
    urls: list[str]


class PipelineWrapper(BasePipelineWrapper):
    """Fetch pages durably, then stream the generated answer to the SSE endpoint."""

    durable_revision = "durable-chat-with-website-v1"
    def setup(self) -> None:
        self.pipeline = Pipeline.loads((Path(__file__).parent / "chat_with_website.yml").read_text())
        # `async_streaming_generator` passes its callback per run in `pipeline_run_args`.
        # That cannot work under checkpointing: run data is serialized into the
        # PipelineSnapshot, Haystack drops the callable it cannot serialize, and
        # `Pipeline.run` rebuilds `data` from the snapshot on resume, so the callback
        # is gone from the first checkpoint on. Binding it to the component survives.
        #
        # The helper resolves its destination per call from a ContextVar, so one bound
        # callback keeps concurrent durable executions isolated. A run-time callback
        # still wins, leaving ordinary streaming endpoints unaffected.
        self.pipeline.get_component("llm").streaming_callback = durable_streaming_callback

    async def run_durable_async(self, context: DurableContext, request: ChatRequest) -> ChatAnswer:
        # This body re-runs from the top on every attempt, so the message has to say
        # what the attempt will actually do rather than what the first one did.
        resumed = context.record.checkpoint is not None
        await context.report_progress(
            "Resuming from the fetch checkpoint" if resumed else f"Fetching {len(request.urls)} page(s)",
            kind="resume" if resumed else "fetch",
        )
        try:
            outputs = await context.run_pipeline_async(
                {"fetcher": {"urls": request.urls}, "prompt": {"query": request.question}},
                # Fetching is the slow, flaky step. Checkpointing once it is done means a
                # later attempt resumes into generation instead of hitting the network again.
                checkpoint_at=["prompt"],
            )
        except PipelineRuntimeError as error:
            # Without this, a transient generator failure ends the execution and the
            # checkpoint above never pays for itself. The retry is bounded by
            # `durable_max_attempts`.
            await context.retry(f"Pipeline attempt failed: {error}")
        await context.report_progress("Answer complete", kind="completed")
        return ChatAnswer(reply=outputs["llm"]["replies"][0].text, urls=request.urls)
