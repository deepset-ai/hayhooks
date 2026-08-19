"""A durable chat-with-website Pipeline that streams its answer while it runs."""

from pathlib import Path

from haystack import Pipeline
from haystack.core.errors import PipelineRuntimeError
from haystack.dataclasses import StreamingChunk
from pydantic import BaseModel, Field

from hayhooks import BasePipelineWrapper, DurableContext, current_durable_context

DEFAULT_URLS = ["https://haystack.deepset.ai", "https://www.redis.io"]


class ChatRequest(BaseModel):
    """A question to answer from the live contents of a few web pages."""

    question: str = Field(min_length=1, max_length=2_000)
    urls: list[str] = Field(default=DEFAULT_URLS, min_length=1, max_length=5)


class ChatAnswer(BaseModel):
    """The finished answer, already delivered token by token over the stream."""

    reply: str
    urls: list[str]


def stream_to_execution(chunk: StreamingChunk) -> None:
    """
    Forward one generated token to the SSE stream of whichever execution is running.

    Resolving the execution per call, rather than closing over one, is what lets a
    single shared Pipeline serve concurrent durable executions without crossing
    their streams.
    """
    if context := current_durable_context():
        context.stream_chunk_sync(chunk)


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
        # Sharing one bound callback across concurrent executions is safe for the same
        # reason `_async_streaming_callback` is: it is a module-level function that
        # resolves its destination per call from a ContextVar. Hayhooks routes on
        # `_ASYNC_STREAMING_QUEUE`; this routes on the durable execution context. A
        # run-time `streaming_callback` still wins, so ordinary streaming endpoints on
        # this wrapper are unaffected.
        self.pipeline.get_component("llm").streaming_callback = stream_to_execution

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
