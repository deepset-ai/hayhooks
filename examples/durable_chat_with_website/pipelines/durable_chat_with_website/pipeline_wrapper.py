"""Durable website Q&A with checkpointed fetching and bounded display streaming."""

import re

from haystack import Document, Pipeline, component
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from pydantic import BaseModel, Field, HttpUrl

from hayhooks import BasePipelineWrapper, DurableContext
from hayhooks.durable import current_durable_context, durable_streaming_callback


class WebsiteRequest(BaseModel):
    urls: list[HttpUrl] = Field(min_length=1, max_length=3)
    question: str = Field(min_length=1, max_length=500)


class WebsiteAnswer(BaseModel):
    answer: str
    sources: list[str]


@component
class Answer:
    @component.output_types(answer=str)
    def run(self, documents: list[Document], question: str) -> dict[str, str]:
        terms = {term.lower() for term in re.findall(r"[\w'-]+", question)}
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(document.content or "" for document in documents))
        matches = [sentence.strip() for sentence in sentences if terms & set(sentence.lower().split())]
        answer = " ".join(matches[:3])[:2_000] or "No matching passage was found in the fetched pages."
        for text in re.findall(r"\S+\s*", answer):
            if context := current_durable_context():
                context.check_cancelled_sync()
            durable_streaming_callback({"text": text})
        return {"answer": answer}


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "durable-chat-with-website-v1"

    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.pipeline.add_component("fetch", LinkContentFetcher(timeout=10, retry_attempts=1))
        self.pipeline.add_component("convert", HTMLToDocument())
        self.pipeline.add_component("answer", Answer())
        self.pipeline.connect("fetch.streams", "convert.sources")
        self.pipeline.connect("convert.documents", "answer.documents")

    async def run_durable_async(self, context: DurableContext, request: WebsiteRequest) -> WebsiteAnswer:
        urls = [str(url) for url in request.urls]
        result = await context.run_pipeline_async(
            {"fetch": {"urls": urls}, "answer": {"question": request.question}},
            checkpoint_at="answer",
        )
        return WebsiteAnswer(answer=result["answer"]["answer"], sources=urls)
