"""A durable document-preparation Pipeline using real Haystack components."""

import time

from haystack import Document, Pipeline, component
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from pydantic import BaseModel, Field

from hayhooks import BasePipelineWrapper, DurableContext, current_durable_context


class SourceDocument(BaseModel):
    """One raw document accepted by the durable indexing-preparation job."""

    document_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=10_000)


class DocumentPreparationRequest(BaseModel):
    """Documents to clean and split into embedding-ready chunks."""

    documents: list[SourceDocument] = Field(min_length=1, max_length=25)
    fail_first_attempt: bool = False
    require_approval: bool = False
    demo_delay_seconds: float = Field(default=0, ge=0, le=300)


class ApprovalInput(BaseModel):
    """Typed input accepted by the generated resume endpoint."""

    approved: bool


class PreparedChunk(BaseModel):
    """A compact, client-safe projection of a Haystack Document chunk."""

    document_id: str
    chunk_id: str
    content: str


class DocumentPreparationResult(BaseModel):
    """The chunks produced by the real Haystack preprocessing Pipeline."""

    document_count: int
    chunk_count: int
    chunks: list[PreparedChunk]


@component
class DemoDelay:
    """Optional pause after a persisted checkpoint, used only for restart demonstrations."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], seconds: float) -> dict[str, list[Document]]:
        if seconds:
            context = current_durable_context()
            if context is not None:
                context.report_progress_sync(
                    f"Checkpointed demo delay started for {seconds:g} seconds",
                    kind="demo_delay",
                )
            time.sleep(seconds)
        return {"documents": documents}


class PipelineWrapper(BasePipelineWrapper):
    """Clean and chunk documents before a later embedding/indexing stage."""

    durable_revision = "durable-document-preparation"
    durable_resume_model = ApprovalInput

    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.pipeline.add_component("clean", DocumentCleaner(remove_empty_lines=True))
        self.pipeline.add_component("demo_delay", DemoDelay())
        self.pipeline.add_component(
            "split",
            DocumentSplitter(split_by="word", split_length=80, split_overlap=10),
        )
        self.pipeline.connect("clean.documents", "demo_delay.documents")
        self.pipeline.connect("demo_delay.documents", "split.documents")

    async def run_durable_async(
        self, context: DurableContext, request: DocumentPreparationRequest
    ) -> DocumentPreparationResult:
        await context.report_progress("Document preparation accepted", kind="accepted")
        if request.fail_first_attempt and context.attempt == 1:
            await context.report_progress("Demonstrating one bounded retry", kind="retry_demo")
            await context.retry("Intentional first-attempt failure", delay=1)

        if request.require_approval and context.resume_input is None:
            await context.suspend(
                {
                    "kind": "approval",
                    "message": "Approve document preparation",
                    "expected_input_schema": ApprovalInput.model_json_schema(),
                }
            )
        if request.require_approval:
            approval = ApprovalInput.model_validate(context.take_resume_input())
            if not approval.approved:
                msg = "Document preparation was not approved"
                raise ValueError(msg)

        documents = [
            Document(id=source.document_id, content=source.content, meta={"document_id": source.document_id})
            for source in request.documents
        ]
        outputs = await context.run_pipeline_async(
            {
                "clean": {"documents": documents},
                "demo_delay": {"seconds": request.demo_delay_seconds},
            },
            # The delay begins after clean has completed and the snapshot before
            # demo_delay has been persisted, creating a reliable crash window.
            checkpoint_at=["clean", "demo_delay", "split"],
        )
        chunks = outputs["split"]["documents"]
        await context.report_progress("Document preparation completed", kind="completed")
        return DocumentPreparationResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            chunks=[
                PreparedChunk(
                    document_id=str(chunk.meta["document_id"]),
                    chunk_id=str(chunk.id),
                    content=chunk.content or "",
                )
                for chunk in chunks
            ],
        )
