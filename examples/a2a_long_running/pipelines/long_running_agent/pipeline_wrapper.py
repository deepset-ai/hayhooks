"""A durable A2A Agent that calls a real Haystack document-preparation Pipeline."""

import json
import os
import sqlite3
import time
from typing import Annotated

from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.hooks.from_function import FunctionHook
from haystack.tools import tool

from hayhooks import A2APipelineWrapper, current_durable_context, current_execution_id


def require_approval(state: State) -> None:  # noqa: ARG001
    """Suspend before the first model call so A2A exposes input-required."""
    context = current_durable_context()
    if context is None or context.state.get("approval_requested"):
        return
    context.state["approval_requested"] = True
    context.suspend_sync(
        {
            "kind": "approval",
            "message": "Approve the indexing side effect",
            "expected_input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        }
    )


async def require_approval_async(state: State) -> None:  # noqa: ARG001
    context = current_durable_context()
    if context is None or context.state.get("approval_requested"):
        return
    context.state["approval_requested"] = True
    await context.suspend(
        {
            "kind": "approval",
            "message": "Approve the indexing side effect",
            "expected_input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        }
    )


@tool
def prepare_document_for_indexing(
    document_id: Annotated[str, "A stable identifier for the source document"],
    content: Annotated[str, "Raw document text to clean and split into chunks"],
) -> str:
    """Clean, chunk, and idempotently record an indexing side effect."""
    execution_id = current_execution_id()
    if execution_id is None:
        raise RuntimeError("This example tool must run inside a durable execution")
    effect_key = f"{execution_id}:index:{document_id}"
    preparation_pipeline = Pipeline()
    preparation_pipeline.add_component("clean", DocumentCleaner(remove_empty_lines=True))
    preparation_pipeline.add_component(
        "split",
        DocumentSplitter(split_by="word", split_length=80, split_overlap=10),
    )
    preparation_pipeline.connect("clean.documents", "split.documents")
    outputs = preparation_pipeline.run(
        {"clean": {"documents": [Document(id=document_id, content=content, meta={"document_id": document_id})]}}
    )
    chunks = outputs["split"]["documents"]
    # Make detached polling, cancellation, and restart behavior observable.
    time.sleep(3)
    database = os.getenv("HAYHOOKS_EXAMPLE_INDEX_DB", "/tmp/hayhooks-durable-a2a.sqlite3")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS indexing_effects "
            "(idempotency_key TEXT PRIMARY KEY, document_id TEXT NOT NULL, chunk_count INTEGER NOT NULL)"
        )
        cursor = connection.execute(
            "INSERT OR IGNORE INTO indexing_effects (idempotency_key, document_id, chunk_count) VALUES (?, ?, ?)",
            (effect_key, document_id, len(chunks)),
        )
        applied = cursor.rowcount == 1
    return json.dumps(
        {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "idempotency_key": effect_key,
            "side_effect_applied": applied,
            "chunks": [{"chunk_id": str(chunk.id), "preview": (chunk.content or "")[:160]} for chunk in chunks],
        }
    )


class PipelineWrapper(A2APipelineWrapper):
    """Let Hayhooks map this real tool-using Agent to durable A2A executions."""

    durable = True

    def setup(self) -> None:
        self.pipeline = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            tools=[prepare_document_for_indexing],
            system_prompt=(
                "You prepare documents for retrieval. When a user supplies a document identifier and content, "
                "always call prepare_document_for_indexing before responding. Report the number of chunks and "
                "a concise readiness summary. Treat a follow-up approval message as authorization to proceed."
            ),
            hooks={
                "before_llm": [
                    FunctionHook(function=require_approval, async_function=require_approval_async)
                ]
            },
        )
