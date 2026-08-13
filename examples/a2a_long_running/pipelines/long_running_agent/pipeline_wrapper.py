"""A durable A2A Agent that calls a real Haystack document-preparation Pipeline."""

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Annotated

from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.hooks.from_function import FunctionHook
from haystack.tools import tool

from hayhooks import A2APipelineWrapper, current_durable_context, current_execution_id

_MAX_DEMO_TOOL_DELAY_SECONDS = 300.0


def _demo_delay_seconds(name: str, default: str = "0") -> float:
    raw_delay = os.getenv(name, default)
    try:
        delay = float(raw_delay)
    except ValueError as error:
        msg = f"{name} must be a number"
        raise ValueError(msg) from error
    if not 0 <= delay <= _MAX_DEMO_TOOL_DELAY_SECONDS:
        msg = f"{name} must be between 0 and 300"
        raise ValueError(msg)
    return delay


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
    context = current_durable_context()
    execution_id = current_execution_id()
    if context is None or execution_id is None:
        msg = "This example tool must run inside a durable execution"
        raise RuntimeError(msg)
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
    default_database = Path(tempfile.gettempdir()) / "hayhooks-durable-a2a.sqlite3"
    database = os.getenv("HAYHOOKS_EXAMPLE_INDEX_DB", str(default_database))
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

    # Hold the tool open *after* its external effect. Killing the server in
    # this window replays the tool from the previous Agent checkpoint, while
    # the SQLite primary key proves that the effect is still applied once.
    delay = _demo_delay_seconds("HAYHOOKS_EXAMPLE_TOOL_DELAY_SECONDS", "3")
    context.report_progress_sync(
        f"Indexing effect committed; holding the tool open for {delay:g} seconds",
        kind="side_effect_committed",
        metadata={
            "idempotency_key": effect_key,
            "side_effect_applied": applied,
        },
    )
    if delay:
        time.sleep(delay)

    return json.dumps(
        {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "idempotency_key": effect_key,
            "side_effect_applied": applied,
            "chunks": [{"chunk_id": str(chunk.id), "preview": (chunk.content or "")[:160]} for chunk in chunks],
        }
    )


@tool
def publish_indexing_receipt(
    document_id: Annotated[str, "The stable identifier returned by document preparation"],
    chunk_count: Annotated[int, "The prepared chunk count returned by document preparation"],
) -> str:
    """Perform the inexpensive follow-up step after document preparation succeeds."""
    context = current_durable_context()
    if context is None:
        msg = "This example tool must run inside a durable execution"
        raise RuntimeError(msg)

    delay = _demo_delay_seconds("HAYHOOKS_EXAMPLE_RECEIPT_DELAY_SECONDS")
    context.report_progress_sync(
        f"Indexing is checkpointed; holding the lightweight receipt update for {delay:g} seconds",
        kind="receipt_started",
    )
    if delay:
        time.sleep(delay)

    return json.dumps({"document_id": document_id, "chunk_count": chunk_count, "receipt": "published"})


class PipelineWrapper(A2APipelineWrapper):
    """Let Hayhooks map this real tool-using Agent to durable A2A executions."""

    durable_revision = "a2a-long-running-agent"

    def setup(self) -> None:
        self.pipeline = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            tools=[prepare_document_for_indexing, publish_indexing_receipt],
            system_prompt=(
                "You prepare documents for retrieval and publish their catalog status. When a user supplies a "
                "document identifier and content, first call only prepare_document_for_indexing. After its result, "
                "in a later tool turn call only publish_indexing_receipt with its document_id and chunk_count. "
                "Do not prepare a document again once its successful result is present. Then report the number "
                "of chunks and a concise readiness summary. Treat a follow-up approval message as authorization "
                "to proceed."
            ),
            hooks={"before_llm": [FunctionHook(function=require_approval, async_function=require_approval_async)]},
        )
