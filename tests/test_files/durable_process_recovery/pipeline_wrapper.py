"""Deterministic durable wrapper used by the process-recovery smoke test."""

import sqlite3
import time
from pathlib import Path

from haystack import Pipeline, component
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext, current_execution_id


class RecoveryRequest(BaseModel):
    database_path: str
    ready_file: str


class RecoveryResult(BaseModel):
    attempt: int
    effect_applied: bool


@component
class Checkpoint:
    @component.output_types(value=str)
    def run(self, database_path: str) -> dict[str, str]:
        execution_id = current_execution_id()
        if execution_id is None:
            msg = "checkpoint component requires a durable execution"
            raise RuntimeError(msg)
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS checkpoint_runs (execution_id TEXT PRIMARY KEY)")
            connection.execute("INSERT INTO checkpoint_runs VALUES (?)", (execution_id,))
        return {"value": "checkpointed"}


@component
class Effect:
    @component.output_types(effect_applied=bool)
    def run(self, value: str, database_path: str, ready_file: str) -> dict[str, bool]:
        del value
        execution_id = current_execution_id()
        if execution_id is None:
            msg = "effect component requires a durable execution"
            raise RuntimeError(msg)
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS effects (execution_id TEXT PRIMARY KEY)")
            effect_applied = (
                connection.execute("INSERT OR IGNORE INTO effects VALUES (?)", (execution_id,)).rowcount == 1
            )
        if effect_applied:
            Path(ready_file).write_text("ready", encoding="utf-8")
            time.sleep(60)
        return {"effect_applied": effect_applied}


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "process-recovery"

    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.pipeline.add_component("checkpoint", Checkpoint())
        self.pipeline.add_component("effect", Effect())
        self.pipeline.connect("checkpoint.value", "effect.value")

    async def run_durable_async(self, context: DurableContext, request: RecoveryRequest) -> RecoveryResult:
        outputs = await context.run_pipeline_async(
            {
                "checkpoint": {"database_path": request.database_path},
                "effect": {"database_path": request.database_path, "ready_file": request.ready_file},
            },
            checkpoint_at=["checkpoint", "effect"],
        )
        return RecoveryResult(attempt=context.attempt, effect_applied=outputs["effect"]["effect_applied"])
