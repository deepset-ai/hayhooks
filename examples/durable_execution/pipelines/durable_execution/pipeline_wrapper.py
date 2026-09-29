"""Retryable, resumable durable Pipeline with no external services."""

from haystack import Pipeline, component
from haystack.core.errors import PipelineRuntimeError
from pydantic import BaseModel, Field

from hayhooks import BasePipelineWrapper, DurableContext
from hayhooks.durable import current_durable_context, durable_streaming_callback


class ExecutionRequest(BaseModel):
    value: int
    require_approval: bool = True
    fail_once: bool = True
    retry_delay_seconds: float = Field(default=0.1, ge=0, le=10)


class Approval(BaseModel):
    approved: bool


class ExecutionResult(BaseModel):
    result: int


@component
class Prepare:
    def __init__(self) -> None:
        self.calls = 0

    @component.output_types(prepared=int)
    def run(self, value: int) -> dict[str, int]:
        self.calls += 1
        if context := current_durable_context():
            context.check_cancelled_sync()
        return {"prepared": value * 2}


@component
class Finish:
    @component.output_types(result=int)
    def run(self, prepared: int, fail_once: bool) -> dict[str, int]:
        context = current_durable_context()
        if context is not None and fail_once and not context.state.get("failure_injected"):
            context.state["failure_injected"] = True
            raise RuntimeError("intentional first-attempt failure")
        if context is not None:
            context.check_cancelled_sync()
        durable_streaming_callback({"text": f"result={prepared + 1}"})
        return {"result": prepared + 1}


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "durable-execution-v1"
    durable_resume_model = Approval

    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.pipeline.add_component("prepare", Prepare())
        self.pipeline.add_component("finish", Finish())
        self.pipeline.connect("prepare.prepared", "finish.prepared")

    async def run_durable_async(self, context: DurableContext, request: ExecutionRequest) -> ExecutionResult:
        if request.require_approval and not context.state.get("approved"):
            resume_input = context.resume_input
            if resume_input is None:
                await context.suspend(
                    {"kind": "approval", "message": "Approve this execution?"},
                    update={"approval_requested": True},
                )
            if not Approval.model_validate(resume_input).approved:
                raise ValueError("execution was rejected")
            context.state["approved"] = True
            await context.report_progress("Execution approved", kind="approval")

        try:
            result = await context.run_pipeline_async(
                {
                    "prepare": {"value": request.value},
                    "finish": {"fail_once": request.fail_once},
                },
                checkpoint_at="finish",
            )
        except PipelineRuntimeError:
            await context.retry("Retrying the intentional failure", delay=request.retry_delay_seconds)
            raise
        return ExecutionResult.model_validate(result["finish"])
