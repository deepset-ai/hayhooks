"""Deterministic durable A2A Agent used by the process-restart smoke test."""

from haystack import component
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.dataclasses import ChatMessage
from haystack.hooks.from_function import FunctionHook

from hayhooks import A2APipelineWrapper, current_durable_context


def require_approval(state: State) -> None:
    context = current_durable_context()
    if context is None or context.state.get("approval_requested"):
        return
    context.state["approval_requested"] = True
    context.suspend_sync({"kind": "approval", "message": "Approve this task"})


async def require_approval_async(state: State) -> None:
    context = current_durable_context()
    if context is None or context.state.get("approval_requested"):
        return
    context.state["approval_requested"] = True
    await context.suspend({"kind": "approval", "message": "Approve this task"})


@component
class FakeChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools=None):
        return {"replies": [ChatMessage.from_assistant("approved and complete")]}


class PipelineWrapper(A2APipelineWrapper):
    durable_revision = "durable-a2a-process-recovery"

    def setup(self) -> None:
        self.pipeline = Agent(
            chat_generator=FakeChatGenerator(),
            tools=[],
            hooks={"before_llm": [FunctionHook(function=require_approval, async_function=require_approval_async)]},
        )
