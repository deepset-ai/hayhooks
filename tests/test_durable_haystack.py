"""Haystack 3.1 durable checkpoint adapters."""

import asyncio
import threading
from importlib.metadata import version

import pytest

from hayhooks.durable.context import _ExecutionSuspendedError, durable_context_scope
from hayhooks.durable.engine import ExecutionLeaseLostError, PayloadKind, Resume, ScheduleRetry
from hayhooks.durable.haystack import HaystackDurableAdapter, _agent_checkpoint
from hayhooks.durable.models import ExecutionKind, encode_json
from tests.durable_store_contract import decode_checkpoint

try:
    _HAYSTACK_VERSION = tuple(int(part) for part in version("haystack-ai").split(".", maxsplit=2)[:2])
except ValueError:
    _HAYSTACK_VERSION = (0, 0)
_HAYSTACK_V3 = (3, 1) <= _HAYSTACK_VERSION < (4, 0)
requires_haystack_v3 = pytest.mark.skipif(not _HAYSTACK_V3, reason="durable adapters require Haystack 3.1+")

if _HAYSTACK_V3:
    from haystack import Pipeline, component
    from haystack.components.agents import Agent
    from haystack.components.agents.state import State
    from haystack.components.generators.chat import MockChatGenerator
    from haystack.core.errors import PipelineRuntimeError
    from haystack.dataclasses import ChatMessage, ToolCall
    from haystack.dataclasses.breakpoints import PipelineSnapshot
    from haystack.hooks import FunctionHook, hook
    from haystack.tools import Tool

    @component
    class Increment:
        def __init__(self, *, fail_once: bool = False) -> None:
            self.calls = 0
            self.fail_once = fail_once

        @component.output_types(value=int)
        def run(self, value: int) -> dict[str, int]:
            self.calls += 1
            if self.fail_once and self.calls == 1:
                message = "interrupted component"
                raise RuntimeError(message)
            return {"value": value + 1}

    @component
    class RecordingChatGenerator:
        def __init__(self, *, fail_on_call: int | None = None, tool_on_first_call: bool = False) -> None:
            self.calls = 0
            self.fail_on_call = fail_on_call
            self.tool_on_first_call = tool_on_first_call

        @component.output_types(replies=list[ChatMessage])
        def run(self, messages: list[ChatMessage], tools=None):
            del messages, tools
            self.calls += 1
            if self.calls == self.fail_on_call:
                message = "interrupted agent"
                raise RuntimeError(message)
            if self.calls == 1 and self.tool_on_first_call:
                return {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="work", arguments={})])]}
            return {"replies": [ChatMessage.from_assistant("done")]}


@pytest.mark.skipif(_HAYSTACK_V3, reason="the supported dependency is installed")
def test_adapter_reports_the_targeted_haystack_installation_error() -> None:
    with pytest.raises(RuntimeError, match=r"hayhooks\[durable\]"):
        HaystackDurableAdapter(object())


@requires_haystack_v3
def test_adapter_rejects_non_haystack_targets() -> None:
    with pytest.raises(TypeError, match="real Haystack"):
        HaystackDurableAdapter(object())


@requires_haystack_v3
@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
async def test_pipeline_failure_snapshot_resumes_without_repeating_completed_components(
    context_factory, run_async: bool
) -> None:
    store, create = context_factory
    first, second = Increment(), Increment(fail_once=True)
    pipeline = Pipeline()
    pipeline.add_component("first", first)
    pipeline.add_component("second", second)
    pipeline.connect("first.value", "second.value")
    adapter = HaystackDurableAdapter(pipeline)
    context, claim = await create()

    with durable_context_scope(context), pytest.raises(PipelineRuntimeError):
        if run_async:
            await adapter.run_pipeline_async(context, {"first": {"value": 1}}, checkpoint_at="second")
        else:
            await asyncio.to_thread(
                adapter.run_pipeline,
                context,
                {"first": {"value": 1}},
                checkpoint_at="second",
            )

    failed = await store.read(context.execution_id)
    assert failed is not None
    checkpoint = decode_checkpoint(failed.payloads[PayloadKind.CHECKPOINT])
    assert isinstance(checkpoint.adapter_checkpoint, dict)
    PipelineSnapshot.from_dict(checkpoint.adapter_checkpoint)
    await claim.transition(ScheduleRetry(claim.control.fence, claim.worker_id, 0, 0, 2, b"{}"))
    resumed, _ = await create(context.execution_id, submit=False)
    with durable_context_scope(resumed):
        result = (
            await adapter.run_pipeline_async(resumed, {"unused": {}}, checkpoint_at="second")
            if run_async
            else await asyncio.to_thread(
                adapter.run_pipeline,
                resumed,
                {"unused": {}},
                checkpoint_at="second",
            )
        )

    assert result == {"second": {"value": 3}}
    assert (first.calls, second.calls) == (1, 2)


@requires_haystack_v3
async def test_async_pipeline_cancellation_returns_before_its_thread_finishes(context_factory, monkeypatch) -> None:
    _, create = context_factory
    adapter = HaystackDurableAdapter(Pipeline())
    context, _ = await create()
    started, release = threading.Event(), threading.Event()

    def blocking_run(_context, _data, *, checkpoint_at):
        assert checkpoint_at is None
        started.set()
        assert release.wait(timeout=2)
        return {"done": True}

    monkeypatch.setattr(adapter, "run_pipeline", blocking_run)
    task = asyncio.create_task(adapter.run_pipeline_async(context, {}))
    assert await asyncio.to_thread(started.wait, 1)
    try:
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.05)
        assert task in done and task.cancelled()
    finally:
        release.set()
        if not task.done():
            await task


@requires_haystack_v3
async def test_pipeline_rejects_multiple_conditional_boundaries(context_factory) -> None:
    _, create = context_factory
    adapter = HaystackDurableAdapter(Pipeline())
    context, _ = await create()

    with durable_context_scope(context), pytest.raises(TypeError, match="one Pipeline component"):
        await asyncio.to_thread(
            adapter.run_pipeline,
            context,
            {},
            checkpoint_at=["left", "right"],
        )


@requires_haystack_v3
@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
async def test_agent_state_resume_and_final_checkpoint_are_recoverable(context_factory, run_async: bool) -> None:
    store, create = context_factory
    seen = {}

    def capture(state: State) -> None:
        seen.update(
            counter=state.data["counter"],
            messages=[message.text for message in state.data["messages"]],
            tools=state.data["tools"],
            hook_context=state.data["hook_context"],
        )

    generator = RecordingChatGenerator()
    agent = Agent(
        chat_generator=generator,
        tools=[],
        state_schema={"counter": {"type": int}},
        hooks={"before_run": [FunctionHook(function=capture)]},
    )
    adapter = HaystackDurableAdapter(agent)
    context, _ = await create(kind=ExecutionKind.AGENT)
    state = State(
        schema=agent.resolved_state_schema,
        data={
            "counter": 7,
            "messages": [ChatMessage.from_user("before restart")],
            "step_count": 0,
            "token_usage": {},
            "context_tokens": 0,
            "tool_call_counts": {},
            "exit_reason": None,
            "continue_run": False,
            "tools": ["stale tool"],
            "hook_context": {"stale": True},
        },
    )
    before_version = context._claim.control.version
    await context.report_progress("Agent step checkpoint saved", kind="checkpoint")
    await context.checkpoint(_agent_checkpoint(state))
    checkpointed = await store.read(context.execution_id)
    assert checkpointed is not None and checkpointed.control.version == before_version + 1
    checkpoint = decode_checkpoint(checkpointed.payloads[PayloadKind.CHECKPOINT])
    assert isinstance(checkpoint.adapter_checkpoint, dict)
    serialized = checkpoint.adapter_checkpoint["state"]["data"]["serialized_data"]
    assert not ({"tools", "hook_context"} & serialized.keys())
    assert len(checkpointed.progress) == 1

    with pytest.raises(_ExecutionSuspendedError):
        await context.suspend({"kind": "message"})
    waiting = await store.read(context.execution_id)
    assert waiting is not None
    checkpoint = decode_checkpoint(waiting.payloads[PayloadKind.CHECKPOINT]).model_copy(
        update={"resume_input": {"messages": [ChatMessage.from_user("after restart").to_dict()]}}
    )
    await store.transition(
        context.execution_id,
        Resume(
            0,
            "v1",
            encode_json(checkpoint.model_dump(mode="json"), max_bytes=4_096),
            expected_version=waiting.control.version,
        ),
    )
    resumed, claim = await create(context.execution_id, submit=False)
    before_final = claim.control.version
    with durable_context_scope(resumed):
        kwargs = {
            "messages": [ChatMessage.from_user("unused")],
            "counter": 0,
            "hook_context": {"live": True},
        }
        result = (
            await adapter.run_agent_async(resumed, **kwargs)
            if run_async
            else await asyncio.to_thread(adapter.run_agent, resumed, **kwargs)
        )

    finalized = await store.read(context.execution_id)
    assert finalized is not None and finalized.control.version == before_final + 1
    assert len(finalized.progress) == 2
    assert seen == {
        "counter": 7,
        "messages": ["before restart", "after restart"],
        "tools": [],
        "hook_context": {"live": True},
    }
    assert result["counter"] == 7 and resumed.resume_input is None

    await claim.transition(ScheduleRetry(claim.control.fence, claim.worker_id, 0, 0, 2, b"{}"))
    recovered, _ = await create(context.execution_id, submit=False)
    with durable_context_scope(recovered):
        messages = [ChatMessage.from_user("must not run")]
        replay = (
            await adapter.run_agent_async(recovered, messages=messages)
            if run_async
            else await asyncio.to_thread(adapter.run_agent, recovered, messages=messages)
        )
    assert replay == result
    assert generator.calls == 1


@requires_haystack_v3
@pytest.mark.parametrize("continuation", ["tool", "on_exit"])
@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
async def test_agent_continuation_checkpoints_before_the_next_llm_call(
    context_factory, continuation: str, run_async: bool
) -> None:
    store, create = context_factory

    @hook
    def continue_once(state: State) -> None:
        if state.data.get("marker") is None:
            state.set("marker", "saved")
            state.set("continue_run", True)

    def work() -> str:
        return "worked"

    tool = Tool(
        name="work",
        description="Complete one step",
        parameters={"type": "object", "properties": {}},
        function=work,
    )
    agent = Agent(
        chat_generator=RecordingChatGenerator(fail_on_call=2, tool_on_first_call=continuation == "tool"),
        tools=[tool] if continuation == "tool" else [],
        state_schema={"marker": {"type": str}},
        hooks={"on_exit": [continue_once]} if continuation == "on_exit" else None,
    )
    adapter = HaystackDurableAdapter(agent)
    context, _ = await create(kind=ExecutionKind.AGENT)
    with durable_context_scope(context), pytest.raises(RuntimeError, match="interrupted agent"):
        messages = [ChatMessage.from_user("question")]
        if run_async:
            await adapter.run_agent_async(context, messages=messages)
        else:
            await asyncio.to_thread(adapter.run_agent, context, messages=messages)

    stored = await store.read(context.execution_id)
    assert stored is not None and len(stored.progress) == 1
    checkpoint = decode_checkpoint(stored.payloads[PayloadKind.CHECKPOINT]).adapter_checkpoint
    serialized = checkpoint["state"]["data"]["serialized_data"]
    assert serialized["step_count"] == 1
    if continuation == "on_exit":
        assert serialized["marker"] == "saved"
    else:
        assert serialized["messages"][-1]["content"][0]["tool_call_result"]["result"] == "worked"


@requires_haystack_v3
async def test_agent_hooks_are_idempotent_and_ignore_ordinary_runs() -> None:
    generator = RecordingChatGenerator()
    agent = Agent(chat_generator=generator, tools=[])
    HaystackDurableAdapter(agent)
    hook_counts = {name: len(hooks) for name, hooks in agent.hooks.items()}
    HaystackDurableAdapter(agent)

    sync_result = agent.run(messages=[ChatMessage.from_user("sync")])
    async_result = await agent.run_async(messages=[ChatMessage.from_user("async")])

    assert {name: len(hooks) for name, hooks in agent.hooks.items()} == hook_counts
    assert sync_result["last_message"].text == async_result["last_message"].text == "done"
    assert generator.calls == 2


@requires_haystack_v3
def test_adapted_agent_remains_serializable() -> None:
    agent = Agent(chat_generator=MockChatGenerator(responses="done"), tools=[])
    HaystackDurableAdapter(agent)
    hook_counts = {name: len(hooks) for name, hooks in agent.hooks.items()}

    restored = Agent.from_dict(agent.to_dict())
    HaystackDurableAdapter(restored)

    assert {name: len(hooks) for name, hooks in restored.hooks.items()} == hook_counts


@requires_haystack_v3
@pytest.mark.parametrize(
    "method",
    ["run_pipeline", "run_pipeline_async", "run_agent", "run_agent_async"],
)
async def test_sync_and_async_adapters_reject_a_lost_fence(context_factory, method: str) -> None:
    _, create = context_factory
    agent_method = method.startswith("run_agent")
    target = Agent(chat_generator=RecordingChatGenerator(), tools=[]) if agent_method else Pipeline()
    adapter = HaystackDurableAdapter(target)
    context, claim = await create(kind=adapter.kind)
    claim.mark_lost()

    with pytest.raises(ExecutionLeaseLostError):
        if method == "run_pipeline":
            await asyncio.to_thread(adapter.run_pipeline, context, {})
        elif method == "run_pipeline_async":
            await adapter.run_pipeline_async(context, {})
        elif method == "run_agent":
            await asyncio.to_thread(adapter.run_agent, context, messages=[])
        else:
            await adapter.run_agent_async(context, messages=[])
