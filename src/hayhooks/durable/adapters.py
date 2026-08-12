"""
Haystack 3 adapters used by :class:`hayhooks.durable.context.DurableContext`.

The adapters use only Haystack's public PipelineSnapshot, Agent, State, and
hook APIs.  The imports intentionally stay lazy so the base Hayhooks install
continues to support Haystack 2 for non-durable deployments.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from typing import Any, cast

from haystack.lazy_imports import LazyImport

from hayhooks.durable.context import DurableContext
from hayhooks.durable.models import ExecutionCheckpoint, ExecutionKind, RetryableExecutionError, validate_json

_HAYSTACK_V3_ERROR = (
    "Durable execution requires Haystack 3. Install `hayhooks[durable]` in the durable server environment."
)
_AGENT_CHECKPOINT_PHASE = "_hayhooks_agent_checkpoint_phase"
_AGENT_FINAL_PHASE = "after_run"
_AGENT_INTERNAL_STATE_KEYS = frozenset(("continue_run", "tools", "hook_context"))


async def _run_fenced_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Keep the caller's durable claim alive until non-cancellable thread work exits."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        return await task


# Keep every Haystack 3-only symbol behind Haystack's supported optional-import
# boundary. This module is imported by the base package in Haystack 2
# environments, where Agent, hooks, and snapshots intentionally do not exist.
with LazyImport(_HAYSTACK_V3_ERROR) as haystack_v3_import:
    from haystack import Pipeline
    from haystack.components.agents import Agent
    from haystack.components.agents.state import State
    from haystack.core.errors import BreakpointException, PipelineRuntimeError
    from haystack.dataclasses import ChatMessage
    from haystack.dataclasses.breakpoints import Breakpoint, PipelineSnapshot

    FunctionHook = importlib.import_module("haystack.hooks.from_function").FunctionHook


def require_haystack_v3() -> None:
    """Fail durable deployment explicitly when the optional v3 extra is missing."""
    try:
        haystack_v3_import.check()
        import haystack
    except ImportError as error:  # pragma: no cover - dependency failure
        raise RuntimeError(_HAYSTACK_V3_ERROR) from error
    major = str(getattr(haystack, "__version__", "0")).split(".", maxsplit=1)[0]
    if major != "3":
        raise RuntimeError(_HAYSTACK_V3_ERROR)


class HaystackDurableAdapter:
    """Bind a validated Haystack 3 Pipeline or Agent to execution contexts."""

    def __init__(self, pipeline: Any, kind: ExecutionKind) -> None:
        require_haystack_v3()
        self.pipeline = pipeline
        self.kind = kind
        if kind is ExecutionKind.PIPELINE:
            self._validate_pipeline()
        else:
            self._validate_agent()
            self._install_agent_checkpoint_hooks()

    def _validate_pipeline(self) -> None:
        haystack_v3_import.check()
        if not isinstance(self.pipeline, Pipeline):
            msg = "run_durable Pipeline wrappers must set self.pipeline to a Haystack 3 Pipeline"
            raise TypeError(msg)

    def _validate_agent(self) -> None:
        haystack_v3_import.check()
        if not isinstance(self.pipeline, Agent):
            msg = "durable Agent wrappers must set self.pipeline to a Haystack 3 Agent"
            raise TypeError(msg)

    async def run_pipeline_async(
        self, context: DurableContext, data: dict[str, Any], *, checkpoint_at: list[str]
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            await _run_fenced_thread(self.run_pipeline, context, data, checkpoint_at=checkpoint_at),
        )

    def run_pipeline(
        self, context: DurableContext, data: dict[str, Any], *, checkpoint_at: list[str]
    ) -> dict[str, Any]:
        if self.kind is not ExecutionKind.PIPELINE:
            msg = "run_pipeline is available only when self.pipeline is a Haystack Pipeline"
            raise TypeError(msg)
        snapshot = None
        if context.record.checkpoint is not None:
            checkpoint = context.record.checkpoint
            if checkpoint.kind is not ExecutionKind.PIPELINE:
                msg = "The persisted checkpoint is not a PipelineSnapshot"
                raise TypeError(msg)
            snapshot = PipelineSnapshot.from_dict(cast(dict[str, Any], checkpoint.data["snapshot"]))
        boundaries = list(checkpoint_at)
        if snapshot is not None:
            break_point: Any = snapshot.break_point
            completed_visits = snapshot.pipeline_state.component_visits
            boundaries = [
                name
                for name in boundaries
                if completed_visits.get(name, 0) == 0
                and not (name == break_point.component_name and break_point.visit_count == 0)
            ]

        next_data = data if snapshot is None else {}
        try:
            while boundaries:
                component_name = boundaries.pop(0)
                break_point = Breakpoint(component_name=component_name)
                try:
                    return cast(
                        dict[str, Any],
                        self.pipeline.run(data=next_data, pipeline_snapshot=snapshot, break_point=break_point),
                    )
                except BreakpointException as error:
                    if error.pipeline_snapshot is None:
                        msg = "Haystack breakpoint did not expose a PipelineSnapshot"
                        raise RetryableExecutionError(msg) from error
                    snapshot = error.pipeline_snapshot
                    context.record.append_progress(
                        f"Checkpoint saved before pipeline component '{component_name}'",
                        kind="checkpoint",
                    )
                    context._sync_await(context.checkpoint(_pipeline_checkpoint(context, snapshot)))
                    next_data = {}
            return cast(dict[str, Any], self.pipeline.run(data=next_data, pipeline_snapshot=snapshot))
        except PipelineRuntimeError as error:
            if error.pipeline_snapshot is not None:
                context._sync_await(context.checkpoint(_pipeline_checkpoint(context, error.pipeline_snapshot)))
            raise

    async def run_agent_async(self, context: DurableContext, *, messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        if self.kind is not ExecutionKind.AGENT:
            msg = "run_agent_async is available only when self.pipeline is a Haystack Agent"
            raise TypeError(msg)
        final_result = _final_agent_result(context)
        if final_result is not None:
            return final_result
        method = getattr(self.pipeline, "run_async", None)
        if callable(method):
            return cast(dict[str, Any], await method(messages=messages, **kwargs))
        return cast(
            dict[str, Any],
            await _run_fenced_thread(self.run_agent, context, messages=messages, **kwargs),
        )

    def run_agent(self, context: DurableContext, *, messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        if self.kind is not ExecutionKind.AGENT:
            msg = "run_agent is available only when self.pipeline is a Haystack Agent"
            raise TypeError(msg)
        final_result = _final_agent_result(context)
        if final_result is not None:
            return final_result
        return cast(dict[str, Any], self.pipeline.run(messages=messages, **kwargs))

    def _install_agent_checkpoint_hooks(self) -> None:  # noqa: C901, PLR0915
        """Install once; hooks select the active execution through ContextVar."""
        if getattr(self.pipeline, "_hayhooks_durable_hooks_installed", False):
            return

        def restore_before_run(state: State) -> None:
            if context := _current_durable_context():
                _restore_agent_state(context, state)

        async def restore_before_run_async(state: State) -> None:
            if context := _current_durable_context():
                _restore_agent_state(context, state)

        def check_cancelled_before_llm(state: State) -> None:
            del state
            if context := _current_durable_context():
                context.check_cancelled_sync()

        async def check_cancelled_before_llm_async(state: State) -> None:
            del state
            if context := _current_durable_context():
                await context.check_cancelled()

        def checkpoint_after_tool(state: State) -> None:
            context = _current_durable_context()
            if context is None:
                return
            if not _agent_exits_after_tools(state, self.pipeline.exit_conditions):
                context._sync_await(_checkpoint_agent_state(context, state))
            context.check_cancelled_sync()

        async def checkpoint_after_tool_async(state: State) -> None:
            context = _current_durable_context()
            if context is None:
                return
            if not _agent_exits_after_tools(state, self.pipeline.exit_conditions):
                await _checkpoint_agent_state(context, state)
            await context.check_cancelled()

        def checkpoint_on_exit(state: State) -> None:
            context = _current_durable_context()
            if context is not None and state.data["continue_run"]:
                context._sync_await(_checkpoint_agent_state(context, state))

        async def checkpoint_on_exit_async(state: State) -> None:
            context = _current_durable_context()
            if context is not None and state.data["continue_run"]:
                await _checkpoint_agent_state(context, state)

        def checkpoint_after_run(state: State) -> None:
            if context := _current_durable_context():
                context._sync_await(_checkpoint_agent_state(context, state, final=True))

        async def checkpoint_after_run_async(state: State) -> None:
            if context := _current_durable_context():
                await _checkpoint_agent_state(context, state, final=True)

        # The module uses postponed annotations, while Haystack validates hook
        # signatures with ``inspect.signature`` rather than resolving hints.
        for function in (
            restore_before_run,
            restore_before_run_async,
            check_cancelled_before_llm,
            check_cancelled_before_llm_async,
            checkpoint_after_tool,
            checkpoint_after_tool_async,
            checkpoint_on_exit,
            checkpoint_on_exit_async,
            checkpoint_after_run,
            checkpoint_after_run_async,
        ):
            function.__annotations__["state"] = State

        hooks = dict(getattr(self.pipeline, "hooks", {}) or {})
        hooks["before_run"] = [
            FunctionHook(function=restore_before_run, async_function=restore_before_run_async),
            *hooks.get("before_run", []),
        ]
        hooks["before_llm"] = [
            FunctionHook(function=check_cancelled_before_llm, async_function=check_cancelled_before_llm_async),
            *hooks.get("before_llm", []),
        ]
        hooks["after_tool"] = [
            *hooks.get("after_tool", []),
            FunctionHook(function=checkpoint_after_tool, async_function=checkpoint_after_tool_async),
        ]
        hooks["on_exit"] = [
            *hooks.get("on_exit", []),
            FunctionHook(function=checkpoint_on_exit, async_function=checkpoint_on_exit_async),
        ]
        hooks["after_run"] = [
            *hooks.get("after_run", []),
            FunctionHook(function=checkpoint_after_run, async_function=checkpoint_after_run_async),
        ]
        self.pipeline.hooks = hooks
        self.pipeline._hayhooks_durable_hooks_installed = True


def _current_durable_context() -> DurableContext | None:
    from hayhooks.durable.context import get_current_durable_context

    return get_current_durable_context()


def _pipeline_checkpoint(context: DurableContext, snapshot: Any) -> ExecutionCheckpoint:
    return ExecutionCheckpoint(
        ExecutionKind.PIPELINE,
        {"snapshot": validate_json(snapshot.to_dict(), limit=context.record.max_record_bytes, label="snapshot")},
    )


def _checkpoint_data(state: Any, context: DurableContext, *, final: bool = False) -> dict[str, Any]:
    """Exclude live resources from State's otherwise public serialization."""
    payload = _without_live_agent_resources(state.to_dict())
    if final:
        payload[_AGENT_CHECKPOINT_PHASE] = _AGENT_FINAL_PHASE
    return cast(
        dict[str, Any],
        validate_json(payload, limit=context.record.max_record_bytes, label="Agent state"),
    )


def _without_live_agent_resources(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove per-run Agent resources from a serialized state checkpoint."""
    cleaned = dict(payload)
    data = dict(cast(Mapping[str, Any], cleaned.get("data", {})))
    schema = dict(cast(Mapping[str, Any], cleaned.get("schema", {})))
    serialization_schema = dict(data.get("serialization_schema", {}))
    properties = dict(serialization_schema.get("properties", {}))
    serialized_data = dict(data.get("serialized_data", {}))
    for key in ("tools", "hook_context"):
        schema.pop(key, None)
        properties.pop(key, None)
        serialized_data.pop(key, None)
    serialization_schema["properties"] = properties
    data["serialization_schema"] = serialization_schema
    data["serialized_data"] = serialized_data
    cleaned["schema"] = schema
    cleaned["data"] = data
    return cleaned


async def _checkpoint_agent_state(context: DurableContext, state: Any, *, final: bool = False) -> None:
    context.record.append_progress(
        "Agent final checkpoint saved" if final else "Agent step checkpoint saved", kind="checkpoint"
    )
    await context.checkpoint(ExecutionCheckpoint(ExecutionKind.AGENT, _checkpoint_data(state, context, final=final)))


def _agent_exits_after_tools(state: State, exit_conditions: list[str]) -> bool:
    """Return whether Haystack will stop after the current tool-result messages."""
    if exit_conditions == ["text"]:
        return False
    matched = False
    for message in reversed(state.data.get("messages", [])):
        result = message.tool_call_result
        if result is None:
            break
        if result.origin.tool_name not in exit_conditions:
            continue
        if result.error:
            return False
        matched = True
    return matched


def _final_agent_result(context: DurableContext) -> dict[str, Any] | None:
    """Return a checkpointed terminal Agent result without re-entering the Agent loop."""
    checkpoint = context.record.checkpoint
    if (
        checkpoint is None
        or checkpoint.kind is not ExecutionKind.AGENT
        or checkpoint.data.get(_AGENT_CHECKPOINT_PHASE) != _AGENT_FINAL_PHASE
    ):
        return None
    state = State.from_dict(_without_live_agent_resources(checkpoint.data))
    result = {key: value for key, value in state.data.items() if key not in _AGENT_INTERNAL_STATE_KEYS}
    if messages := result.get("messages"):
        result["last_message"] = messages[-1]
    return result


def _restore_agent_state(context: DurableContext, state: Any) -> None:
    """Restore a recovered State, retaining fresh per-run live resources."""
    checkpoint = context.record.checkpoint
    if checkpoint is None or checkpoint.kind is not ExecutionKind.AGENT:
        return
    restored = State.from_dict(_without_live_agent_resources(checkpoint.data))
    live_tools = state.data.get("tools")
    live_hook_context = state.data.get("hook_context")
    state.data.clear()
    state.data.update(restored.data)
    if live_tools is not None:
        state.data["tools"] = live_tools
    if live_hook_context is not None:
        state.data["hook_context"] = live_hook_context
    resume = context.take_resume_input()
    if isinstance(resume, dict) and isinstance(resume.get("messages"), list):
        state.data.setdefault("messages", []).extend(
            ChatMessage.from_dict(message) for message in resume["messages"] if isinstance(message, dict)
        )


def execution_kind(pipeline: Any) -> ExecutionKind:
    """Classify a real Haystack 3 Pipeline or Agent behind the lazy boundary."""
    require_haystack_v3()
    if isinstance(pipeline, Pipeline):
        return ExecutionKind.PIPELINE
    if isinstance(pipeline, Agent):
        return ExecutionKind.AGENT
    msg = "Durable wrappers must set self.pipeline to a real Haystack 3 Pipeline or Agent"
    raise TypeError(msg)


__all__ = ["HaystackDurableAdapter", "execution_kind", "require_haystack_v3"]
