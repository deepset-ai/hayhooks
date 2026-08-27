"""Haystack 3.1 Pipeline and Agent checkpoint adapters."""
# ruff: noqa: EM101

from __future__ import annotations

import asyncio
import importlib
from contextvars import ContextVar
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

from haystack.lazy_imports import LazyImport

from hayhooks.durable.context import DurableContext, current_durable_context
from hayhooks.durable.models import ExecutionKind, JsonValue

_HAYSTACK_ERROR = "Durable execution requires Haystack 3.1+. Install `hayhooks[durable]`."
_MIN_HAYSTACK_VERSION = (3, 1)
_MAX_HAYSTACK_VERSION = (4, 0)
_AGENT_FINAL = "final"
_AGENT_PRIVATE_STATE = frozenset(("continue_run", "tools", "hook_context", "context_tokens"))
_active_agent_exit_conditions: ContextVar[tuple[str, ...] | None] = ContextVar(
    "hayhooks_durable_agent_exit_conditions", default=None
)

with LazyImport(_HAYSTACK_ERROR) as _haystack_import:
    from haystack import Pipeline
    from haystack.components.agents import Agent
    from haystack.components.agents.state import State
    from haystack.core.errors import BreakpointException, PipelineRuntimeError
    from haystack.dataclasses import ChatMessage
    from haystack.dataclasses.breakpoints import Breakpoint, PipelineSnapshot

    FunctionHook = importlib.import_module("haystack.hooks").FunctionHook
    _serialization = cast(Any, importlib.import_module("haystack.core.serialization"))
    _allow_deserialization_module = _serialization.allow_deserialization_module


def _agent_checkpoint(state: State, *, final: bool = False) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "state": cast(Any, state).to_dict(skip_keys=["tools", "hook_context"]),
            "phase": _AGENT_FINAL if final else None,
        },
    )


def _restore_agent_state(context: DurableContext, state: State) -> None:
    checkpoint = context._adapter_checkpoint
    state_payload = checkpoint.get("state") if isinstance(checkpoint, dict) else None
    if not isinstance(state_payload, dict):
        return
    restored = State.from_dict(state_payload)
    live = {key: state.data.get(key) for key in ("tools", "hook_context")}
    state.data.clear()
    state.data.update(restored.data)
    state.data.update({key: value for key, value in live.items() if value is not None})
    resume = context.resume_input
    messages = resume.get("messages") if isinstance(resume, dict) else None
    if isinstance(messages, list):
        state.data.setdefault("messages", []).extend(
            ChatMessage.from_dict(message) for message in messages if isinstance(message, dict)
        )


def _continues_after_tools(state: State) -> bool:
    exit_conditions = _active_agent_exit_conditions.get()
    if exit_conditions is None:
        return False
    if exit_conditions == ("text",):
        return True
    matched = False
    for message in reversed(state.data.get("messages", [])):
        result = message.tool_call_result
        if result is None:
            break
        if result.origin.tool_name not in exit_conditions:
            continue
        if result.error:
            return True
        matched = True
    return not matched


def _restore_before_run(state: State) -> None:
    if context := current_durable_context():
        _restore_agent_state(context, state)


async def _restore_before_run_async(state: State) -> None:
    if context := current_durable_context():
        _restore_agent_state(context, state)


def _check_before_llm(state: State) -> None:
    del state
    if context := current_durable_context():
        context.check_cancelled_sync()


async def _check_before_llm_async(state: State) -> None:
    del state
    if context := current_durable_context():
        await context.check_cancelled()


def _checkpoint_after_tool(state: State) -> None:
    if (context := current_durable_context()) is None:
        return
    if _continues_after_tools(state):
        state.set("step_count", state.data["step_count"] + 1)
        context.report_progress_sync("Agent step checkpoint saved", kind="checkpoint")
        context.checkpoint_sync(_agent_checkpoint(state))
    context.check_cancelled_sync()


async def _checkpoint_after_tool_async(state: State) -> None:
    if (context := current_durable_context()) is None:
        return
    if _continues_after_tools(state):
        state.set("step_count", state.data["step_count"] + 1)
        await context.report_progress("Agent step checkpoint saved", kind="checkpoint")
        await context.checkpoint(_agent_checkpoint(state))
    await context.check_cancelled()


def _checkpoint_on_exit(state: State) -> None:
    if (context := current_durable_context()) is not None and state.data["continue_run"]:
        context.report_progress_sync("Agent continuation checkpoint saved", kind="checkpoint")
        context.checkpoint_sync(_agent_checkpoint(state))


async def _checkpoint_on_exit_async(state: State) -> None:
    if (context := current_durable_context()) is not None and state.data["continue_run"]:
        await context.report_progress("Agent continuation checkpoint saved", kind="checkpoint")
        await context.checkpoint(_agent_checkpoint(state))


def _checkpoint_after_run(state: State) -> None:
    if context := current_durable_context():
        context.report_progress_sync("Agent final checkpoint saved", kind="checkpoint")
        context.checkpoint_sync(_agent_checkpoint(state, final=True))


async def _checkpoint_after_run_async(state: State) -> None:
    if context := current_durable_context():
        await context.report_progress("Agent final checkpoint saved", kind="checkpoint")
        await context.checkpoint(_agent_checkpoint(state, final=True))


class HaystackDurableAdapter:
    """Checkpoint and resume one shared Haystack Pipeline or Agent."""

    def __init__(self, target: object) -> None:
        try:
            _haystack_import.check()
            installed = version("haystack-ai")
            major, minor = (int(part) for part in installed.split(".", maxsplit=2)[:2])
        except (ImportError, PackageNotFoundError, TypeError, ValueError) as error:
            raise RuntimeError(_HAYSTACK_ERROR) from error
        if not _MIN_HAYSTACK_VERSION <= (major, minor) < _MAX_HAYSTACK_VERSION:
            raise RuntimeError(_HAYSTACK_ERROR)
        self.target = cast(Any, target)
        if isinstance(target, Pipeline):
            self.kind = ExecutionKind.PIPELINE
        elif isinstance(target, Agent):
            self.kind = ExecutionKind.AGENT
            _allow_deserialization_module(__name__)
        else:
            raise TypeError("durable execution requires a real Haystack 3.1 Pipeline or Agent")
        if self.kind is ExecutionKind.PIPELINE:
            return
        agent = cast(Any, target)
        hooks = dict(getattr(agent, "hooks", {}) or {})
        if any(
            isinstance(installed, FunctionHook) and installed.function is _restore_before_run
            for installed in hooks.get("before_run", [])
        ):
            return
        hooks["before_run"] = [
            FunctionHook(function=_restore_before_run, async_function=_restore_before_run_async),
            *hooks.get("before_run", []),
        ]
        hooks["before_llm"] = [
            FunctionHook(function=_check_before_llm, async_function=_check_before_llm_async),
            *hooks.get("before_llm", []),
        ]
        hooks["after_tool"] = [
            *hooks.get("after_tool", []),
            FunctionHook(function=_checkpoint_after_tool, async_function=_checkpoint_after_tool_async),
        ]
        hooks["on_exit"] = [
            *hooks.get("on_exit", []),
            FunctionHook(function=_checkpoint_on_exit, async_function=_checkpoint_on_exit_async),
        ]
        hooks["after_run"] = [
            *hooks.get("after_run", []),
            FunctionHook(function=_checkpoint_after_run, async_function=_checkpoint_after_run_async),
        ]
        agent.hooks = hooks

    def run_pipeline(  # noqa: C901
        self,
        context: DurableContext,
        data: dict[str, Any],
        *,
        checkpoint_at: str | None = None,
    ) -> dict[str, Any]:
        """Run a Pipeline, persisting the requested component boundary."""
        if self.kind is not ExecutionKind.PIPELINE:
            raise TypeError("run_pipeline requires a Haystack Pipeline")
        context._require_owned()
        checkpoint = context._adapter_checkpoint
        if checkpoint is not None and not isinstance(checkpoint, dict):
            raise TypeError("persisted Pipeline checkpoint must be an object")
        if checkpoint_at is not None and not isinstance(checkpoint_at, str):
            raise TypeError("checkpoint_at must name one Pipeline component")
        snapshot = PipelineSnapshot.from_dict(checkpoint) if checkpoint is not None else None
        boundary = checkpoint_at
        if snapshot is not None and boundary is not None:
            visits = snapshot.pipeline_state.component_visits
            previous = cast(Any, snapshot.break_point)
            if visits.get(boundary, 0) or (boundary == previous.component_name and previous.visit_count == 0):
                boundary = None

        next_data = data if snapshot is None else {}
        try:
            if boundary is not None:
                try:
                    return cast(
                        dict[str, Any],
                        self.target.run(
                            data=next_data,
                            pipeline_snapshot=snapshot,
                            break_point=Breakpoint(component_name=boundary),
                        ),
                    )
                except BreakpointException as error:
                    if error.pipeline_snapshot is None:
                        raise RuntimeError("Haystack breakpoint did not provide a PipelineSnapshot") from error
                    snapshot = error.pipeline_snapshot
                    context.report_progress_sync(
                        f"Checkpoint saved before pipeline component '{boundary}'",
                        kind="checkpoint",
                    )
                    context.checkpoint_sync(cast(JsonValue, snapshot.to_dict()))
                    next_data = {}
            return cast(dict[str, Any], self.target.run(data=next_data, pipeline_snapshot=snapshot))
        except PipelineRuntimeError as error:
            if error.pipeline_snapshot is not None:
                context.checkpoint_sync(cast(JsonValue, error.pipeline_snapshot.to_dict()))
            raise

    async def run_pipeline_async(
        self,
        context: DurableContext,
        data: dict[str, Any],
        *,
        checkpoint_at: str | None = None,
    ) -> dict[str, Any]:
        """Run the synchronous Pipeline in a cancellation-shielded thread."""
        context._require_owned()
        task = asyncio.create_task(asyncio.to_thread(self.run_pipeline, context, data, checkpoint_at=checkpoint_at))
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    return task.result()

    def run_agent(self, context: DurableContext, *, messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        """Run or recover a synchronous Agent execution."""
        if self.kind is not ExecutionKind.AGENT:
            raise TypeError("run_agent requires a Haystack Agent")
        context._require_owned()
        if (result := self._final_agent_result(context)) is not None:
            return result
        token = _active_agent_exit_conditions.set(tuple(self.target.exit_conditions))
        try:
            return cast(dict[str, Any], self.target.run(messages=messages, **kwargs))
        finally:
            _active_agent_exit_conditions.reset(token)

    async def run_agent_async(self, context: DurableContext, *, messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        """Run or recover an asynchronous Agent execution."""
        if self.kind is not ExecutionKind.AGENT:
            raise TypeError("run_agent_async requires a Haystack Agent")
        context._require_owned()
        if (result := self._final_agent_result(context)) is not None:
            return result
        token = _active_agent_exit_conditions.set(tuple(self.target.exit_conditions))
        try:
            return cast(dict[str, Any], await self.target.run_async(messages=messages, **kwargs))
        finally:
            _active_agent_exit_conditions.reset(token)

    def _final_agent_result(self, context: DurableContext) -> dict[str, Any] | None:
        checkpoint = context._adapter_checkpoint
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("phase") != _AGENT_FINAL
            or not isinstance(checkpoint.get("state"), dict)
        ):
            return None
        state_payload = cast(dict[str, Any], checkpoint["state"])
        result = {
            key: value for key, value in State.from_dict(state_payload).data.items() if key not in _AGENT_PRIVATE_STATE
        }
        if messages := result.get("messages"):
            result["last_message"] = messages[-1]
        return result


__all__ = ["HaystackDurableAdapter"]
