"""
Haystack 3 adapters used by :class:`hayhooks.execution.DurableContext`.

The adapters use only Haystack's public PipelineSnapshot, Agent, State, and
hook APIs.  The imports intentionally stay lazy so the base Hayhooks install
continues to support Haystack 2 for non-durable deployments.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import types
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, cast

from haystack.lazy_imports import LazyImport

from hayhooks.execution import (
    DurableContext,
    ExecutionCheckpoint,
    ExecutionKind,
    RetryableExecutionError,
    validate_json,
)

_HAYSTACK_V3_ERROR = (
    "Durable execution requires Haystack 3. Install `hayhooks[durable]` in the durable server environment."
)


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


def definition_revision(pipeline: Any, override: str | None = None, *, wrapper: Any | None = None) -> str:
    """Return a stable configuration-and-code revision, or use an explicit boundary."""
    if override:
        return override
    serializer = getattr(pipeline, "to_dict", None)
    if not callable(serializer):
        msg = "The deployed Agent or Pipeline cannot be fingerprinted; set DurableOptions(revision='...')."
        raise ValueError(msg)
    try:
        definition = serializer()
        code = _definition_code(definition, wrapper=wrapper)
        encoded = json.dumps(
            {"definition": definition, "code": code},
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
    except (TypeError, ValueError) as error:
        msg = "The deployed Agent or Pipeline cannot be fingerprinted; set DurableOptions(revision='...')."
        raise ValueError(msg) from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _definition_code(definition: Any, *, wrapper: Any | None) -> dict[str, str]:
    """Fingerprint wrapper and serialized component implementations without filesystem paths."""
    implementations: dict[str, str] = {}
    if wrapper is not None:
        wrapper_type = type(wrapper)
        implementations[f"wrapper:{wrapper_type.__module__}.{wrapper_type.__qualname__}"] = _type_code(wrapper_type)

    for import_path in sorted(_serialized_types(definition)):
        implementation = _import_type(import_path)
        if implementation is None:
            msg = (
                f"The component implementation '{import_path}' cannot be fingerprinted; "
                "set DurableOptions(revision='...')."
            )
            raise ValueError(msg)
        implementations[import_path] = _type_code(implementation)
    return implementations


def _serialized_types(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "type" and isinstance(item, str) and "." in item:
                found.add(item)
            found.update(_serialized_types(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_serialized_types(item))
    return found


def _import_type(import_path: str) -> type[Any] | None:
    module_name, _, attribute_path = import_path.rpartition(".")
    if not module_name:
        return None
    try:
        module = __import__(module_name, fromlist=[attribute_path])
        value: Any = module
        for part in attribute_path.split("."):
            value = getattr(value, part)
    except (AttributeError, ImportError):
        return None
    return value if isinstance(value, type) else None


def _type_code(value: type[Any]) -> str:
    """Hash source when available, with a stable bytecode fallback for dynamic classes."""
    sources: dict[str, str] = {}
    with suppress(OSError, TypeError):
        sources["class"] = inspect.getsource(value)
    module = inspect.getmodule(value)
    if module is not None:
        with suppress(OSError, TypeError):
            sources["module"] = inspect.getsource(module)
    if sources:
        payload: Any = sources
    else:
        methods = {}
        for name, member in sorted(vars(value).items()):
            code = _callable_code(member)
            if code is not None:
                methods[name] = code
        if not methods:
            msg = (
                f"The implementation for {value.__module__}.{value.__qualname__} cannot be fingerprinted; "
                "set DurableOptions(revision='...')."
            )
            raise ValueError(msg) from None
        payload = methods
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _callable_code(value: Any) -> Any | None:
    if isinstance(value, staticmethod | classmethod):
        value = value.__func__
    elif isinstance(value, property):
        value = value.fget
    code = getattr(value, "__code__", None)
    if not isinstance(code, types.CodeType):
        return None
    return _code_payload(code)


def _code_payload(code: types.CodeType) -> dict[str, Any]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "consts": [
            _code_payload(constant) if isinstance(constant, types.CodeType) else repr(constant)
            for constant in code.co_consts
        ],
        "names": code.co_names,
        "varnames": code.co_varnames,
    }


def _json_default(value: Any) -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return converter()
    msg = f"{type(value).__name__} is not serializable"
    raise TypeError(msg)


class HaystackDurableAdapter:
    """Bind a validated Haystack 3 Pipeline or Agent to execution contexts."""

    def __init__(self, pipeline: Any, kind: ExecutionKind) -> None:
        require_haystack_v3()
        self.pipeline = pipeline
        self.kind = kind
        self._agent_checkpoint_installed = False
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

    def run_pipeline(  # noqa: C901
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
                checkpoint = ExecutionCheckpoint(
                    ExecutionKind.PIPELINE,
                    {
                        "snapshot": validate_json(
                            snapshot.to_dict(), limit=context.record.max_record_bytes, label="snapshot"
                        )
                    },
                )
                context._sync_await(context.checkpoint(checkpoint))
                context.report_progress_sync(
                    f"Checkpoint saved before pipeline component '{component_name}'", kind="checkpoint"
                )
                next_data = {}
            except PipelineRuntimeError as error:
                if error.pipeline_snapshot is not None:
                    checkpoint = ExecutionCheckpoint(
                        ExecutionKind.PIPELINE,
                        {
                            "snapshot": validate_json(
                                error.pipeline_snapshot.to_dict(),
                                limit=context.record.max_record_bytes,
                                label="snapshot",
                            )
                        },
                    )
                    context._sync_await(context.checkpoint(checkpoint))
                raise

        try:
            return cast(dict[str, Any], self.pipeline.run(data=next_data, pipeline_snapshot=snapshot))
        except PipelineRuntimeError as error:
            if error.pipeline_snapshot is not None:
                checkpoint = ExecutionCheckpoint(
                    ExecutionKind.PIPELINE,
                    {
                        "snapshot": validate_json(
                            error.pipeline_snapshot.to_dict(),
                            limit=context.record.max_record_bytes,
                            label="snapshot",
                        )
                    },
                )
                context._sync_await(context.checkpoint(checkpoint))
            raise

    async def run_agent_async(self, context: DurableContext, *, messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        if self.kind is not ExecutionKind.AGENT:
            msg = "run_agent_async is available only when self.pipeline is a Haystack Agent"
            raise TypeError(msg)
        method = getattr(self.pipeline, "run_async", None)
        if callable(method):
            return cast(dict[str, Any], await method(messages=messages, **kwargs))
        return cast(
            dict[str, Any],
            await _run_fenced_thread(self.run_agent, context, messages=messages, **kwargs),
        )

    def run_agent(self, _context: DurableContext, *, messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        if self.kind is not ExecutionKind.AGENT:
            msg = "run_agent is available only when self.pipeline is a Haystack Agent"
            raise TypeError(msg)
        return cast(dict[str, Any], self.pipeline.run(messages=messages, **kwargs))

    def _install_agent_checkpoint_hooks(self) -> None:
        """Install once; hooks select the active execution through ContextVar."""
        if self._agent_checkpoint_installed:
            return

        def before_llm(state: State) -> None:
            context = _durable_context()
            _restore_agent_state(context, state)
            context.check_cancelled_sync()

        async def before_llm_async(state: State) -> None:
            context = _durable_context()
            _restore_agent_state(context, state)
            await context.check_cancelled()

        def checkpoint_after_tool(state: State) -> None:
            context = _durable_context()
            context._sync_await(_checkpoint_agent_state(context, state))
            context.check_cancelled_sync()

        async def checkpoint_after_tool_async(state: State) -> None:
            context = _durable_context()
            await _checkpoint_agent_state(context, state)
            await context.check_cancelled()

        def checkpoint_on_exit(state: State) -> None:
            context = _durable_context()
            context._sync_await(_checkpoint_agent_state(context, state))

        async def checkpoint_on_exit_async(state: State) -> None:
            context = _durable_context()
            await _checkpoint_agent_state(context, state)

        # The module uses postponed annotations, while Haystack validates hook
        # signatures with ``inspect.signature`` rather than resolving hints.
        for function in (
            before_llm,
            before_llm_async,
            checkpoint_after_tool,
            checkpoint_after_tool_async,
            checkpoint_on_exit,
            checkpoint_on_exit_async,
        ):
            function.__annotations__["state"] = State

        hooks = dict(getattr(self.pipeline, "hooks", {}) or {})
        hooks["before_llm"] = [
            FunctionHook(function=before_llm, async_function=before_llm_async),
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
        self.pipeline.hooks = hooks
        self._agent_checkpoint_installed = True


def _durable_context() -> DurableContext:
    from hayhooks.execution import get_current_durable_context

    context = get_current_durable_context()
    if context is None:
        msg = "Haystack durable hook ran outside a durable execution"
        raise RuntimeError(msg)
    return context


def _checkpoint_data(state: Any, context: DurableContext) -> dict[str, Any]:
    """Exclude live resources from State's otherwise public serialization."""
    payload = state.to_dict()
    data = dict(payload.get("data", {}))
    schema = dict(payload.get("schema", {}))
    for key in ("tools", "hook_context"):
        data.pop(key, None)
        schema.pop(key, None)
    return cast(
        dict[str, Any],
        validate_json({"schema": schema, "data": data}, limit=context.record.max_record_bytes, label="Agent state"),
    )


async def _checkpoint_agent_state(context: DurableContext, state: Any) -> None:
    await context.checkpoint(ExecutionCheckpoint(ExecutionKind.AGENT, _checkpoint_data(state, context)))
    await context.report_progress("Agent step checkpoint saved", kind="checkpoint")


def _restore_agent_state(context: DurableContext, state: Any) -> None:
    """Restore a recovered State once, retaining fresh per-run live resources."""
    checkpoint = context.record.checkpoint
    if checkpoint is None or checkpoint.kind is not ExecutionKind.AGENT:
        return
    restored_flag = "_hayhooks_durable_restored"
    if state.get(restored_flag, False):
        return
    restored = State.from_dict(cast(dict[str, Any], checkpoint.data))
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
    state.data[restored_flag] = True


def execution_kind(pipeline: Any) -> ExecutionKind:
    """Classify a real Haystack 3 Pipeline or Agent behind the lazy boundary."""
    require_haystack_v3()
    if isinstance(pipeline, Pipeline):
        return ExecutionKind.PIPELINE
    if isinstance(pipeline, Agent):
        return ExecutionKind.AGENT
    msg = "Durable wrappers must set self.pipeline to a real Haystack 3 Pipeline or Agent"
    raise TypeError(msg)


__all__ = ["HaystackDurableAdapter", "definition_revision", "execution_kind", "require_haystack_v3"]
