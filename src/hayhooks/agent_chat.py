from __future__ import annotations

import json
from collections import Counter
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Literal

import anyio
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk

from hayhooks.events import PipelineEvent
from hayhooks.server.pipelines.utils import async_streaming_generator


@dataclass
class _Call:
    call_id: str = ""
    name: str = ""
    parts: list[str] = field(default_factory=list)


def _call_key(provider_id: str | None, name: str, arguments: str | dict[str, Any]) -> tuple[str, str, str] | None:
    if not provider_id or not name:
        return None
    try:
        values = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(values, dict):
            return None
        return provider_id, name, json.dumps(values, sort_keys=True)
    except (ValueError, TypeError, RecursionError):
        return None


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _completion_reason(
    agent: Agent, result: dict[str, Any], incoming: list[ChatMessage], last_model_finish: str | None
) -> Literal["stop", "length", "content_filter"] | None:
    reason = result.get("exit_reason")
    if reason in {"max_agent_steps", "length", "content_filter"}:
        return "content_filter" if reason == "content_filter" else "length"

    incoming_ids = {id(message) for message in incoming}
    messages = [message for message in result.get("messages", []) if id(message) not in incoming_ids]
    final_tools = []
    final_replies = []
    for message in reversed(messages):
        if not isinstance(message, ChatMessage):
            break
        if message.is_from(ChatRole.TOOL) and not final_replies:
            final_tools.extend(message.tool_call_results)
        elif message.is_from(ChatRole.ASSISTANT):
            final_replies.append(message)
        else:
            break
    final_replies.reverse()
    finish = final_replies[-1].meta.get("finish_reason", last_model_finish) if final_replies else last_model_finish
    if finish in {"length", "content_filter"}:
        return finish
    if not final_replies:
        return "stop" if reason else None
    # Haystack v2 has no exit_reason. Reproduce its final generation's public exit contract.
    if (
        reason
        or not agent.tools
        or (final_replies[-1].text and not any(message.tool_calls for message in final_replies))
    ):
        return "stop"
    exit_calls = {
        call.tool_name
        for message in final_replies
        for call in message.tool_calls
        if call.tool_name in agent.exit_conditions
    }
    return (
        "stop"
        if exit_calls and not any(tool.error and tool.origin.tool_name in exit_calls for tool in final_tools)
        else "length"
    )


async def _agent_events(  # noqa: C901, PLR0912
    agent: Agent, messages: list[ChatMessage], agent_inputs: dict[str, Any]
) -> AsyncGenerator[StreamingChunk | PipelineEvent, None]:
    run_result: dict[str, Any] = {}

    def capture_result(result: dict[str, Any]) -> None:
        nonlocal run_result
        run_result = result

    source = async_streaming_generator(
        agent, pipeline_run_args={"messages": messages, **agent_inputs}, on_pipeline_end=capture_result
    )
    calls: dict[int, _Call] = {}
    pending_calls: dict[tuple[str, str, str], str] = {}
    call_numbers = count(1)
    last_model_finish_reason = None
    after_tool_result = False
    try:
        async for item in source:
            if not isinstance(item, StreamingChunk):
                if isinstance(item, PipelineEvent):
                    yield item
                continue
            if (
                after_tool_result
                and not item.tool_call_result
                and (item.content or item.tool_calls or item.finish_reason not in {None, "tool_call_results"})
            ):
                last_model_finish_reason = None
                after_tool_result = False
            if item.finish_reason and item.finish_reason != "tool_call_results" and not item.tool_call_result:
                last_model_finish_reason = item.finish_reason
            for delta in item.tool_calls or []:
                call = calls.setdefault(delta.index, _Call())
                call.call_id = delta.id or call.call_id
                call.name = delta.tool_name or call.name
                if delta.arguments:
                    call.parts.append(delta.arguments)
            if item.finish_reason == "tool_calls":
                call_id_counts = Counter(call.call_id for call in calls.values())
                for call in calls.values():
                    arguments = "".join(call.parts)
                    if call_id_counts[call.call_id] == 1 and (key := _call_key(call.call_id, call.name, arguments)):
                        call_id = f"agent:{next(call_numbers)}:{call.call_id}"
                        pending_calls[key] = call_id
                        yield PipelineEvent(
                            type="hayhooks.tool_call",
                            data={
                                "call_id": call_id,
                                "actor": "Agent",
                                "name": call.name,
                                "arguments": arguments,
                            },
                        )
                calls.clear()
            if result := item.tool_call_result:
                calls.clear()
                after_tool_result = True
                origin = result.origin
                key = _call_key(origin.id, origin.tool_name, origin.arguments)
                if key is None or (call_id := pending_calls.pop(key, None)) is None:
                    call_id = f"agent:{next(call_numbers)}:{origin.id or 'result'}"
                    yield PipelineEvent(
                        type="hayhooks.tool_call",
                        data={
                            "call_id": call_id,
                            "actor": "Agent",
                            "name": origin.tool_name,
                            "arguments": _text(origin.arguments),
                        },
                    )
                yield PipelineEvent(
                    type="hayhooks.tool_result",
                    data={
                        "call_id": call_id,
                        "result": _text(result.result),
                        "error": bool(result.error),
                    },
                )
            if item.content:
                yield StreamingChunk(content=item.content)
        if finish := _completion_reason(agent, run_result, messages, last_model_finish_reason):
            yield StreamingChunk(content="", finish_reason=finish)
    finally:
        with anyio.CancelScope(shield=True):
            await source.aclose()


def stream_agent_chat(
    agent: Agent, messages: Sequence[ChatMessage], **agent_inputs: Any
) -> AsyncGenerator[StreamingChunk | PipelineEvent, None]:
    """
    Stream one invocation of the configured Haystack Agent as chat events.

    Pass state inputs as keyword arguments. Configure tools on the Agent or its
    Toolsets. This helper preserves the Agent configuration and closes the stream
    on disconnect. Synchronous tools already running may continue to completion.
    """
    if not isinstance(agent, Agent):
        message = "stream_agent_chat requires a Haystack Agent assigned to self.pipeline in setup()"
        raise TypeError(message)
    if "streaming_callback" in agent_inputs:
        message = "stream_agent_chat owns streaming_callback"
        raise ValueError(message)
    return _agent_events(agent, list(messages), agent_inputs)
