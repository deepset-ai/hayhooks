import json
from types import ModuleType

import httpx
import pytest
from fastapi import FastAPI
from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ReasoningContent, StreamingChunk, ToolCall
from haystack.dataclasses.streaming_chunk import ToolCallDelta
from haystack.tools import Tool

from hayhooks import BasePipelineWrapper, stream_agent_chat
from hayhooks.chat.client import ChatClient, ChatError
from hayhooks.chat.events import ReasoningSummary, ResponseComplete, ResponseDelta, ToolFinished, ToolStarted
from hayhooks.server.pipelines.registry import _PipelineRegistry
from hayhooks.server.routers.openai import create_openai_router
from hayhooks.server.utils.module_loader import create_pipeline_wrapper_instance


@component
class CompletionGenerator:
    def __init__(self, replies: list[ChatMessage], leading_chunks: tuple[StreamingChunk, ...] = ()):
        self.replies = replies
        self.step = 0
        self.leading_chunks = leading_chunks

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools=None, streaming_callback=None):
        reply = self.replies[self.step]
        self.step += 1
        return {"replies": [reply]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(self, messages: list[ChatMessage], tools=None, streaming_callback=None):
        if self.step == 0:
            for chunk in self.leading_chunks:
                await streaming_callback(chunk)
        result = self.run(messages, tools, streaming_callback)
        reply = result["replies"][0]
        if reply.text:
            await streaming_callback(StreamingChunk(content=reply.text))
        for index, call in enumerate(reply.tool_calls):
            await streaming_callback(
                StreamingChunk(
                    content="",
                    index=index,
                    tool_calls=[
                        ToolCallDelta(
                            index=index,
                            id=call.id,
                            tool_name=call.tool_name,
                            arguments=json.dumps(call.arguments),
                        )
                    ],
                )
            )
        if reply.meta.get("stream_finish", True):
            await streaming_callback(StreamingChunk(content="", finish_reason=reply.meta.get("finish_reason", "stop")))
        await streaming_callback(StreamingChunk(content="", meta={"usage": {"total_tokens": 12}}))
        return result


class CompletionWrapper(BasePipelineWrapper):
    def setup(self):
        pass

    async def run_chat_completion_async(self, model, messages, body):
        return stream_agent_chat(self.pipeline, [ChatMessage.from_user(messages[-1]["content"])])


@pytest.fixture
def lookup_tool():
    return Tool(
        name="lookup",
        description="Look up a value",
        parameters={"type": "object", "properties": {}},
        function=lambda: "Found",
    )


@pytest.fixture
def chat_client(monkeypatch):
    original = httpx.AsyncClient

    def create(agent):
        module = ModuleType("completion_wrapper")
        module.PipelineWrapper = CompletionWrapper
        wrapper = create_pipeline_wrapper_instance(module)
        wrapper.pipeline = agent
        registry = _PipelineRegistry()
        registry.add("completion", wrapper)
        app = FastAPI()
        app.include_router(create_openai_router(registry))
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.ASGITransport(app), **kwargs)
        )
        return ChatClient("http://test", "completion")

    return create


@pytest.mark.parametrize("reason, cause", [("length", "output limit"), ("content_filter", "filtered")])
@pytest.mark.parametrize("tool_exit", [False, True])
async def test_terminal_model_failure_retains_text_without_success_or_history(
    chat_client, lookup_tool, reason, cause, tool_exit
):
    generator = CompletionGenerator(
        [
            ChatMessage.from_assistant(
                "Partial",
                tool_calls=[ToolCall(tool_name="lookup", arguments={}, id="lookup-1")] if tool_exit else None,
                meta={"finish_reason": reason},
            )
        ]
    )
    client = chat_client(
        Agent(
            chat_generator=generator,
            tools=[lookup_tool] if tool_exit else None,
            exit_conditions=["lookup"] if tool_exit else None,
        )
    )
    history = [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Complete"}]
    client.history = history.copy()
    events = []

    with pytest.raises(ChatError, match=cause) as error:
        await client.run("Hi", lambda _: None, emit=events.append)

    assert error.value.kind == "provider"
    assert [event.text for event in events if isinstance(event, ResponseDelta)] == ["Partial"]
    if tool_exit:
        assert [event.result for event in events if isinstance(event, ToolFinished)] == ["Found"]
    assert not any(isinstance(event, ResponseComplete) and event.final for event in events)
    assert client.history == history


@pytest.mark.parametrize("first_reason", ["tool_calls", "length", "content_filter"])
async def test_later_model_success_finishes_after_tool_result(chat_client, lookup_tool, first_reason):
    generator = CompletionGenerator(
        [
            ChatMessage.from_assistant(
                "Checking ",
                tool_calls=[ToolCall(tool_name="lookup", arguments={}, id="lookup-1")],
                meta={"finish_reason": first_reason},
            ),
            ChatMessage.from_assistant("Done", meta={"finish_reason": "stop"}),
        ]
    )
    client = chat_client(Agent(chat_generator=generator, tools=[lookup_tool]))
    events = []

    assert await client.run("Hi", lambda _: None, emit=events.append) == "Checking Done"

    assert [type(event) for event in events] == [
        ResponseDelta,
        ToolStarted,
        ToolFinished,
        ResponseDelta,
        ResponseComplete,
    ]
    assert events[1].call_id == events[2].call_id
    assert events[1].call_id.endswith(":lookup-1")
    assert events[2].result == "Found"
    assert events[-1].final is True
    assert client.history == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Checking Done"}]


@pytest.mark.parametrize("tool_exit", [False, True])
async def test_success_without_assistant_text_commits_history(chat_client, lookup_tool, tool_exit):
    reply = ChatMessage.from_assistant(
        tool_calls=[ToolCall(tool_name="lookup", arguments={}, id="lookup-1")] if tool_exit else None,
        meta={"finish_reason": "tool_calls" if tool_exit else "stop"},
    )
    client = chat_client(
        Agent(
            chat_generator=CompletionGenerator([reply]),
            tools=[lookup_tool] if tool_exit else None,
            exit_conditions=["lookup"] if tool_exit else None,
            max_agent_steps=1,
        )
    )
    events = []

    assert await client.run("Hi", lambda _: None, emit=events.append) == ""

    assert sum(isinstance(event, ResponseComplete) and event.final for event in events) == 1
    assert client.history == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}]
    if tool_exit:
        assert [event.result for event in events if isinstance(event, ToolFinished)] == ["Found"]


@pytest.mark.parametrize("preamble", [None, "Checking "])
async def test_agent_step_limit_preserves_work_without_committing_history(chat_client, lookup_tool, preamble):
    client = chat_client(
        Agent(
            chat_generator=CompletionGenerator(
                [
                    ChatMessage.from_assistant(
                        preamble,
                        tool_calls=[ToolCall(tool_name="lookup", arguments={}, id="lookup-1")],
                        meta={"finish_reason": "tool_calls"},
                    )
                ]
            ),
            tools=[lookup_tool],
            max_agent_steps=1,
        )
    )
    events = []

    with pytest.raises(ChatError, match="max_agent_steps") as error:
        await client.run("Hi", lambda _: None, emit=events.append)

    assert error.value.kind == "provider"
    assert "step limit" in str(error.value)
    assert "before" in str(error.value) and "retry" in str(error.value)
    assert [event.text for event in events if isinstance(event, ResponseDelta)] == ([preamble] if preamble else [])
    assert [event.result for event in events if isinstance(event, ToolFinished)] == ["Found"]
    assert not any(isinstance(event, ResponseComplete) and event.final for event in events)
    assert client.history == []


@pytest.mark.parametrize("exit_conditions", [["text"], ["lookup"], ["text", "lookup"]])
async def test_text_success_on_last_available_step_commits_history(chat_client, lookup_tool, exit_conditions):
    client = chat_client(
        Agent(
            chat_generator=CompletionGenerator([ChatMessage.from_assistant("Done", meta={"finish_reason": "stop"})]),
            tools=[lookup_tool],
            exit_conditions=exit_conditions,
            max_agent_steps=1,
        )
    )
    events = []

    assert await client.run("Hi", lambda _: None, emit=events.append) == "Done"
    assert events[-1] == ResponseComplete("answer", "Agent", "Done", final=True)
    assert client.history[-1] == {"role": "assistant", "content": "Done"}


@pytest.mark.parametrize("historical_success", [False, True])
async def test_failed_exit_tool_does_not_make_step_limit_success(chat_client, lookup_tool, historical_success):
    def fail():
        message = "Could not finish"
        raise ValueError(message)

    failing_tool = Tool(
        name="fail",
        description="Fail the exit",
        parameters={"type": "object", "properties": {}},
        function=fail,
    )
    calls = [ToolCall(tool_name="fail", arguments={}, id="fail-1")]
    if historical_success:
        calls.insert(0, ToolCall(tool_name="lookup", arguments={}, id="lookup-1"))
    replies = [ChatMessage.from_assistant(tool_calls=calls, meta={"finish_reason": "tool_calls"})]
    if historical_success:
        replies.append(
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="fail", arguments={}, id="fail-2")],
                meta={"finish_reason": "tool_calls"},
            )
        )
    client = chat_client(
        Agent(
            chat_generator=CompletionGenerator(replies),
            tools=[lookup_tool, failing_tool],
            exit_conditions=["lookup", "fail"],
            raise_on_tool_invocation_failure=False,
            max_agent_steps=len(replies),
        )
    )
    events = []

    with pytest.raises(ChatError, match="step limit"):
        await client.run("Hi", lambda _: None, emit=events.append)

    results = [event for event in events if isinstance(event, ToolFinished)]
    assert any(event.error for event in results)
    if historical_success:
        assert any(event.result == "Found" and not event.error for event in results)
    assert not any(isinstance(event, ResponseComplete) and event.final for event in events)
    assert client.history == []


@pytest.mark.parametrize("exit_conditions", [["lookup"], ["text", "lookup"]])
async def test_parallel_exit_ignores_failure_of_non_exit_tool(chat_client, lookup_tool, exit_conditions):
    def fail():
        message = "Other tool failed"
        raise ValueError(message)

    failing_tool = Tool(
        name="other",
        description="Fail another tool",
        parameters={"type": "object", "properties": {}},
        function=fail,
    )
    reply = ChatMessage.from_assistant(
        tool_calls=[
            ToolCall(tool_name="other", arguments={}, id="other-1"),
            ToolCall(tool_name="lookup", arguments={}, id="lookup-1"),
        ],
        meta={"finish_reason": "tool_calls"},
    )
    client = chat_client(
        Agent(
            chat_generator=CompletionGenerator([reply]),
            tools=[lookup_tool, failing_tool],
            exit_conditions=exit_conditions,
            raise_on_tool_invocation_failure=False,
            max_agent_steps=1,
        )
    )
    events = []

    assert await client.run("Hi", lambda _: None, emit=events.append) == ""
    assert [event.error for event in events if isinstance(event, ToolFinished)] == [True, False]
    assert events[-1] == ResponseComplete("answer", "Agent", "", final=True)


async def test_recovered_model_limit_without_final_stream_marker_succeeds(chat_client, lookup_tool):
    replies = [
        ChatMessage.from_assistant(
            "Checking ",
            tool_calls=[ToolCall(tool_name="lookup", arguments={}, id="lookup-1")],
            meta={"finish_reason": "length"},
        ),
        ChatMessage.from_assistant("Done", meta={"stream_finish": False}),
    ]
    client = chat_client(Agent(chat_generator=CompletionGenerator(replies), tools=[lookup_tool]))
    events = []

    assert await client.run("Hi", lambda _: None, emit=events.append) == "Checking Done"
    assert events[-1] == ResponseComplete("answer", "Agent", "Checking Done", final=True)
    assert client.history[-1] == {"role": "assistant", "content": "Checking Done"}


async def test_responses_reasoning_summary_is_display_only(chat_client, lookup_tool):
    summary = "I'll look up the value first."
    generator = CompletionGenerator(
        [
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="lookup", arguments={}, id="lookup-1")],
                meta={"finish_reason": "tool_calls"},
            ),
            ChatMessage.from_assistant("Done", meta={"finish_reason": "stop"}),
        ],
        leading_chunks=(
            StreamingChunk(
                content="",
                index=0,
                reasoning=ReasoningContent(
                    reasoning_text="private trace",
                    extra={"type": "response.reasoning_text.delta", "encrypted_content": "opaque"},
                ),
            ),
            StreamingChunk(content="", meta={"type": "response.reasoning_text.done", "text": "private trace"}),
            StreamingChunk(content="", meta={"type": "response.reasoning_summary_text.done", "text": ""}),
            StreamingChunk(
                content="",
                meta={"type": "response.reasoning_summary_text.done", "text": summary, "item_id": "rs_provider"},
            ),
        ),
    )
    client = chat_client(Agent(chat_generator=generator, tools=[lookup_tool]))
    events = []

    assert await client.run("Hi", lambda _: None, emit=events.append) == "Done"
    assert [type(event) for event in events] == [
        ReasoningSummary,
        ToolStarted,
        ToolFinished,
        ResponseDelta,
        ResponseComplete,
    ]
    assert events[0] == ReasoningSummary(summary)
    assert client.history == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Done"}]


async def test_v3_custom_hook_stop_can_complete_without_model_text(chat_client):
    hooks = pytest.importorskip("haystack.hooks")
    from haystack.components.agents.state import State

    @hooks.hook
    def finish(state: State) -> None:
        state.set("stop_run", "custom_success")

    agent = Agent(chat_generator=CompletionGenerator([]), hooks={"before_llm": [finish]})
    if "stop_run" not in agent.resolved_state_schema:
        pytest.skip("Haystack does not support stopping an agent from a before_llm hook")

    client = chat_client(agent)
    events = []

    assert await client.run("Hi", lambda _: None, emit=events.append) == ""
    assert events == [ResponseComplete("answer", "Agent", "", final=True)]
    assert client.history[-1] == {"role": "assistant", "content": ""}

    agent = Agent(chat_generator=CompletionGenerator([]), hooks={"before_llm": [finish]})
    output = [
        item
        async for item in stream_agent_chat(
            agent,
            [
                ChatMessage.from_assistant("Old truncated response", meta={"finish_reason": "length"}),
                ChatMessage.from_user("Stop now"),
            ],
        )
    ]
    assert len(output) == 1 and output[0].finish_reason == "stop"


async def test_v3_hook_continuation_cannot_turn_step_limit_into_tool_success(chat_client, lookup_tool):
    hooks = pytest.importorskip("haystack.hooks")
    from haystack.components.agents.state import State

    @hooks.hook
    def continue_run(state: State) -> None:
        state.set("continue_run", True)

    reply = ChatMessage.from_assistant(
        tool_calls=[ToolCall(tool_name="lookup", arguments={}, id="lookup-1")],
        meta={"finish_reason": "tool_calls"},
    )
    client = chat_client(
        Agent(
            chat_generator=CompletionGenerator([reply]),
            tools=[lookup_tool],
            exit_conditions=["lookup"],
            max_agent_steps=1,
            hooks={"on_exit": [continue_run]},
        )
    )
    events = []

    with pytest.raises(ChatError, match="step limit"):
        await client.run("Hi", lambda _: None, emit=events.append)

    assert [event.result for event in events if isinstance(event, ToolFinished)] == ["Found"]
    assert not any(isinstance(event, ResponseComplete) and event.final for event in events)
    assert client.history == []
