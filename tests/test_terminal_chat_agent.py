import asyncio
import contextlib
import json
import socket
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import ModuleType

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall, ToolCallResult
from haystack.dataclasses.streaming_chunk import ToolCallDelta
from haystack.tools import Tool, Toolset

from examples.pipeline_wrappers.terminal_chat_agent import pipeline_wrapper
from hayhooks import BasePipelineWrapper, Pipeline, PipelineEvent, stream_agent_chat
from hayhooks.server.pipelines.registry import _PipelineRegistry
from hayhooks.server.routers.openai import create_openai_router
from hayhooks.server.utils.deploy_utils import prepare_pipeline_files
from hayhooks.server.utils.module_loader import create_pipeline_wrapper_instance
from tests.test_agent_chat_completion import CompletionGenerator


@component
class ToolGenerator:
    def __init__(self, tool_name="echo", arguments=None):
        self.tool_name = tool_name
        self.arguments = arguments or {}

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools=None, streaming_callback=None):
        return {"replies": [ChatMessage.from_assistant("Done")]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(self, messages: list[ChatMessage], tools=None, streaming_callback=None):
        if any(message.tool_call_results for message in messages):
            await streaming_callback(StreamingChunk(content="Done"))
            return {"replies": [ChatMessage.from_assistant("Done")]}
        arguments = json.dumps(self.arguments)
        await streaming_callback(
            StreamingChunk(
                content="",
                index=0,
                tool_calls=[ToolCallDelta(index=0, tool_name=self.tool_name, id="actual-call")],
            )
        )
        for fragment in [arguments[:2], arguments[2:]]:
            await streaming_callback(
                StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(index=0, arguments=fragment)])
            )
        await streaming_callback(StreamingChunk(content="", finish_reason="tool_calls"))
        return {
            "replies": [
                ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name=self.tool_name, arguments=self.arguments, id="actual-call")]
                )
            ]
        }


@component
class RequestGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools=None, streaming_callback=None):
        return {"replies": [ChatMessage.from_assistant("Done")]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(self, messages: list[ChatMessage], tools=None, streaming_callback=None):
        label = next(message.text for message in messages if message.role.value == "user")
        return await ToolGenerator("echo", {"label": label}).run_async(messages, tools, streaming_callback)


def _tool(function, *, name="echo", properties=None, **kwargs):
    return Tool(
        name=name,
        description="Return the tool's actual output",
        parameters={"type": "object", "properties": properties or {}},
        function=function,
        **kwargs,
    )


@pytest.mark.parametrize("fails", [False, True])
async def test_ordinary_agent_tools_keep_results_errors_and_correlation(fails):
    invoked = []

    def echo(text):
        invoked.append(text)
        if fails:
            message = "Cannot read notes.txt"
            raise ValueError(message)
        return "full [literal] output\nsecond line"

    agent = Agent(
        chat_generator=ToolGenerator(arguments={"text": "hello"}),
        tools=[_tool(echo, properties={"text": {"type": "string"}})],
        raise_on_tool_invocation_failure=False,
    )
    output = [item async for item in stream_agent_chat(agent, [ChatMessage.from_user("Read")])]
    events = [item for item in output if isinstance(item, PipelineEvent)]
    assert [event.type for event in events] == ["hayhooks.tool_call", "hayhooks.tool_result"]
    start, result = events
    assert start.data["call_id"] == result.data["call_id"]
    assert start.data["call_id"].endswith(":actual-call")
    assert start.data["name"] == "echo"
    assert json.loads(start.data["arguments"]) == {"text": "hello"}
    assert invoked == ["hello"]
    assert result.data["error"] is fails
    if fails:
        assert "Cannot read notes.txt" in result.data["result"]
    else:
        assert result.data["result"] == "full [literal] output\nsecond line"
    chunks = [item for item in output if isinstance(item, StreamingChunk)]
    assert "".join(item.content for item in chunks) == "Done"
    assert not any(item.tool_calls or item.reasoning for item in chunks)
    assert [item.finish_reason for item in chunks if item.finish_reason] == ["stop"]


@pytest.mark.parametrize("argument", ["normalized", None], ids=["rewrite", "drop"])
async def test_before_tool_hook_reports_only_executed_call(argument):
    hooks = pytest.importorskip("haystack.hooks")
    from haystack.components.agents.state import State

    @hooks.hook
    def edit_call(state: State) -> None:
        if argument is None:
            state.data["messages"].pop()
        else:
            state.data["messages"][-1].tool_calls[0].arguments["text"] = argument

    invoked = []

    def echo(text):
        invoked.append(text)
        return text.upper()

    proposed_call = ToolCall(tool_name="echo", arguments={"text": "original"}, id="actual-call")
    agent = Agent(
        chat_generator=CompletionGenerator(
            [
                ChatMessage.from_assistant(tool_calls=[proposed_call], meta={"finish_reason": "tool_calls"}),
                ChatMessage.from_assistant("Done", meta={"finish_reason": "stop"}),
            ]
        ),
        tools=[_tool(echo, properties={"text": {"type": "string"}})],
        hooks={"before_tool": [edit_call]},
    )

    output = [item async for item in stream_agent_chat(agent, [ChatMessage.from_user("Check")])]
    events = [item for item in output if isinstance(item, PipelineEvent)]
    assert [item.finish_reason for item in output if isinstance(item, StreamingChunk) and item.finish_reason] == [
        "stop"
    ]
    if argument is None:
        assert invoked == []
        assert events == []
    else:
        assert invoked == [argument]
        assert [event.type for event in events] == ["hayhooks.tool_call", "hayhooks.tool_result"]
        start, result = events
        assert start.data["call_id"] == result.data["call_id"]
        assert start.data["name"] == "echo"
        assert json.loads(start.data["arguments"]) == {"text": argument}
        assert result.data["result"] == argument.upper()
        assert result.data["error"] is False


@pytest.mark.parametrize("first_reason", ["tool_calls", "length", "content_filter"])
@pytest.mark.parametrize("second_id", ["second-id", "first-id"])
async def test_tool_calls_do_not_reuse_arguments_or_event_ids_between_model_invocations(first_reason, second_id):
    provider_ids = ["first-id", second_id]
    words = ["first", "second"]
    replies = [
        ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="echo", arguments={"word": word}, id=call_id)],
            meta={"finish_reason": reason},
        )
        for word, call_id, reason in zip(words, provider_ids, [first_reason, "tool_calls"], strict=True)
    ]
    replies.append(ChatMessage.from_assistant("Done", meta={"finish_reason": "stop"}))
    agent = Agent(
        chat_generator=CompletionGenerator(replies),
        tools=[_tool(lambda word: word, properties={"word": {"type": "string"}})],
    )

    output = [item async for item in stream_agent_chat(agent, [ChatMessage.from_user("Echo twice")])]

    events = [item for item in output if isinstance(item, PipelineEvent)]
    assert [event.type for event in events] == ["hayhooks.tool_call", "hayhooks.tool_result"] * 2
    starts, results = events[::2], events[1::2]
    assert len({start.data["call_id"] for start in starts}) == 2
    for start, result, word, provider_id in zip(starts, results, words, provider_ids, strict=True):
        assert start.data["call_id"] == result.data["call_id"]
        assert start.data["call_id"].endswith(":" + provider_id)
        assert json.loads(start.data["arguments"]) == {"word": word}
        assert result.data["result"] == word
        assert result.data["error"] is False
    assert [item.finish_reason for item in output if isinstance(item, StreamingChunk) and item.finish_reason] == [
        "stop"
    ]


@component
class PartialToolGenerator:
    def __init__(self, calls, deltas):
        self.calls = calls
        self.deltas = deltas

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools=None, streaming_callback=None):
        return {"replies": [ChatMessage.from_assistant("Done", meta={"finish_reason": "stop"})]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(self, messages: list[ChatMessage], tools=None, streaming_callback=None):
        if any(message.tool_call_results for message in messages):
            await streaming_callback(StreamingChunk(content="Done", finish_reason="stop"))
            return self.run(messages, tools, streaming_callback)
        for delta in self.deltas:
            await streaming_callback(StreamingChunk(content="", index=delta.index, tool_calls=[delta]))
        await streaming_callback(StreamingChunk(content="", finish_reason="tool_calls"))
        return {"replies": [ChatMessage.from_assistant(tool_calls=self.calls, meta={"finish_reason": "tool_calls"})]}


@pytest.mark.parametrize("first_delta", ["with-id", "without-id", "omitted"])
@pytest.mark.parametrize("reverse_results", [False, True])
@pytest.mark.parametrize("different_names", [False, True])
async def test_parallel_tool_calls_with_same_provider_id_have_matching_start_and_result(
    first_delta, reverse_results, different_names
):
    if different_names:
        first = ToolCall(tool_name="ping", arguments={}, id="same")
        second = ToolCall(tool_name="echo", arguments={}, id="same")
        second_arguments = "{}"
    else:
        first = ToolCall(tool_name="echo", arguments={"value": True, "suffix": "!"}, id="same")
        second = ToolCall(tool_name="echo", arguments={"value": 1, "suffix": "!"}, id="same")
        second_arguments = '{ "suffix": "!", "value": 1 }'
    calls = [second, first] if reverse_results else [first, second]
    deltas = []
    for index, call in enumerate(calls):
        if call is first and first_delta == "omitted":
            continue
        deltas.append(
            ToolCallDelta(
                index=index,
                id=None if call is first and first_delta == "without-id" else call.id,
                tool_name=call.tool_name,
                arguments=second_arguments if call is second else json.dumps(call.arguments),
            )
        )
    agent = Agent(
        chat_generator=PartialToolGenerator(calls, deltas),
        tools=[
            _tool(lambda **args: json.dumps(["echo", args], sort_keys=True)),
            _tool(lambda: json.dumps(["ping", {}]), name="ping"),
        ],
    )

    events = [
        item
        async for item in stream_agent_chat(agent, [ChatMessage.from_user("Run both")])
        if isinstance(item, PipelineEvent)
    ]

    assert len(events) == 4
    if first_delta == "with-id":
        assert [event.type for event in events] == ["hayhooks.tool_call", "hayhooks.tool_result"] * 2
    else:
        assert events[0].type == "hayhooks.tool_call"
        assert events[0].data["name"] == second.tool_name
        assert events[0].data["arguments"] == (second_arguments or "{}")
    starts = {event.data["call_id"]: event.data for event in events if event.type == "hayhooks.tool_call"}
    results = [event for event in events if event.type == "hayhooks.tool_result"]
    assert len(starts) == len(results) == 2
    for call, result in zip(calls, results, strict=True):
        start = starts[result.data["call_id"]]
        assert start["call_id"].endswith(":same")
        expected = json.dumps([call.tool_name, call.arguments], sort_keys=True)
        assert result.data["result"] == expected
        assert json.dumps([start["name"], json.loads(start["arguments"] or "{}")], sort_keys=True) == expected
        assert result.data["error"] is False


async def test_omitted_streamed_arguments_use_completed_origin():
    invoked = []

    def echo(word):
        invoked.append(word)
        return word

    call = ToolCall(tool_name="echo", arguments={"word": "first"}, id="ordinary-id")
    delta = ToolCallDelta(index=0, id=call.id, tool_name=call.tool_name)
    agent = Agent(chat_generator=PartialToolGenerator([call], [delta]), tools=[_tool(echo)])

    events = [item async for item in stream_agent_chat(agent, []) if isinstance(item, PipelineEvent)]

    assert invoked == ["first"]
    assert [event.type for event in events] == ["hayhooks.tool_call", "hayhooks.tool_result"]
    start, result = events
    assert start.data["name"] == "echo"
    assert json.loads(start.data["arguments"]) == {"word": "first"}
    assert start.data["call_id"] == result.data["call_id"]
    assert result.data["result"] == "first"
    assert result.data["error"] is False


async def test_malformed_streamed_arguments_use_completed_origin():
    call = ToolCall(tool_name="echo", arguments={"word": "first"}, id="same")
    delta = ToolCallDelta(index=0, id=call.id, tool_name=call.tool_name, arguments='{"word":')
    agent = Agent(chat_generator=PartialToolGenerator([call], [delta]), tools=[_tool(lambda word: word)])

    events = [item async for item in stream_agent_chat(agent, []) if isinstance(item, PipelineEvent)]

    start, result = events
    assert json.loads(start.data["arguments"]) == call.arguments
    assert start.data["call_id"] == result.data["call_id"]
    assert result.data["result"] == "first"


@pytest.mark.parametrize("provider_id", ["reused", None])
@pytest.mark.parametrize("unannounced_deltas", [False, True])
async def test_result_only_tool_calls_have_unique_correlated_ids(monkeypatch, provider_id, unannounced_deltas):
    async def source():
        for word in ["first", "second"]:
            if unannounced_deltas:
                yield StreamingChunk(
                    content="",
                    index=0,
                    tool_calls=[ToolCallDelta(index=0, id=provider_id, tool_name="echo", arguments='{"word":')],
                    finish_reason="length",
                )
            yield StreamingChunk(
                content="",
                index=0,
                tool_call_result=ToolCallResult(
                    result={"word": word},
                    origin=ToolCall(tool_name="echo", arguments={"word": word}, id=provider_id),
                    error=False,
                ),
            )

    monkeypatch.setattr("hayhooks.agent_chat.async_streaming_generator", lambda *_args, **_kwargs: source())
    output = [item async for item in stream_agent_chat(Agent(chat_generator=ToolGenerator()), [])]

    events = [item for item in output if isinstance(item, PipelineEvent)]
    assert [event.type for event in events] == ["hayhooks.tool_call", "hayhooks.tool_result"] * 2
    starts, results = events[::2], events[1::2]
    assert len({start.data["call_id"] for start in starts}) == 2
    for start, result, word in zip(starts, results, ["first", "second"], strict=True):
        assert start.data["call_id"] == result.data["call_id"]
        if provider_id:
            assert start.data["call_id"].endswith(":" + provider_id)
        assert start.data["name"] == "echo"
        assert json.loads(start.data["arguments"]) == json.loads(result.data["result"]) == {"word": word}


@pytest.mark.parametrize("second_id", ["real-18", "real-17"])
@pytest.mark.parametrize("reverse_results", [False, True])
async def test_fragmented_arguments_multiple_indices_and_missing_start(monkeypatch, second_id, reverse_results):
    closed = asyncio.Event()

    async def source():
        try:
            for delta in [
                ToolCallDelta(index=0, id="real-17", tool_name="read_file", arguments='{"path":'),
                ToolCallDelta(index=1, tool_name="read_file", arguments='{"path":"b.txt"}'),
                ToolCallDelta(index=0, arguments='"a.txt"}'),
                ToolCallDelta(index=1, id=second_id),
            ]:
                yield StreamingChunk(content="", index=0, tool_calls=[delta])
            yield StreamingChunk(content="", finish_reason="tool_calls")
            results = [("real-17", "a.txt"), (second_id, "b.txt"), ("real-19", "c.txt")]
            for call_id, path in reversed(results) if reverse_results else results:
                yield StreamingChunk(
                    content="",
                    index=0,
                    tool_call_result=ToolCallResult(
                        result=f"[literal] {path}",
                        origin=ToolCall(tool_name="read_file", arguments={"path": path}, id=call_id),
                        error=path == "b.txt",
                    ),
                )
            yield StreamingChunk(content="Final")
        finally:
            closed.set()

    monkeypatch.setattr("hayhooks.agent_chat.async_streaming_generator", lambda *_args, **_kwargs: source())
    output = [item async for item in stream_agent_chat(Agent(chat_generator=ToolGenerator()), [])]
    events = [item for item in output if isinstance(item, PipelineEvent)]
    if second_id == "real-17":
        assert [event.type for event in events] == ["hayhooks.tool_call", "hayhooks.tool_result"] * 3
    else:
        assert [event.type for event in events[:2]] == ["hayhooks.tool_call"] * 2
    starts = {
        item.data["call_id"]: item.data
        for item in output
        if isinstance(item, PipelineEvent) and item.type == "hayhooks.tool_call"
    }
    results = [item.data for item in output if isinstance(item, PipelineEvent) and item.type == "hayhooks.tool_result"]
    assert len(starts) == len(results) == 3
    for result in results:
        start = starts[result["call_id"]]
        path = json.loads(start["arguments"])["path"]
        assert result["result"] == f"[literal] {path}"
        assert result["error"] is (path == "b.txt")
    assert output[-1].content == "Final" and closed.is_set()


@pytest.mark.parametrize("reason", ["length", "content_filter"])
async def test_usage_trailer_after_tool_result_preserves_model_failure(monkeypatch, reason):
    async def source():
        yield StreamingChunk(content="Partial", finish_reason=reason)
        yield StreamingChunk(
            content="",
            index=0,
            tool_call_result=ToolCallResult(
                result="Found",
                origin=ToolCall(tool_name="lookup", arguments={}, id="lookup-1"),
                error=False,
            ),
        )
        yield StreamingChunk(content="", meta={"usage": {"total_tokens": 12}})

    monkeypatch.setattr("hayhooks.agent_chat.async_streaming_generator", lambda *_args, **_kwargs: source())
    output = [item async for item in stream_agent_chat(Agent(chat_generator=ToolGenerator()), [])]

    terminal_reasons = [
        item.finish_reason for item in output if isinstance(item, StreamingChunk) and item.finish_reason
    ]
    assert terminal_reasons == [reason]


@pytest.mark.parametrize("in_toolset", [False, True])
async def test_shared_agent_preserves_tools_toolsets_and_request_state(in_toolset):
    barrier = threading.Barrier(2)
    received = []

    def echo(label, request_label):
        barrier.wait(timeout=3)
        received.append((label, request_label))
        return f"{label}/{request_label}"

    class OpaqueToolset(Toolset):
        def to_dict(self):
            message = "The supplied Toolset must not be serialized"
            raise AssertionError(message)

    tool = _tool(
        echo,
        properties={"label": {"type": "string"}},
        inputs_from_state={"request_label": "request_label"},
    )
    tools = OpaqueToolset([tool]) if in_toolset else [tool]
    agent = Agent(chat_generator=RequestGenerator(), tools=tools, state_schema={"request_label": {"type": str}})
    configured = agent.tools

    async def collect(label):
        return [
            item
            async for item in stream_agent_chat(agent, [ChatMessage.from_user(label)], request_label=f"state-{label}")
        ]

    first, second = await asyncio.wait_for(asyncio.gather(collect("one"), collect("two")), timeout=8)
    assert agent.tools is configured
    assert sorted(received) == [("one", "state-one"), ("two", "state-two")]
    for label, output in [("one", first), ("two", second)]:
        events = [item for item in output if isinstance(item, PipelineEvent)]
        assert len(events) == 2
        assert json.loads(events[0].data["arguments"]) == {"label": label}
        assert events[0].data["call_id"] == events[1].data["call_id"]
        assert events[0].data["call_id"].endswith(":actual-call")
        assert events[1].data["result"] == f"{label}/state-{label}"


async def test_helper_forwards_runtime_tools_without_changing_agent():
    original = _tool(lambda: "configured")
    override = _tool(lambda: "run-specific tool", name="selected")
    agent = Agent(chat_generator=ToolGenerator("selected"), tools=[original])
    configured = agent.tools
    output = [item async for item in stream_agent_chat(agent, [ChatMessage.from_user("Select")], tools=[override])]
    assert agent.tools is configured
    assert any(isinstance(item, PipelineEvent) and item.data.get("result") == "run-specific tool" for item in output)


def test_helper_requires_agent_and_owns_callback():
    with pytest.raises(TypeError, match="requires a Haystack Agent"):
        stream_agent_chat(Pipeline(), [])
    with pytest.raises(ValueError, match="owns streaming_callback"):
        stream_agent_chat(Agent(chat_generator=ToolGenerator()), [], streaming_callback=lambda _item: None)


async def test_stream_can_close_from_another_task():
    closed = asyncio.Event()
    release = asyncio.Event()

    @component
    class WaitingGenerator:
        @component.output_types(replies=list[ChatMessage])
        def run(self, messages: list[ChatMessage], streaming_callback=None):
            return {"replies": [ChatMessage.from_assistant("one")]}

        @component.output_types(replies=list[ChatMessage])
        async def run_async(self, messages: list[ChatMessage], streaming_callback=None):
            try:
                await streaming_callback(StreamingChunk(content="one"))
                await release.wait()
                return {"replies": [ChatMessage.from_assistant("one")]}
            finally:
                closed.set()

    stream = await asyncio.to_thread(stream_agent_chat, Agent(chat_generator=WaitingGenerator()), [])
    try:
        assert (await asyncio.wait_for(anext(stream), timeout=2)).content == "one"
        await asyncio.wait_for(asyncio.create_task(stream.aclose()), timeout=2)
        assert closed.is_set()
    finally:
        release.set()
        await stream.aclose()


@pytest.mark.integration
async def test_http_disconnect_closes_stream_and_bounded_sync_tool_settles():  # noqa: C901, PLR0915
    started = threading.Event()
    finished = threading.Event()
    release = threading.Event()
    closed = asyncio.Event()

    def echo():
        started.set()
        try:
            release.wait(timeout=1)
            return "settled"
        finally:
            finished.set()

    class Wrapper(BasePipelineWrapper):
        def setup(self):
            self.pipeline = Agent(chat_generator=ToolGenerator(), tools=[_tool(echo)])

        async def run_chat_completion_async(self, model, messages, body):
            stream = stream_agent_chat(self.pipeline, [ChatMessage.from_openai_dict_format(item) for item in messages])

            async def output():
                try:
                    async for item in stream:
                        yield item
                finally:
                    await stream.aclose()
                    closed.set()

            return output()

    module = ModuleType("bounded_agent_chat")
    module.PipelineWrapper = Wrapper
    wrapper = create_pipeline_wrapper_instance(module)
    registry = _PipelineRegistry()
    registry.add("agent", wrapper)
    app = FastAPI()
    app.include_router(create_openai_router(registry))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    request = {"model": "agent", "messages": [{"role": "user", "content": "Run"}], "stream": True}

    async def exercise():
        while not server.started:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient(timeout=3) as client:
            url = f"http://127.0.0.1:{sock.getsockname()[1]}/v1/chat/completions"
            async with client.stream("POST", url, json=request) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if "hayhooks.tool_call" in line:
                        break
                else:
                    pytest.fail("The Agent did not emit a tool call")
                assert await asyncio.to_thread(started.wait, 2)
                assert not finished.is_set()
            await closed.wait()
            assert await asyncio.to_thread(finished.wait, 2)
            assert not any(
                "Agent.run_async" in getattr(task.get_coro(), "__qualname__", "")
                for task in asyncio.all_tasks()
                if not task.done()
            )
            release.set()
            follow_up = await client.post(url, json=request)
            assert follow_up.status_code == 200
            assert '"content":"Done"' in follow_up.text
            assert '"finish_reason":"stop"' in follow_up.text

    try:
        await asyncio.wait_for(exercise(), timeout=8)
    finally:
        release.set()
        server.should_exit = True
        try:
            await asyncio.wait_for(serving, 3)
        finally:
            if not serving.done():
                serving.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await serving
            sock.close()


def test_ordinary_agent_chat_does_not_import_optional_sdk():
    script = textwrap.dedent("""
        import asyncio
        import sys
        class NoSDK:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "deepseek_harness" or fullname.startswith("deepseek_harness."):
                    raise AssertionError("Ordinary Agent chat imported the SDK")
        sys.meta_path.insert(0, NoSDK())
        from haystack import component
        from haystack.components.agents import Agent
        from haystack.dataclasses import ChatMessage, StreamingChunk
        from hayhooks import stream_agent_chat
        @component
        class Generator:
            @component.output_types(replies=list[ChatMessage])
            def run(self, messages: list[ChatMessage], streaming_callback=None):
                return {"replies": [ChatMessage.from_assistant("Ready")]}
            @component.output_types(replies=list[ChatMessage])
            async def run_async(self, messages: list[ChatMessage], streaming_callback=None):
                await streaming_callback(StreamingChunk(content="Ready"))
                return {"replies": [ChatMessage.from_assistant("Ready")]}
        async def main():
            output = [item async for item in stream_agent_chat(Agent(chat_generator=Generator()), [])]
            assert "".join(item.content for item in output) == "Ready"
            assert "deepseek_harness" not in sys.modules
        asyncio.run(main())
    """)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], check=False, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


async def test_custom_pipeline_chat_remains_supported():
    class PipelineWrapper(BasePipelineWrapper):
        def setup(self):
            self.pipeline = Pipeline()

        async def run_chat_completion_async(self, model, messages, body):
            async def output():
                yield StreamingChunk(content="Custom pipeline chat")

            return output()

    module = ModuleType("custom_pipeline_chat")
    module.PipelineWrapper = PipelineWrapper
    wrapper = create_pipeline_wrapper_instance(module)
    registry = _PipelineRegistry()
    registry.add("custom", wrapper)
    app = FastAPI()
    app.include_router(create_openai_router(registry))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "custom", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
    assert response.status_code == 200
    assert '"content":"Custom pipeline chat"' in response.text
    assert '"finish_reason":"stop"' in response.text


def test_example_loads_as_ordinary_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-for-model-requests")
    monkeypatch.setenv("EXAMPLE_CHAT_WORKSPACE", str(tmp_path))
    path = Path(pipeline_wrapper.__file__)
    wrapper = prepare_pipeline_files(
        "terminal_chat_agent_reload",
        {name: (path.parent / name).read_text() for name in ["pipeline_wrapper.py", "file_tools.py"]},
        save_files=False,
    ).wrapper
    assert isinstance(wrapper.pipeline, Agent)
    assert wrapper._is_run_chat_completion_async_implemented
    assert not Path(sys.modules[type(wrapper).__module__].__file__).parent.exists()
