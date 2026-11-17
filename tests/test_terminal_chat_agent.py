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
    assert start.data["call_id"] == result.data["call_id"] == "agent:actual-call"
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


async def test_fragmented_arguments_multiple_indices_and_missing_start(monkeypatch):
    closed = asyncio.Event()

    async def source():
        try:
            for delta in [
                ToolCallDelta(index=0, id="real-17", tool_name="read_file", arguments='{"path":'),
                ToolCallDelta(index=1, tool_name="read_file", arguments='{"path":"b.txt"}'),
                ToolCallDelta(index=0, arguments='"a.txt"}'),
                ToolCallDelta(index=1, id="real-18"),
            ]:
                yield StreamingChunk(content="", index=0, tool_calls=[delta])
            yield StreamingChunk(content="", finish_reason="tool_calls")
            for call_id, path in [("real-17", "a.txt"), ("real-18", "b.txt"), ("real-19", "c.txt")]:
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
        assert events[0].data["call_id"] == events[1].data["call_id"] == "agent:actual-call"
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
