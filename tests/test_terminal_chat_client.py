import asyncio
import json

import httpx
import pytest

from hayhooks.chat.client import ChatClient, ChatError
from hayhooks.chat.events import ResponseComplete, ResponseDelta, ToolFinished


def frame(payload):
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def choice(text=None, finish=None):
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}


class Stream(httpx.AsyncByteStream):
    def __init__(self, frames, *, error=None):
        self.frames = frames
        self.error = error
        self.paused = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for item in self.frames:
            if isinstance(item, asyncio.Event):
                self.paused.set()
                await item.wait()
            else:
                yield item
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


def remote(monkeypatch, *, frames=(), error=None, status=200, payload=None, headers=None):  # noqa: PLR0913
    requests = []
    stream = Stream(frames, error=error)

    def handle(request):
        requests.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json=payload, headers=headers)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handle), **kw))
    return ChatClient("http://server", "workspace-agent"), requests, stream


@pytest.mark.parametrize(
    ("status", "kind", "action"),
    [
        (401, "access", "Check access"),
        (403, "access", "Check access"),
        (404, "deployment", "deployment name"),
        (429, "server", "server logs"),
        (500, "server", "server logs"),
        (501, "server", "server logs"),
    ],
)
async def test_http_failure_classification_and_no_retry(monkeypatch, status, kind, action):
    client, requests, _ = remote(monkeypatch, status=status, payload={"detail": "Known cause"})
    with pytest.raises(ChatError) as caught:
        await client.run("Hi", lambda _: None, emit=lambda _: None)
    assert caught.value.kind == kind
    assert all(part in str(caught.value) for part in ("workspace-agent", f"HTTP {status}", "Known cause", action))
    assert client.history == [] and len(requests) == 1


@pytest.mark.parametrize("terminal", [frame(choice("answer", "stop")), frame(choice("answer")) + b"data: [DONE]\n\n"])
async def test_confirmed_completion_accepts_both_endpoint_markers(monkeypatch, terminal):
    client, requests, stream = remote(monkeypatch, frames=[terminal])
    events = []
    assert await client.run("Hi", lambda _: None, emit=events.append) == "answer"
    assert events == [ResponseDelta("answer", "Agent", "answer"), ResponseComplete("answer", "Agent", "answer", True)]
    assert stream.closed and len(requests) == 1
    assert client.history == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "answer"}]


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("chunk_size", [1, 4096])
async def test_unescaped_unicode_survives_chunked_chat_stream(monkeypatch, line_ending, chunk_size):
    content = "Hello\u2028World\u2029Next\x85Line\ufeff\x0b\x0c\x1c\x1d\x1e"
    payloads = [
        choice(content),
        {
            "event": {
                "type": "hayhooks.message",
                "data": {"message_id": "side\u2028id", "actor": "Agent", "text": content},
            }
        },
        {"event": {"type": "hayhooks.tool_result", "data": {"call_id": "tool\u2029id", "result": content}}},
        choice(finish="stop"),
    ]
    wire = "".join("data: " + json.dumps(payload, ensure_ascii=False) + line_ending * 2 for payload in payloads)
    wire = (wire + "data: [DONE]" + line_ending * 2).encode("utf-8-sig")
    client, requests, stream = remote(
        monkeypatch, frames=[wire[index : index + chunk_size] for index in range(0, len(wire), chunk_size)]
    )
    events = []
    assert await client.run("Hi", lambda _: None, emit=events.append) == content
    assert events == [
        ResponseDelta("answer", "Agent", content),
        ResponseComplete("side\u2028id", "Agent", content),
        ToolFinished("tool\u2029id", content),
        ResponseComplete("answer", "Agent", content, True),
    ]
    assert stream.closed and len(requests) == 1
    assert client.history == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": content}]


@pytest.mark.parametrize(
    ("chunks", "after_pause"),
    [
        ([b'data: {"choices":[{"delta":{"content":"answer"}}]}\r\r'], b"data: [DONE]"),
        ([b'data: {"choices":[\r', b"", b'\ndata: {"delta":{"content":"answer"}}]}\r', b"\n\r"], b"\ndata: [DONE]"),
    ],
    ids=["cr", "split-crlf"],
)
async def test_complete_event_arrives_before_next_byte(monkeypatch, chunks, after_pause):
    release = asyncio.Event()
    client, _, stream = remote(monkeypatch, frames=[*chunks, release, after_pause])
    events = []
    task = asyncio.create_task(client.run("Hi", lambda _: None, emit=events.append))
    try:
        await asyncio.wait_for(stream.paused.wait(), 5)
        assert events == [ResponseDelta("answer", "Agent", "answer")]
    finally:
        release.set()
        answer = await task
    assert answer == "answer" and stream.closed


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("terminal", ["[DONE]", json.dumps(choice(finish="stop"))])
async def test_multiline_data_and_unterminated_completion_marker(monkeypatch, line_ending, terminal):
    wire = line_ending.join(
        [
            ": keepalive",
            "event: message",
            "id: ignored",
            'data: {"choices": [',
            'data:  {"delta": {"content": "answer"}}]}',
            "",
            "data: " + terminal,
        ]
    ).encode()
    client, requests, stream = remote(monkeypatch, frames=[wire[index : index + 1] for index in range(len(wire))])
    events = []
    assert await client.run("Hi", lambda _: None, emit=events.append) == "answer"
    assert events == [ResponseDelta("answer", "Agent", "answer"), ResponseComplete("answer", "Agent", "answer", True)]
    assert stream.closed and len(requests) == 1
    assert client.history == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "answer"}]


@pytest.mark.parametrize("finish", [None, "tool_calls", "tool_call_results"])
async def test_partial_eof_is_incomplete_and_not_history(monkeypatch, finish):
    client, requests, _ = remote(monkeypatch, frames=[frame(choice("Partial", finish))])
    events = []
    with pytest.raises(ChatError, match="Completion is unknown") as caught:
        await client.run("Hi", lambda _: None, emit=events.append)
    assert caught.value.kind == "incomplete"
    assert events == [ResponseDelta("answer", "Agent", "Partial")]
    assert client.history == [] and len(requests) == 1


@pytest.mark.parametrize("finish", ["length", "content_filter"])
async def test_finish_failure_keeps_same_frame_text_even_before_done(monkeypatch, finish):
    client, _, _ = remote(monkeypatch, frames=[frame(choice("Partial", finish)), b"data: [DONE]\n\n"])
    events = []
    with pytest.raises(ChatError) as caught:
        await client.run("Hi", lambda _: None, emit=events.append)
    assert caught.value.kind == "provider"
    assert events == [ResponseDelta("answer", "Agent", "Partial")]
    assert client.history == []


@pytest.mark.parametrize(
    "late", [frame({"error": {"message": "Quota exhausted", "code": "insufficient_quota"}}), b"data: {\n\n"]
)
async def test_errors_after_stop_are_observed(monkeypatch, late):
    client, requests, _ = remote(monkeypatch, frames=[frame(choice("Partial", "stop")), late])
    events = []
    with pytest.raises(ChatError):
        await client.run("Hi", lambda _: None, emit=events.append)
    assert events == [ResponseDelta("answer", "Agent", "Partial")]
    assert client.history == [] and len(requests) == 1


async def test_valid_content_usage_and_unknown_events_after_marker_are_allowed(monkeypatch):
    client, _, _ = remote(
        monkeypatch,
        frames=[
            b"data: [DONE]\n\n",
            frame(choice("Still valid", "stop")),
            frame({"choices": [], "usage": {"total_tokens": 7}}),
            frame({"event": {"type": "custom.progress", "data": [1, 2]}}),
        ],
    )
    assert await client.run("Hi", lambda _: None, emit=lambda _: None) == "Still valid"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"choices": {}},
        {"choices": [None]},
        {"choices": [{"delta": []}]},
        {"choices": [{"delta": {"content": 42}}]},
        {"choices": [{"finish_reason": []}]},
        {"event": []},
        {"event": {"type": "status", "data": []}},
        {"event": {"type": "status", "data": {"description": []}}},
        {"event": {"type": "hayhooks.tool_result", "data": {"result": "missing ID"}}},
        {"event": {"type": "hayhooks.message", "data": []}},
    ],
)
async def test_malformed_recognized_envelopes_are_protocol_failures(monkeypatch, payload):
    client, requests, _ = remote(monkeypatch, frames=[frame(payload)])
    with pytest.raises(ChatError, match="invalid chat data") as caught:
        await client.run("Hi", lambda _: None, emit=lambda _: None)
    assert caught.value.kind == "protocol"
    assert client.history == [] and len(requests) == 1


async def test_tool_error_can_recover_and_keeps_result_literal(monkeypatch):
    result = "[bold]error[/bold] /private/workspace/file api_key=literal"
    client, _, _ = remote(
        monkeypatch,
        frames=[
            frame(
                {"event": {"type": "hayhooks.tool_result", "data": {"call_id": "one", "result": result, "error": True}}}
            ),
            frame(
                {
                    "event": {
                        "type": "hayhooks.message",
                        "data": {"message_id": "intermediate", "actor": "Agent", "text": "Working"},
                    }
                }
            ),
            frame(choice("Recovered", "stop")),
        ],
    )
    events = []
    assert await client.run("Hi", lambda _: None, emit=events.append) == "Recovered"
    assert events[0] == ToolFinished("one", result, True)


async def test_side_message_alone_does_not_complete_turn(monkeypatch):
    client, _, _ = remote(
        monkeypatch,
        frames=[
            frame(
                {
                    "event": {
                        "type": "hayhooks.message",
                        "data": {
                            "message_id": "one",
                            "actor": "Agent",
                            "text": "Intermediate",
                            "final": True,
                        },
                    }
                }
            )
        ],
    )
    events = []
    with pytest.raises(ChatError):
        await client.run("Hi", lambda _: None, emit=events.append)
    assert events == [ResponseComplete("one", "Agent", "Intermediate")]
    assert client.history == []


@pytest.mark.parametrize(
    "error_class",
    [
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        httpx.ReadTimeout,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        httpx.WriteError,
    ],
)
async def test_connection_setup_versus_unknown_pre_header_failure(monkeypatch, error_class):
    requests = []

    def handler(request):
        requests.append(request)
        message = "secret internal exception blob /private/path sk-secret"
        raise error_class(message)

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    client = ChatClient("http://server", "workspace-agent")
    with pytest.raises(ChatError) as caught:
        await client.run("Hi", lambda _: None, emit=lambda _: None)
    unavailable = error_class in (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
    assert caught.value.kind == ("unavailable" if unavailable else "incomplete")
    assert "secret" not in str(caught.value) and "internal" not in str(caught.value)
    assert len(requests) == 1 and client.history == []


async def test_read_failure_after_stop_prevents_success(monkeypatch):
    client, requests, stream = remote(
        monkeypatch, frames=[frame(choice("Partial", "stop"))], error=httpx.ReadError("hidden")
    )
    events = []
    with pytest.raises(ChatError, match="Completion is unknown"):
        await client.run("Hi", lambda _: None, emit=events.append)
    assert stream.closed and client.history == [] and len(requests) == 1
    assert events == [ResponseDelta("answer", "Agent", "Partial")]


async def test_cancellation_after_terminal_marker_still_discards_turn(monkeypatch):
    client, requests, stream = remote(monkeypatch, frames=[frame(choice("Partial", "stop")), asyncio.Event()])
    events = []
    task = asyncio.create_task(client.run("Hi", lambda _: None, emit=events.append))
    await asyncio.wait_for(stream.paused.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed and client.history == [] and len(requests) == 1
    assert events == [ResponseDelta("answer", "Agent", "Partial")]


@pytest.mark.parametrize("failure", [TypeError, AttributeError])
async def test_emitter_failures_are_not_wire_errors(monkeypatch, failure):
    client, _, _ = remote(monkeypatch, frames=[frame(choice("Hi", "stop"))])

    def emit(event):
        message = "UI failed"
        raise failure(message)

    with pytest.raises(failure, match="UI failed"):
        await client.run("Hi", lambda _: None, emit=emit)
    assert client.history == []


@pytest.mark.parametrize("status", [429, 500])
async def test_safe_bounded_diagnostic_preserves_cause_and_supplied_id(monkeypatch, status):
    detail = (
        "Quota exhausted\n/private/server/app.py\x1b[31m Bearer token123 api_key=secret123 sk-verysecret " + "x" * 600
    )
    client, _, _ = remote(
        monkeypatch,
        status=status,
        payload={
            "error": {
                "message": detail,
                "code": "insufficient_quota",
                "stack": "NEVER SHOW",
                "request_id": "supplied-123",
            }
        },
    )
    with pytest.raises(ChatError) as caught:
        await client.run("Hi", lambda _: None, emit=lambda _: None)
    message = str(caught.value)
    assert caught.value.kind == "provider"
    assert "Quota exhausted" in message and "supplied-123" in message
    assert all(
        secret not in message
        for secret in ("token123", "secret123", "verysecret", "/private", "NEVER SHOW", "\x1b", "\n")
    )
    assert len(message) < 650


async def test_no_supplied_cause_stays_unknown(monkeypatch):
    client, _, _ = remote(monkeypatch, status=500, payload={"unexpected": {"stack": "hidden"}})
    with pytest.raises(ChatError, match="supplied no cause") as caught:
        await client.run("Hi", lambda _: None, emit=lambda _: None)
    assert caught.value.kind == "server" and "provider" not in str(caught.value)
