import errno
import json
import os
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.tools.errors import ToolInvocationError

from examples.pipeline_wrappers.terminal_chat_agent import file_tools, pipeline_wrapper
from examples.pipeline_wrappers.terminal_chat_agent.file_tools import build_file_tools
from hayhooks import PipelineEvent, stream_agent_chat
from hayhooks.server.pipelines.registry import _PipelineRegistry
from hayhooks.server.routers.openai import create_openai_router
from hayhooks.server.utils.deploy_utils import prepare_pipeline_files
from tests.test_terminal_chat_agent import ToolGenerator


@pytest.fixture
def tools(tmp_path):
    return {tool.name: tool for tool in build_file_tools(tmp_path)}


def test_file_tools_write_list_and_exact_read_back(tools, tmp_path):
    (tmp_path / "notes").mkdir()
    text = "# A note\nCaffè ☕\r\n[literal]\n"
    assert "UTF-8 bytes" in tools["write_file"].invoke(path="notes/hello.txt", content=text)
    assert (tmp_path / "notes/hello.txt").read_bytes() == text.encode("utf-8")
    assert tools["read_file"].invoke(path="notes/hello.txt") == text
    assert tools["list_files"].invoke() == "notes/"
    assert tools["list_files"].invoke(path="notes") == "hello.txt"
    tools["write_file"].invoke(path="notes/hello.txt", content="replacement")
    assert (tmp_path / "notes/hello.txt").read_bytes() == b"replacement"
    assert list((tmp_path / "notes").iterdir()) == [tmp_path / "notes/hello.txt"]


@pytest.mark.parametrize("size", [0, 65_536, 65_537])
@pytest.mark.parametrize("operation", ["read_file", "write_file"])
def test_encoded_byte_limits(tools, tmp_path, size, operation):
    text = "é" * (size // 2) + "x" * (size % 2)
    target = tmp_path / "note.txt"
    target.write_bytes(text.encode("utf-8") if operation == "read_file" else b"original")
    before = target.read_bytes()
    arguments = {"path": "note.txt"}
    if operation == "write_file":
        arguments["content"] = text
    if size > 65_536:
        with pytest.raises(ToolInvocationError, match="65,536-byte limit"):
            tools[operation].invoke(**arguments)
        assert target.read_bytes() == before
    else:
        result = tools[operation].invoke(**arguments)
        if operation == "read_file":
            assert result == text
        else:
            assert target.read_bytes() == text.encode("utf-8")


@pytest.mark.parametrize("count", [0, 200, 201])
def test_listing_bound_and_truthful_truncation(tools, tmp_path, count):
    for index in range(count):
        (tmp_path / f"note-{index:03}.txt").touch()
    result = tools["list_files"].invoke()
    lines = result.splitlines()
    if not count:
        assert result == "[Empty directory]"
    else:
        entries = [line for line in lines if not line.startswith("[Truncated:")]
        assert len(entries) == min(count, 200)
        assert set(entries) <= {f"note-{index:03}.txt" for index in range(count)}
    assert ("Truncated" in result) is (count > 200)


@pytest.mark.parametrize("operation", ["list_files", "read_file", "write_file"])
@pytest.mark.parametrize("path", ["../outside.txt", "sub/../../outside.txt", "/outside.txt"])
def test_paths_cannot_leave_workspace(tools, tmp_path, operation, path):
    target = tmp_path / "keep.txt"
    target.write_bytes(b"unchanged")
    arguments = {"path": path, **({"content": "new"} if operation == "write_file" else {})}
    with pytest.raises(ToolInvocationError, match="relative path within the configured workspace"):
        tools[operation].invoke(**arguments)
    assert target.read_bytes() == b"unchanged"
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("operation", ["list_files", "read_file", "write_file"])
@pytest.mark.parametrize("path", ["alias/note.txt", "link.txt", "dangling.txt"])
def test_symlinks_are_rejected_without_following(tools, tmp_path, operation, path):
    directory = tmp_path / "real"
    directory.mkdir()
    note = directory / "note.txt"
    note.write_bytes(b"original")
    (tmp_path / "alias").symlink_to(directory, target_is_directory=True)
    (tmp_path / "link.txt").symlink_to(note)
    (tmp_path / "dangling.txt").symlink_to(tmp_path / "missing.txt")
    arguments = {"path": path, **({"content": "replacement"} if operation == "write_file" else {})}
    with pytest.raises(ToolInvocationError, match="symlinks are not allowed"):
        tools[operation].invoke(**arguments)
    assert note.read_bytes() == b"original"
    listing = tools["list_files"].invoke()
    assert "alias [symlink]" in listing and "link.txt [symlink]" in listing


def test_trusted_workspace_alias_is_canonicalized(tmp_path):
    directory = tmp_path / "workspace"
    directory.mkdir()
    alias = tmp_path / "configured-alias"
    alias.symlink_to(directory, target_is_directory=True)
    tools = {tool.name: tool for tool in build_file_tools(alias)}
    tools["write_file"].invoke(path="note.txt", content="through alias")
    assert (directory / "note.txt").read_bytes() == b"through alias"
    assert tools["read_file"].invoke(path="note.txt") == "through alias"


@pytest.mark.parametrize("operation", ["list_files", "read_file", "write_file"])
def test_escaping_symlink_keeps_outside_file_unchanged(tools, tmp_path, operation):
    outside = tmp_path.with_name(tmp_path.name + "-outside")
    outside.mkdir()
    target = outside / "note.txt"
    target.write_bytes(b"outside original")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    arguments = {"path": "escape/note.txt", **({"content": "changed"} if operation == "write_file" else {})}
    with pytest.raises(ToolInvocationError, match="symlinks are not allowed"):
        tools[operation].invoke(**arguments)
    assert target.read_bytes() == b"outside original"


@pytest.mark.parametrize("operation", ["list_files", "read_file", "write_file"])
def test_invalid_paths_are_actionable_tool_errors(tools, operation):
    arguments = {"path": "bad\x00name", **({"content": "text"} if operation == "write_file" else {})}
    with pytest.raises(ToolInvocationError, match="relative path without null characters") as error:
        tools[operation].invoke(**arguments)
    assert operation in str(error.value)


@pytest.mark.parametrize("operation", ["read_file", "write_file"])
@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_nonregular_targets_are_rejected(tools, tmp_path, operation, kind):
    target = tmp_path / "target"
    if kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    arguments = {"path": "target", **({"content": "text"} if operation == "write_file" else {})}
    with pytest.raises(ToolInvocationError, match="only regular UTF-8 text files"):
        tools[operation].invoke(**arguments)
    assert target.is_dir() if kind == "directory" else target.is_fifo()


@pytest.mark.parametrize(
    ("operation", "arguments", "expected"),
    [
        ("read_file", {"path": "missing.txt"}, "Use list_files"),
        ("write_file", {"path": "missing/note.txt", "content": "text"}, "Use list_files"),
        ("read_file", {"path": "binary.txt"}, "only UTF-8 text"),
        ("write_file", {"path": "keep.txt", "content": "\ud800"}, "only UTF-8 text"),
    ],
)
def test_missing_and_utf8_errors_preserve_files(tools, tmp_path, operation, arguments, expected):
    (tmp_path / "binary.txt").write_bytes(b"\xff")
    (tmp_path / "keep.txt").write_bytes(b"original")
    with pytest.raises(ToolInvocationError, match=expected) as error:
        tools[operation].invoke(**arguments)
    assert operation in str(error.value) and arguments["path"] in str(error.value)
    assert str(tmp_path) not in str(error.value)
    assert error.value.__cause__.__cause__ is not None
    assert (tmp_path / "keep.txt").read_bytes() == b"original"
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("operation", ["list_files", "read_file", "write_file"])
def test_permission_errors_keep_relative_path_and_cause(tools, tmp_path, monkeypatch, operation):
    (tmp_path / "note.txt").write_text("original")
    failure = PermissionError(errno.EACCES, "denied", str(tmp_path / "secret-internal-path"))

    def denied(*_args, **_kwargs):
        raise failure

    if operation == "list_files":
        monkeypatch.setattr(file_tools.os, "scandir", denied)
    elif operation == "read_file":
        monkeypatch.setattr(Path, "open", denied)
    else:
        monkeypatch.setattr(file_tools.tempfile, "NamedTemporaryFile", denied)
    path = "." if operation == "list_files" else "note.txt"
    arguments = {"path": path, **({"content": "replacement"} if operation == "write_file" else {})}
    with pytest.raises(ToolInvocationError, match="Check server-side workspace permissions") as error:
        tools[operation].invoke(**arguments)
    assert error.value.__cause__.__cause__ is failure
    assert operation in str(error.value) and repr(path) in str(error.value)
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("failure_point", ["write", "replace"])
def test_failed_writes_keep_original_and_clean_staged_file(tools, tmp_path, monkeypatch, failure_point):
    target = tmp_path / "note.txt"
    target.write_bytes(b"original")
    failure = OSError(errno.EIO, "injected failure", str(target))
    original_temporary_file = file_tools.tempfile.NamedTemporaryFile

    def fail(*_args, **_kwargs):
        raise failure

    if failure_point == "replace":
        monkeypatch.setattr(file_tools.os, "replace", fail)
    else:

        def failing_stage(*args, **kwargs):
            stage = original_temporary_file(*args, **kwargs)
            stage.write = fail
            return stage

        monkeypatch.setattr(file_tools.tempfile, "NamedTemporaryFile", failing_stage)
    with pytest.raises(ToolInvocationError, match="Write completion is unknown") as error:
        tools["write_file"].invoke(path="note.txt", content="replacement")
    assert error.value.__cause__.__cause__ is failure
    assert str(tmp_path) not in str(error.value)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("configured", [None, "missing", "file"])
def test_wrapper_requires_existing_server_workspace(monkeypatch, tmp_path, configured):
    if configured is None:
        monkeypatch.delenv("EXAMPLE_CHAT_WORKSPACE", raising=False)
    else:
        target = tmp_path / configured
        if configured == "file":
            target.touch()
        monkeypatch.setenv("EXAMPLE_CHAT_WORKSPACE", str(target))
    with pytest.raises(ValueError, match="EXAMPLE_CHAT_WORKSPACE"):
        pipeline_wrapper.PipelineWrapper().setup()


async def test_agent_marks_file_error_and_recovers(tools):
    agent = Agent(
        chat_generator=ToolGenerator("read_file", {"path": "missing.txt"}),
        tools=list(tools.values()),
        raise_on_tool_invocation_failure=False,
    )
    output = [item async for item in stream_agent_chat(agent, [ChatMessage.from_user("Read missing.txt")])]
    events = [item for item in output if isinstance(item, PipelineEvent)]
    assert len(events) == 2
    assert events[0].data["call_id"] == events[1].data["call_id"]
    assert events[0].data["call_id"].endswith(":actual-call")
    assert events[1].data["error"] is True
    assert "missing.txt" in events[1].data["result"] and "Use list_files" in events[1].data["result"]
    assert "".join(item.content for item in output if isinstance(item, StreamingChunk)) == "Done"


async def test_deployed_example_loads_helper_and_streams_real_file_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-for-model-requests")
    monkeypatch.setenv("EXAMPLE_CHAT_WORKSPACE", str(tmp_path))
    path = Path(pipeline_wrapper.__file__).parent
    wrapper = prepare_pipeline_files(
        "workspace_file_tools",
        {name: (path / name).read_text() for name in ["pipeline_wrapper.py", "file_tools.py"]},
        save_files=False,
    ).wrapper
    registry = _PipelineRegistry()
    registry.add("workspace", wrapper)
    app = FastAPI()
    app.include_router(create_openai_router(registry))
    text = "# Summary\nCaffè ☕\n"
    requests = [
        ("list_files", {}, False),
        ("read_file", {"path": "missing.txt"}, True),
        ("write_file", {"path": "summary.txt", "content": text}, False),
        ("read_file", {"path": "summary.txt"}, False),
    ]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for name, arguments, fails in requests:
            wrapper.pipeline = Agent(
                chat_generator=ToolGenerator(name, arguments),
                tools=wrapper.pipeline.tools,
                raise_on_tool_invocation_failure=False,
            )
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "workspace", "messages": [{"role": "user", "content": name}], "stream": True},
            )
            assert response.status_code == 200
            frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            events = [frame["event"] for frame in frames if "event" in frame]
            assert len(events) == 2
            start, result = [event["data"] for event in events]
            assert start["call_id"] == result["call_id"]
            assert start["call_id"].endswith(":actual-call")
            assert start["name"] == name and json.loads(start["arguments"]) == arguments
            assert result["error"] is fails
            if name == "read_file":
                assert "missing.txt" in result["result"] if fails else result["result"] == text
            assert '"content":"Done"' in response.text
            assert '"finish_reason":"stop"' in response.text
    assert (tmp_path / "summary.txt").read_bytes() == text.encode("utf-8")
