import asyncio
import json
from contextlib import suppress
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

pytest.importorskip("textual")
from textual import events
from textual.widgets import Button, Collapsible, Markdown, Static, TextArea

from hayhooks.chat.app import ChatApp, ToolCard
from hayhooks.chat.client import ChatClient, ChatError
from hayhooks.chat.events import ResponseComplete, ResponseDelta, ToolFinished, ToolStarted
from hayhooks.cli import hayhooks_cli


class BlockingRunner:
    pipeline = "test-agent"
    server_url = "http://localhost:1416"

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.prompts = []

    async def run(self, prompt, progress, *, emit):
        self.prompts.append(prompt)
        self.entered.set()
        progress("Server · working")
        try:
            await self.release.wait()
        finally:
            await asyncio.sleep(0)
            self.closed.set()
        answer = "## Result\n\n```python\nprint('hello')\n```\n\n" + "Long answer.\n\n" * 30
        emit(ResponseComplete("answer", "Agent", answer, final=True))
        return answer


async def settle(app, pilot):
    for _ in range(100):
        await pilot.pause(0.02)
        if app.turn is None:
            return
    message = "turn did not settle"
    raise AssertionError(message)


@pytest.mark.parametrize("size", [(60, 20), (100, 32)])
async def test_multiline_send_output_scroll_and_stop(size):
    runner = BlockingRunner()
    app = ChatApp(runner)
    async with app.run_test(size=size) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("First line\nSecond line")
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert runner.entered.is_set()
        assert runner.prompts == ["First line\nSecond line"]
        assert app.query_one("#send", Button).disabled
        assert "working" in str(app.query_one("#status", Static).render())
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert runner.closed.is_set()
        await settle(app, pilot)
        assert len(runner.prompts) == 1
        assert any("Stopped receiving." in widget.source for widget in app.query(Markdown))
        runner.release.set()
        assert not app.query_one("#send", Button).disabled
        composer.load_text("Now complete")
        await pilot.press("enter")
        await settle(app, pilot)
        assert any("print('hello')" in widget.source for widget in app.query(Markdown))
        assert app.query_one("#transcript").max_scroll_y > 0
        assert composer.region.width > 30 and composer.region.height >= 3
        assert app.query_one("#send").region.right <= size[0]


async def test_quit_closes_response_before_exit():
    runner = BlockingRunner()
    app = ChatApp(runner)
    async with app.run_test() as pilot:
        app.query_one("#composer", TextArea).load_text("Start")
        await pilot.press("enter")
        await pilot.pause(0.1)
        await pilot.press("ctrl+q")
        await pilot.pause(0.1)
        assert runner.closed.is_set() and app.quit_requested
        runner.release.set()
        await settle(app, pilot)


def test_chat_help_is_lazy():
    result = CliRunner().invoke(hayhooks_cli, ["chat", "--help"])
    assert result.exit_code == 0
    assert "pipeline" in result.stdout.lower() and "required" in result.stdout
    assert CliRunner().invoke(hayhooks_cli, ["chat"]).exit_code != 0


def test_chat_reuses_configured_server_address(monkeypatch):
    from hayhooks.settings import settings

    monkeypatch.setattr(settings, "host", "127.0.0.1")
    monkeypatch.setattr(settings, "port", 14321)
    monkeypatch.setattr(settings, "use_https", True)
    captured = []
    monkeypatch.setattr(
        "hayhooks.chat.client.ChatClient",
        lambda server_url, pipeline, **kwargs: SimpleNamespace(pipeline=pipeline, server_url=server_url, **kwargs),
    )
    monkeypatch.setattr(
        "hayhooks.chat.app.ChatApp",
        lambda runner: SimpleNamespace(run=lambda: captured.append(runner)),
    )
    monkeypatch.setattr(settings, "disable_ssl", True)
    result = CliRunner().invoke(hayhooks_cli, ["chat", "my-agent"])
    assert result.exit_code == 0, result.output
    assert captured[0].pipeline == "my-agent"
    assert captured[0].verify_ssl is False
    assert captured[0].server_url == "https://127.0.0.1:14321"


class StreamingRunner(BlockingRunner):
    def __init__(self, text="Early "):
        super().__init__()
        self.text = text
        self.tool_output = "\n".join(f"line {i} [bold]raw[/bold]" for i in range(150))

    async def run(self, prompt, progress, *, emit):
        self.prompts.append(prompt)
        emit(ResponseDelta("answer", "Agent", self.text))
        emit(ToolStarted("call-1", "Agent", "read_file", '{"path": "[file].txt"}'))
        emit(ToolFinished("call-1", self.tool_output))
        emit(ToolStarted("call-2", "Agent", "read_file", '{"path": "missing.txt"}'))
        emit(ToolFinished("call-2", "File missing [literal]", error=True))
        self.entered.set()
        await self.release.wait()
        emit(ResponseDelta("answer", "Agent", "reply"))
        emit(ResponseComplete("answer", "Agent", self.text + "reply", final=True))
        return self.text + "reply"


async def test_enter_newline_live_deltas_and_expandable_full_tool_results():
    runner = StreamingRunner()
    app = ChatApp(runner)
    async with app.run_test(size=(100, 36)) as pilot:
        composer = app.query_one("#composer", TextArea)
        await pilot.press("h", "i", "ctrl+j", "t", "h", "e", "r", "e", "shift+enter", "!")
        assert composer.text == "hi\nthere\n!"
        assert runner.prompts == []
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert runner.entered.is_set() and app.turn is not None
        assert runner.prompts == ["hi\nthere\n!"]
        assert any(widget.source == "Early " for widget in app.query(Markdown))
        success = app.tool_cards["call-1"]
        error = app.tool_cards["call-2"]
        assert success.state == "Success" and success.collapsed
        assert error.state == "Error" and not error.collapsed
        assert error.output.text == "File missing [literal]"
        title = success.query_one("CollapsibleTitle")
        title.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not success.collapsed
        assert len(runner.prompts) == 1
        assert success.arguments.text == '{"path": "[file].txt"}'
        assert success.output.text == runner.tool_output
        success.arguments.focus()
        await pilot.press("f7")
        assert app.clipboard == success.arguments.text
        success.output.focus()
        await pilot.press("down", "shift+down")
        await pilot.pause()
        assert success.output.scroll_y > 0 and not success.output.selected_text
        await pilot.press("f7")
        assert app.clipboard == runner.tool_output
        await pilot.press("ctrl+c")
        assert app.turn is not None and not app.quit_requested
        await pilot.press("pagedown", "pagedown")
        await pilot.pause()
        assert success.output.scroll_y > 0
        runner.release.set()
        await settle(app, pilot)
        assert sum(widget.source == "Early reply" for widget in app.query(Markdown)) == 1
        assert len(app.query(ToolCard)) == 2


async def test_streaming_preserves_reader_scroll_and_latest_resumes_following():
    runner = StreamingRunner("Paragraph.\n\n" * 60)
    app = ChatApp(runner)
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#composer", TextArea).load_text("Start")
        await pilot.press("enter")
        await pilot.pause(0.3)
        transcript = app.query_one("#transcript")
        assert transcript.max_scroll_y > 50
        transcript.scroll_home(animate=False)
        await pilot.pause()
        assert transcript.scroll_y == 0
        runner.release.set()
        await settle(app, pilot)
        await pilot.pause()
        assert transcript.scroll_y == 0
        await pilot.press("ctrl+end")
        await pilot.pause()
        assert transcript.is_vertical_scroll_end


async def test_live_terminal_resize_preserves_draft_tools_and_reader_scroll():
    runner = StreamingRunner("Paragraph.\n\n" * 60)
    app = ChatApp(runner)
    async with app.run_test(size=(160, 48)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("Start")
        await pilot.press("enter")
        await pilot.pause(0.2)
        card = app.tool_cards["call-1"]
        card.query_one("CollapsibleTitle").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not card.collapsed
        draft = "Keep this draft\nand this second line"
        composer.load_text(draft)
        transcript = app.query_one("#transcript")
        transcript.scroll_home(animate=False)
        await pilot.pause()
        for width, height in [(160, 48), (100, 32), (80, 24), (60, 20), (40, 12), (80, 12), (160, 48)]:
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            assert composer.text == draft
            assert composer.region.height >= 3
            assert composer.content_region.height >= 1
            assert composer.region.width >= width - 4
            assert transcript.region.height >= 4
            assert transcript.region.bottom <= composer.region.y
            assert transcript.scroll_y == 0
            assert transcript.max_scroll_x == 0
            for identifier in ("composer", "send", "stop"):
                widget = app.query_one("#" + identifier)
                assert app.screen.region.contains_region(widget.region), (width, height, identifier, widget.region)
            assert app.query_one("#actions").region.y >= composer.region.bottom
            assert app.query_one("#status").region.y == app.query_one("#send").region.y
            assert app.query_one("#send").region.y == app.query_one("#stop").region.y
            assert card.output.region.height <= max(3, height // 3 + 1)
        card.output.focus()
        await pilot.press("pagedown", "pagedown")
        await pilot.pause()
        assert card.output.scroll_y > 0
        await pilot.press("f7")
        assert app.clipboard == runner.tool_output
        assert composer.text == draft
        assert card.output.content_region.height >= 1
        await pilot.press("escape")
        await settle(app, pilot)
        assert composer.text == draft
        assert not app.query_one("#send", Button).disabled
        assert app.query_one("#stop", Button).disabled


@pytest.mark.parametrize("theme", ["hayhooks-dark", "hayhooks-light"])
async def test_tool_editors_reflow_wrapped_text_after_terminal_resize(theme):
    app = ChatApp(BlockingRunner())
    app.theme = theme
    async with app.run_test(size=(160, 48)) as pilot:
        text = "A long result wraps to match the available terminal width. " * 8
        card = ToolCard(ToolStarted("wrapped", "Agent", "read_file", text))
        await app.query_one("#transcript").mount(card)
        card.finish(ToolFinished("wrapped", text))
        card.collapsed = False
        await pilot.pause()
        wide_height = card.output.region.height
        assert wide_height >= card.output.virtual_size.height
        await pilot.resize_terminal(60, 48)
        await pilot.pause()
        assert card.output.region.height > wide_height
        assert card.output.virtual_size.height > wide_height
        assert card.output.text == text and card.arguments.text == text
        await pilot.resize_terminal(40, 12)
        await pilot.pause()
        card.output.focus()
        await pilot.pause()
        assert card.output.region.height <= 4
        assert card.output.content_region.height >= 1
        assert card.output.max_scroll_y > 0
        await pilot.press("f7")
        assert app.clipboard == text
        await pilot.resize_terminal(160, 48)
        composer = app.query_one("#composer", TextArea)
        composer.focus()
        await pilot.pause()
        assert card.output.region.height == wide_height
        assert composer.region.height >= 5


async def test_stop_remains_incomplete_when_runner_suppresses_cancellation():
    class CancellationSuppressingRunner(BlockingRunner):
        async def run(self, prompt, progress, *, emit):
            emit(ResponseDelta("answer", "Agent", "Partial answer"))
            emit(ToolStarted("call-1", "Agent", "write_file", "{}"))
            self.entered.set()
            with suppress(asyncio.CancelledError):
                await self.release.wait()
            progress("Late progress")
            emit(ResponseDelta("answer", "Agent", "Late delta"))
            emit(ToolFinished("call-1", "Late tool result"))
            emit(ToolStarted("late", "Agent", "Late tool", "{}"))
            emit(ResponseComplete("answer", "Agent", "Late answer", final=True))
            return "Late answer"

    runner = CancellationSuppressingRunner()
    app = ChatApp(runner)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("Start")
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert runner.entered.is_set()
        await pilot.press("escape")
        await settle(app, pilot)
        transcript = "\n".join(widget.source for widget in app.query(Markdown))
        assert "Partial answer" in transcript
        assert "Late delta" not in transcript and "Late answer" not in transcript
        card = app.query_one(ToolCard)
        assert card.state == "Interrupted" and "Late tool result" not in card.output.text
        assert "Stopped receiving." in transcript and "Response incomplete." in transcript
        assert composer.text == "Start"


async def test_stop_queued_with_final_event_preserves_success():
    class ReturningRunner(BlockingRunner):
        async def run(self, prompt, progress, *, emit):
            app.post_message(events.Key("escape", "\x1b"))
            emit(ResponseComplete("answer", "Agent", "Completed answer", final=True))
            return "Completed answer"

    app = ChatApp(ReturningRunner())
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("Start")
        await app.action_send()
        await settle(app, pilot)
        transcript = "\n".join(widget.source for widget in app.query(Markdown))
        assert "Completed answer" in transcript
        assert "Response incomplete." not in transcript
        assert composer.text == ""


async def test_runner_failure_after_final_event_still_shows_incomplete_guidance():
    class FailingRunner(BlockingRunner):
        async def run(self, prompt, progress, *, emit):
            emit(ResponseComplete("answer", "Agent", "Received answer", final=True))
            message = "The response could not be accepted."
            raise ChatError(message)

    app = ChatApp(FailingRunner())
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("Start")
        await pilot.press("enter")
        await settle(app, pilot)
        transcript = "\n".join(widget.source for widget in app.query(Markdown))
        assert "Received answer" in transcript
        assert "Response incomplete. The response could not be accepted." in transcript
        assert composer.text == "Start"


async def test_copy_on_select_chat_and_composer_after_mouse_release():
    app = ChatApp(BlockingRunner())
    async with app.run_test(size=(100, 36)) as pilot:
        await app.query_one("#welcome").remove()
        markdown = await app._append("Agent", "Select this chat text.")
        await pilot.pause()
        paragraph = markdown.query_one("MarkdownParagraph")
        app.copy_to_clipboard("previous clipboard")
        await pilot.mouse_down(paragraph, offset=(0, 0))
        await pilot.hover(paragraph, offset=(5, 0))
        await pilot.pause()
        assert app.clipboard == "previous clipboard"
        await pilot.mouse_up(paragraph, offset=(5, 0))
        await pilot.pause()
        assert app.clipboard == "Select"
        await pilot.press("ctrl+c")
        assert app.is_running and not app.quit_requested
        await markdown.update("A streamed response update.")
        await pilot.pause()
        assert app.clipboard == "Select"
        await pilot.click("#workspace")
        assert app.clipboard == "Select"
        composer = app.query_one("#composer", TextArea)
        composer.load_text("draft text")
        await pilot.pause()
        x = composer.content_region.x - composer.region.x
        y = composer.content_region.y - composer.region.y
        await pilot.mouse_down(composer, offset=(x, y))
        await pilot.hover(composer, offset=(x + 5, y))
        await pilot.pause()
        assert app.clipboard == "Select"
        await pilot.mouse_up(composer, offset=(x + 5, y))
        await pilot.pause()
        assert app.clipboard == "draft"
        await pilot.press("right")
        assert not composer.selected_text
        assert app.clipboard == "draft"
        app.copy_to_clipboard("keyboard clipboard")
        await pilot.press("home", *["shift+right"] * 5)
        assert composer.selected_text == "draft"
        assert app.clipboard == "draft"
        await pilot.press("right")
        assert not composer.selected_text
        assert app.clipboard == "draft"


@pytest.mark.parametrize("theme", ["hayhooks-dark", "hayhooks-light"])
@pytest.mark.parametrize(("editor_name", "separator"), [("arguments", "\r\n"), ("output", "\u2028")])
async def test_tool_selection_copies_after_mouse_release(theme, editor_name, separator):
    app = ChatApp(BlockingRunner())
    async with app.run_test(size=(80, 24)) as pilot:
        app.theme = theme
        await app.query_one("#welcome").remove()
        card = ToolCard(ToolStarted("call", "Agent", "read_file", "Select\r\ntool arguments.\n"))
        await app.query_one("#transcript").mount(card)
        card.finish(ToolFinished("call", "Select\u2028tool result.\u2029"))
        card.collapsed = False
        editor = getattr(card, editor_name)
        editor.focus()
        await pilot.pause()
        x = editor.content_region.x - editor.region.x
        y = editor.content_region.y - editor.region.y
        start, end = (x + 1, y), (x + 3, y + 1)
        expected = f"elect{separator}too"
        for first, last in ((start, end), (end, start)):
            app.copy_to_clipboard("previous clipboard")
            await pilot.mouse_down(editor, offset=first)
            await pilot.hover(editor, offset=last)
            await pilot.pause()
            assert app.clipboard == "previous clipboard"
            await pilot.mouse_up(editor, offset=last)
            await pilot.pause()
            assert editor.selected_text == app.clipboard == expected
        app.copy_to_clipboard("previous clipboard")
        await pilot.press("ctrl+c")
        assert app.is_running and not app.quit_requested and app.clipboard == expected
        await pilot.click(editor, offset=(x + 6, y))
        assert not editor.selected_text
        assert app.clipboard == expected
        await pilot.press("ctrl+c")
        assert not app.is_running


async def test_tool_result_copies_original_line_endings():
    app = ChatApp(BlockingRunner())
    async with app.run_test() as pilot:
        card = ToolCard(ToolStarted("call", "Agent", "read_file", "{}"))
        await app.query_one("#transcript").mount(card)
        card.collapsed = False
        card.output.focus()
        for raw in ("a\r\nb\r\n", "a\nb\n", "a\r\nb\nc\rd\ve\ff\x1cg\x1dh\x1ei\x85j\u2028k\u2029"):
            card.finish(ToolFinished("call", raw))
            assert not card.output.selected_text
            await pilot.press("f7")
            assert card.output.selected_text == app.clipboard == raw
            app.copy_to_clipboard("previous clipboard")
            await pilot.press("ctrl+c")
            assert app.clipboard == raw and app.is_running


@pytest.mark.parametrize("theme", ["hayhooks-dark", "hayhooks-light"])
async def test_compact_layout_keeps_transcript_space_and_safe_server_identity(theme):
    runner = BlockingRunner()
    runner.server_url = "https://user:secret@localhost:1416/private?token=secret"
    app = ChatApp(runner)
    async with app.run_test(size=(80, 24)) as pilot:
        app.theme = theme
        await pilot.pause()
        header = app.query_one("#workspace", Static)
        assert str(header.render()) == "test-agent · https://localhost:1416"
        assert "secret" not in header.tooltip and "/private" not in header.tooltip
        assert header.region.height == 1
        assert app.query_one("#chat").region.width == 80
        assert app.query_one("#transcript").region.height >= 16
        assert app.query_one("#composer").region.height == 3
        assert len(app.query("#shortcuts")) == 1
        shortcuts = app.query_one("#shortcuts", Static)
        assert len(str(shortcuts.render())) <= shortcuts.content_region.width
        assert not app.query("Footer, #brand, #hint")
        for identifier in ("send", "stop"):
            assert app.query_one("#" + identifier).region.height == 1
        await pilot.resize_terminal(40, 12)
        await pilot.pause()
        assert str(header.render()) == "test-agent\nhttps://localhost:1416"
        assert header.region.height == 2
        assert app.query_one("#transcript").region.height >= 4
        assert app.query_one("#composer").region.width == 40
        for identifier in ("send", "stop"):
            assert app.screen.region.contains_region(app.query_one("#" + identifier).region)
        await pilot.resize_terminal(160, 48)
        await pilot.pause()
        assert str(header.render()) == "test-agent · https://localhost:1416"
        assert app.query_one("#composer").region.height == 5


async def test_collapsed_tool_row_keeps_state_visible_with_long_name():
    app = ChatApp(BlockingRunner())
    async with app.run_test(size=(40, 12)) as pilot:
        card = ToolCard(ToolStarted("call", "Agent", "a_very_long_tool_name_that_will_be_truncated", "{}"))
        await app.query_one("#transcript").mount(card)
        await pilot.pause()
        assert card.region.height == 1
        assert card.title.startswith("Running · Agent")
        assert "a_very_long_tool_name_that_will_be_truncated" in card.tooltip
        card.finish(ToolFinished("call", "literal result"))
        await pilot.pause()
        assert card.region.height == 1
        assert card.title.startswith("Success · Agent")
        card.query_one("CollapsibleTitle").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not card.collapsed
        assert card.output.text == "literal result"


class RecoveringRunner(BlockingRunner):
    def __init__(self, error):
        super().__init__()
        self.error = error

    async def run(self, prompt, progress, *, emit):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            emit(ResponseDelta("answer", "Agent", "Received partial answer"))
            emit(ToolStarted("success", "Agent", "read_file", '{"path":"notes.txt"}'))
            emit(ToolFinished("success", "Saved result [literal]"))
            emit(ToolStarted("error", "Agent", "read_file", '{"path":"missing.txt"}'))
            emit(ToolFinished("error", "File missing. List the directory.", error=True))
            emit(ToolStarted("pending", "Agent", "read_file", '{"path":"other.txt"}'))
            self.entered.set()
            await self.release.wait()
            raise self.error
        emit(ResponseComplete("recovery", "Agent", "Recovered answer", final=True))
        return "Recovered answer"


@pytest.mark.parametrize("draft", ["", "Keep my new draft\nwith another line"])
@pytest.mark.parametrize("stop", [False, True])
async def test_incomplete_turn_preserves_output_tools_and_draft_then_recovers(draft, stop):
    safe_error = "The connection was lost. Check server logs and affected files before retrying."
    runner = RecoveringRunner(ChatError(safe_error))
    app = ChatApp(runner)
    async with app.run_test(size=(80, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        prompt = "    print('Read notes.txt')  \n    print('Done')  \n"
        composer.load_text(prompt)
        await pilot.press("enter")
        await pilot.pause(0.15)
        assert runner.entered.is_set()
        composer.load_text(draft)
        success = app.tool_cards["success"]
        error = app.tool_cards["error"]
        pending = app.tool_cards["pending"]
        if stop:
            await pilot.press("escape")
        else:
            runner.release.set()
        await settle(app, pilot)
        assert composer.text == (draft or prompt)
        assert runner.prompts == [prompt]
        assert success.state == "Success" and success.output.text == "Saved result [literal]"
        assert error.state == "Error" and not error.collapsed
        assert pending.state == "Interrupted" and "outcome unknown" in pending.title
        assert "completion is unknown" in pending.output.text
        assert any(widget.source == "Received partial answer" for widget in app.query(Markdown))
        assert any("Agent · Incomplete" in str(widget.render()) for widget in app.query(".role"))
        notice = next(widget.source for widget in app.query(Markdown) if "Response incomplete." in widget.source)
        assert "not included in the next request's history" in notice
        assert ("Stopped receiving. Server work may still be running." if stop else safe_error) in notice
        assert not app.query_one("#send", Button).disabled
        assert app.query_one("#stop", Button).disabled
        composer.load_text("Try a different request")
        await pilot.press("enter")
        await settle(app, pilot)
        assert runner.prompts == [prompt, "Try a different request"]
        assert any(widget.source == "Recovered answer" for widget in app.query(Markdown))
        assert success.output.text == "Saved result [literal]" and not error.collapsed


async def test_unexpected_runner_error_shows_safe_guidance():
    internal_detail = "api_key=private-value /private/internal/service.py"
    runner = RecoveringRunner(RuntimeError(internal_detail))
    runner.release.set()
    app = ChatApp(runner)
    async with app.run_test() as pilot:
        app.query_one("#composer", TextArea).load_text("Read notes.txt")
        await pilot.press("enter")
        await settle(app, pilot)
        transcript = "\n".join(widget.source for widget in app.query(Markdown))
        assert internal_detail not in transcript
        assert "Check the server logs and affected files before retrying." in transcript
        assert "Response incomplete." in transcript


class EventStream(httpx.AsyncByteStream):
    def __init__(self, frames, *, block=False):
        self.frames = frames
        self.block = block
        self.reading = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        for frame in self.frames:
            yield frame
        self.reading.set()
        if self.block:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed.set()


def frame(payload):
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def mock_http(monkeypatch, handler):
    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs)
    )


async def test_stop_preserves_received_text_and_tool_result_queued_for_display(monkeypatch):
    receive_result = asyncio.Event()
    closed = asyncio.Event()

    class FinishingToolStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started = {"call_id": "call-1", "name": "write_file", "actor": "Agent", "arguments": "{}"}
            yield frame({"event": {"type": "hayhooks.tool_call", "data": started}})
            await receive_result.wait()
            app.post_message(events.Key("escape", "\x1b"))
            result = {"call_id": "call-1", "result": "Wrote the file.", "error": False}
            yield frame({"event": {"type": "hayhooks.tool_result", "data": result}})
            yield frame({"choices": [{"delta": {"content": "The write succeeded."}}]})
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    mock_http(
        monkeypatch,
        lambda _request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=FinishingToolStream()
        ),
    )
    client = ChatClient("http://server", "agent")
    app = ChatApp(client)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("Write a file")
        await pilot.press("enter")
        card = app.query_one(ToolCard)
        assert card.state == "Running"
        receive_result.set()
        await settle(app, pilot)
        assert closed.is_set()
        assert card.state == "Success" and card.output.text == "Wrote the file."
        transcript = "\n".join(widget.source for widget in app.query(Markdown))
        assert "The write succeeded." in transcript and "Stopped receiving." in transcript
        assert composer.text == "Write a file" and client.history == []


@pytest.mark.parametrize("key, character", [("escape", "\x1b"), ("ctrl+q", "\x11")])
async def test_stop_or_quit_before_turn_worker_runs_skips_request(monkeypatch, key, character):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        answer = frame({"choices": [{"delta": {"content": "Done"}, "finish_reason": "stop"}]})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=answer)

    mock_http(monkeypatch, handler)
    client = ChatClient("http://server", "agent")
    app = ChatApp(client)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("Try this")
        await app.action_send()
        # Dispatch the priority key binding before yielding to the newly scheduled worker.
        await app.on_event(events.Key(key, character))
        await settle(app, pilot)
        assert requests == [] and client.history == []
        assert composer.text == "Try this"
        assert not app.query_one("#send", Button).disabled
        assert app.query_one("#stop", Button).disabled
        if key == "ctrl+q":
            assert app.quit_requested and not app.is_running
            return
        for blank in ("", " \t\n"):
            composer.load_text(blank)
            await pilot.press("enter")
            assert requests == [] and composer.text == blank
        composer.load_text("Next request")
        await pilot.press("enter")
        await settle(app, pilot)
        assert [request["messages"] for request in requests] == [[{"role": "user", "content": "Next request"}]]
        assert any(widget.source == "Done" for widget in app.query(Markdown))


async def test_side_message_ids_cannot_replace_the_agent_answer(monkeypatch):
    def side_message(message_id, actor, text):
        data = {"message_id": message_id, "actor": actor, "text": text, "final": True}
        return frame({"event": {"type": "hayhooks.message", "data": data}})

    stream = EventStream(
        [
            side_message("answer", "Reviewer", "Checking"),
            frame({"choices": [{"delta": {"content": "Main "}}]}),
            side_message("answer", "Reviewer", "Review complete"),
            side_message("message:answer", "Editor", "Another note"),
            frame({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}),
        ]
    )
    mock_http(
        monkeypatch, lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    client = ChatClient("http://server", "agent")
    app = ChatApp(client)
    async with app.run_test() as pilot:
        app.query_one("#composer", TextArea).load_text("Start")
        await pilot.press("enter")
        await settle(app, pilot)
        rows = [
            (str(entry.query_one(".role").render()), entry.query_one(Markdown).source) for entry in app.query(".entry")
        ]
        assert rows == [
            ("You", "Start"),
            ("Reviewer", "Review complete"),
            ("Agent", "Main answer"),
            ("Editor", "Another note"),
        ]
        assert client.history == [{"role": "user", "content": "Start"}, {"role": "assistant", "content": "Main answer"}]


@pytest.mark.parametrize("key, character", [("escape", "\x1b"), ("ctrl+q", "\x11")])
async def test_queued_stop_or_quit_after_http_completion_preserves_completed_turn(monkeypatch, key, character):
    requests = []
    history_when_key_queued = []

    class CompletingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield frame({"choices": [{"delta": {"content": "Saved successfully"}, "finish_reason": "stop"}]})
            if len(requests) == 1:
                history_when_key_queued.append(list(client.history))
                app.post_message(events.Key(key, character))

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=CompletingStream())

    mock_http(monkeypatch, handler)
    client = ChatClient("http://server", "agent")
    app = ChatApp(client)
    async with app.run_test(size=(80, 24)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("Save this")
        await app.action_send()
        await settle(app, pilot)
        assert history_when_key_queued == [[]]
        transcript = "\n".join(widget.source for widget in app.query(Markdown))
        assert "Saved successfully" in transcript
        assert "Response incomplete." not in transcript
        assert "not included in the next request's history" not in transcript
        assert "retry" not in transcript.lower()
        assert not any("Incomplete" in str(widget.render()) for widget in app.query(".role"))
        assert composer.text == ""
        assert client.history == [
            {"role": "user", "content": "Save this"},
            {"role": "assistant", "content": "Saved successfully"},
        ]
        if key == "ctrl+q":
            assert app.quit_requested and not app.is_running
            return
        assert app.is_running
        composer.load_text("What happened?")
        await pilot.press("enter")
        await settle(app, pilot)
        assert requests[1]["messages"] == [
            {"role": "user", "content": "Save this"},
            {"role": "assistant", "content": "Saved successfully"},
            {"role": "user", "content": "What happened?"},
        ]


async def test_remote_sse_to_eof_tools_and_completed_follow_up_history(monkeypatch):
    requests = []
    stream = EventStream(
        [
            frame(
                {
                    "choices": [
                        {
                            "delta": {"content": "First ", "tool_calls": [{"function": {"name": "never_execute"}}]},
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
            frame(
                {
                    "event": {
                        "type": "hayhooks.tool_call",
                        "data": {
                            "call_id": "real-id",
                            "actor": "Agent",
                            "name": "list_files",
                            "arguments": '{"path":"."}',
                        },
                    }
                }
            ),
            frame(
                {
                    "event": {
                        "type": "hayhooks.tool_result",
                        "data": {"call_id": "real-id", "result": '["a.txt"]', "error": False},
                    }
                }
            ),
            frame(
                {"choices": [{"delta": {"reasoning_content": "private", "content": "answer"}, "finish_reason": "stop"}]}
            ),
        ]
    )

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    mock_http(monkeypatch, handler)
    client = ChatClient("http://server:1416", "custom-agent")
    events = []
    assert await client.run("List", lambda _: None, emit=events.append) == "First answer"
    assert stream.closed.is_set()
    assert isinstance(events[-1], ResponseComplete) and events[-1].final
    assert any(
        isinstance(event, ToolStarted) and event.call_id == "real-id" and '\n  "path"' in event.arguments
        for event in events
    )
    assert any(isinstance(event, ToolFinished) and event.result == '["a.txt"]' for event in events)
    assert "private" not in str(events) and "never_execute" not in str(events)
    await client.run("Follow up", lambda _: None, emit=lambda _: None)
    assert requests[1]["messages"] == [
        {"role": "user", "content": "List"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow up"},
    ]
    assert requests[0]["model"] == "custom-agent" and requests[0]["stream"] is True


async def test_reasoning_summary_can_expand_and_stays_out_of_history(monkeypatch):
    summary = "Checked the **workspace** before answering."
    stream = EventStream(
        [
            frame({"event": {"type": "hayhooks.reasoning_summary", "data": {"text": summary}}}),
            frame(
                {
                    "choices": [
                        {"delta": {"reasoning_content": "private", "content": "Found it."}, "finish_reason": "stop"}
                    ]
                }
            ),
        ]
    )
    mock_http(
        monkeypatch,
        lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream),
    )
    client = ChatClient("http://server", "agent")
    app = ChatApp(client)
    async with app.run_test() as pilot:
        app.query_one("#composer", TextArea).load_text("Find the file")
        await pilot.press("enter")
        await settle(app, pilot)
        card = app.query_one(Collapsible)
        assert card.title == "Reasoning summary" and card.collapsed
        card.query_one("CollapsibleTitle").focus()
        await pilot.press("enter")
        assert not card.collapsed
        assert card.query_one(Markdown).source == summary
        assert [widget.source for widget in app.query(Markdown)] == ["Find the file", summary, "Found it."]
        assert client.history == [
            {"role": "user", "content": "Find the file"},
            {"role": "assistant", "content": "Found it."},
        ]


@pytest.mark.parametrize("status", [404, 501])
async def test_remote_http_failures_do_not_commit_history(monkeypatch, status):
    mock_http(monkeypatch, lambda _request: httpx.Response(status, json={"detail": "Wrapper unavailable"}))
    client = ChatClient("http://server", "missing")
    with pytest.raises(ChatError, match=f"HTTP {status}: Wrapper unavailable"):
        await client.run("Hi", lambda _: None, emit=lambda _: None)
    assert client.history == []


async def test_cancel_blocked_http_read_closes_response_and_discards_history(monkeypatch):
    stream = EventStream([frame({"choices": [{"delta": {"content": "Partial"}}]})], block=True)
    mock_http(
        monkeypatch, lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    client = ChatClient("http://server", "agent")
    events = []
    task = asyncio.create_task(client.run("Hi", lambda _: None, emit=events.append))
    await asyncio.wait_for(stream.reading.wait(), 1)
    assert events == [ResponseDelta("answer", "Agent", "Partial")]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed.is_set() and client.history == []


async def test_invalid_recognized_tool_event_is_a_visible_protocol_error(monkeypatch):
    stream = EventStream([frame({"event": {"type": "hayhooks.tool_result", "data": {"result": "no call ID"}}})])
    mock_http(
        monkeypatch, lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    client = ChatClient("http://server", "agent")
    with pytest.raises(ChatError, match="invalid chat data"):
        await client.run("Hi", lambda _: None, emit=lambda _: None)
    assert client.history == [] and stream.closed.is_set()
