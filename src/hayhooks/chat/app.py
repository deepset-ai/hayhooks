from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import ClassVar, Literal
from urllib.parse import urlsplit

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Size
from textual.message import Message
from textual.widgets import Button, Collapsible, Markdown, Static, TextArea
from textual.worker import Worker

from hayhooks.chat.client import ChatClient, ChatError
from hayhooks.chat.events import ChatEvent, ResponseComplete, ResponseDelta, ToolFinished, ToolStarted
from hayhooks.tui.theme import apply_theme

ROOMY_TERMINAL_HEIGHT = 32
COMPACT_TERMINAL_HEIGHT = 24
FULL_SHORTCUT_WIDTH = 80
SHORTCUTS = "↵ send · Ctrl+J line · Esc stop · Ctrl+End latest · Ctrl+T theme · Ctrl+Q quit"
INCOMPLETE_HISTORY = (
    "This incomplete turn is not included in the next request's history. Edit or copy the request before sending again."
)


class MessageComposer(TextArea):
    BINDINGS: ClassVar = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("ctrl+j,shift+enter", "newline", "New line", key_display="Ctrl+J", priority=True),
    ]

    class Submitted(Message):
        pass

    def action_submit(self) -> None:
        self.post_message(self.Submitted())

    def action_newline(self) -> None:
        self.replace("\n", *self.selection, maintain_selection_offset=False)


class ToolCard(Collapsible):
    def __init__(self, event: ToolStarted) -> None:
        self.call_id = event.call_id
        self.label = f"{event.actor} · {event.name}"
        self.state: Literal["Running", "Success", "Error", "Interrupted"] = "Running"
        self.arguments = TextArea(event.arguments, read_only=True, show_cursor=False, classes="tool-input")
        self.output = TextArea("Waiting for a result…", read_only=True, show_cursor=False, classes="tool-output")
        super().__init__(
            Static("ARGUMENTS", classes="detail-label"),
            self.arguments,
            Static("RESULT", classes="detail-label"),
            self.output,
            title=f"{self.label} · {self.state}",
            classes="tool-card running",
        )
        self.tooltip = f"Call {event.call_id} · Enter or click to expand"

    def finish(self, event: ToolFinished) -> None:
        self.state = "Error" if event.error else "Success"
        self.output.load_text(event.result)
        self.remove_class("running")
        self.set_class(event.error, "error")
        self.title = f"{self.label} · {self.state}"
        if event.error:
            self.collapsed = False

    def interrupt(self) -> None:
        if self.state == "Running":
            self.state = "Interrupted"
            self.remove_class("running")
            self.add_class("interrupted")
            self.title = f"{self.label} · {self.state} · outcome unknown"
            self.output.load_text("No tool result was received. The operation's completion is unknown.")


@dataclass
class ResponseView:
    markdown: Markdown
    actor: str
    text: str = ""


@dataclass
class Turn:
    prompt: str
    state: Literal["active", "stopped", "complete", "failed"] = "active"
    worker: Worker | None = None

    def finish(self, state: Literal["complete", "failed"]) -> None:
        if self.state == "active":
            self.state = state


class ChatApp(App):
    CSS_PATH = "chat.tcss"
    BINDINGS: ClassVar = [
        Binding("escape", "stop", "Stop", priority=True),
        Binding("ctrl+end", "latest", "Latest", priority=True),
        Binding("ctrl+q", "quit_chat", "Quit", priority=True),
        Binding("ctrl+c", "copy_or_quit", show=False, priority=True),
        Binding("ctrl+t", "toggle_theme", "Theme", priority=True),
    ]

    class Progress(Message):
        def __init__(self, turn: Turn, text: str) -> None:
            super().__init__()
            self.turn = turn
            self.text = text

    class Output(Message):
        def __init__(self, turn: Turn, event: ChatEvent) -> None:
            super().__init__()
            self.turn = turn
            self.event = event

    class Finished(Message):
        def __init__(self, turn: Turn, text: str, *, failed: bool = False) -> None:
            super().__init__()
            self.turn = turn
            self.text = text
            self.failed = failed

    def __init__(self, runner: ChatClient) -> None:
        super().__init__()
        apply_theme(self)
        self.runner = runner
        self.turn: Turn | None = None
        self.quit_requested = False
        self.responses: dict[str, ResponseView] = {}
        self.tool_cards: dict[str, ToolCard] = {}
        self.dirty: set[str] = set()
        self.last_controller_text: str | None = None
        server = urlsplit(self.runner.server_url)
        self.server_identity = f"{server.scheme}://{server.netloc.rsplit('@', 1)[-1]}"

    def compose(self) -> ComposeResult:
        with Vertical(id="chat"):
            yield Static(f"{self.runner.pipeline} · {self.server_identity}", id="workspace", markup=False)
            with VerticalScroll(id="transcript"):
                yield Markdown(
                    "## What would you like to work on?\n\n"
                    "Chat with your deployed Agent or pipeline. Its tools run on the server.",
                    id="welcome",
                )
            yield Static("Ready", id="status", markup=False)
            yield MessageComposer(soft_wrap=True, id="composer", show_line_numbers=False)
            with Horizontal(id="actions"):
                yield Button("Send", id="send", compact=True)
                yield Button("Stop", id="stop", disabled=True, compact=True)
            yield Static(SHORTCUTS, id="shortcuts", markup=False)

    def on_mount(self) -> None:
        self._update_layout(self.size)
        self.query_one("#composer", TextArea).focus()
        self.query_one("#composer", TextArea).border_title = "Message"
        self.query_one(
            "#workspace", Static
        ).tooltip = f"Deployment · {self.runner.pipeline}\nServer · {self.server_identity}"
        self.query_one("#shortcuts", Static).tooltip = "Enter sends · " + SHORTCUTS + " · Shift+Enter also adds a line"
        self.set_interval(0.05, self._flush_responses)

    def on_resize(self, event: events.Resize) -> None:
        self._update_layout(event.size)

    def _update_layout(self, size: Size) -> None:
        self.screen.set_class(size.height >= ROOMY_TERMINAL_HEIGHT, "roomy")
        self.screen.set_class(size.height < COMPACT_TERMINAL_HEIGHT, "compact")
        shortcuts = self.query("#shortcuts")
        if shortcuts:
            shortcuts.first(Static).update(
                SHORTCUTS if size.width >= FULL_SHORTCUT_WIDTH else "Enter send · Ctrl+J line · Esc stop"
            )

    async def _append(self, role: str, text: str) -> Markdown:
        transcript = self.query_one("#transcript", VerticalScroll)
        follow = transcript.is_vertical_scroll_end
        entry = Vertical(classes="entry " + (role.lower() if role in {"You", "Error"} else "assistant"))
        markdown = Markdown(text)
        await transcript.mount(entry)
        await entry.mount(Static(role, classes="role", markup=False), markdown)
        if follow:
            transcript.scroll_end(animate=False)
        return markdown

    async def _flush_responses(self) -> None:
        if not self.dirty:
            return
        transcript = self.query_one("#transcript", VerticalScroll)
        follow = transcript.is_vertical_scroll_end
        dirty, self.dirty = self.dirty, set()
        responses = [self.responses[message_id] for message_id in dirty]
        for response in responses:
            await response.markdown.update(response.text)
        if follow:
            transcript.scroll_end(animate=False)

    @on(MessageComposer.Submitted)
    @on(Button.Pressed, "#send")
    async def action_send(self) -> None:
        composer = self.query_one("#composer", TextArea)
        prompt = composer.text.strip()
        if self.turn is not None or not prompt:
            return
        self.turn = Turn(prompt)
        self.responses = {}
        self.tool_cards = {}
        self.dirty.clear()
        self.last_controller_text = None
        welcome = self.query("#welcome")
        if welcome:
            await welcome.remove()
        await self._append("You", prompt)
        composer.clear()
        self.query_one("#send", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        self.query_one("#status", Static).update("Thinking")
        self.turn.worker = self._run_turn(prompt, self.turn)

    @work(exit_on_error=False)
    async def _run_turn(self, prompt: str, turn: Turn) -> None:
        def progress(text: str) -> None:
            if turn.state != "stopped":
                self.post_message(self.Progress(turn, text))

        def emit(event: ChatEvent) -> None:
            if turn.state == "stopped":
                return
            if isinstance(event, ResponseComplete) and event.final:
                turn.finish("complete")
            self.post_message(self.Output(turn, event))

        try:
            result = await self.runner.run(prompt, progress, emit=emit)
        except asyncio.CancelledError:
            finished = self.Finished(turn, "Reception ended before the response completed.", failed=True)
        except ChatError as error:
            finished = self.Finished(turn, str(error), failed=True)
        except Exception:
            finished = self.Finished(
                turn,
                "Chat could not finish this request. Check the server logs and affected files before retrying.",
                failed=True,
            )
        else:
            finished = self.Finished(turn, result)
        turn.finish("failed" if finished.failed else "complete")
        self.post_message(finished)

    def on_chat_app_progress(self, message: Progress) -> None:
        if message.turn is self.turn and message.turn.state != "stopped":
            self.query_one("#status", Static).update(message.text)

    async def on_chat_app_output(self, message: Output) -> None:
        if message.turn is not self.turn or message.turn.state == "stopped":
            return
        event = message.event
        if isinstance(event, (ResponseDelta, ResponseComplete)):
            if event.message_id not in self.responses:
                self.responses[event.message_id] = ResponseView(await self._append(event.actor, ""), event.actor)
            response = self.responses[event.message_id]
            response.text = response.text + event.text if isinstance(event, ResponseDelta) else event.text
            self.dirty.add(event.message_id)
            if isinstance(event, ResponseComplete) and event.final:
                self.last_controller_text = event.text
        else:
            transcript = self.query_one("#transcript", VerticalScroll)
            follow = transcript.is_vertical_scroll_end
            if isinstance(event, ToolStarted):
                card = ToolCard(event)
                self.tool_cards[event.call_id] = card
                await transcript.mount(card)
            elif event.call_id in self.tool_cards:
                self.tool_cards[event.call_id].finish(event)
            if follow:
                transcript.scroll_end(animate=False)

    async def on_chat_app_finished(self, message: Finished) -> None:
        if message.turn is not self.turn:
            return
        await self._flush_responses()
        stopped = message.turn.state == "stopped"
        for card in self.tool_cards.values():
            card.interrupt()
        if stopped or message.failed:
            for response in self.responses.values():
                entry = response.markdown.parent
                assert entry is not None
                entry.query_one(".role", Static).update(f"{response.actor} · Incomplete")
            reason = "Stopped receiving. Server work may still be running." if stopped else message.text
            await self._append(
                "Hayhooks" if stopped else "Error", f"Response incomplete. {reason}\n\n{INCOMPLETE_HISTORY}"
            )
            composer = self.query_one("#composer", TextArea)
            if not composer.text:
                composer.load_text(message.turn.prompt)
        elif message.text != self.last_controller_text:
            await self._append("Agent", message.text)
        self.turn = None
        self.query_one("#send", Button).disabled = False
        self.query_one("#stop", Button).disabled = True
        self.query_one("#status", Static).update("Ready")
        if self.quit_requested:
            self.exit()
        else:
            self.query_one("#composer", TextArea).focus()

    @on(Button.Pressed, "#stop")
    def action_stop(self) -> None:
        if self.turn is None or self.turn.state != "active":
            return
        self.turn.state = "stopped"
        self.query_one("#status", Static).update("Stopping reception")
        self.query_one("#stop", Button).disabled = True
        if self.turn.worker is not None:
            self.turn.worker.cancel()

    def action_latest(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def on_text_selected(self) -> None:
        if text := self.screen.get_selected_text():
            self.copy_to_clipboard(text)

    @on(TextArea.SelectionChanged)
    def copy_editor_selection(self, event: TextArea.SelectionChanged) -> None:
        editor = event.text_area
        if editor.has_focus and self.mouse_captured is not editor and editor.selected_text:
            editor.action_copy()

    def on_mouse_up(self) -> None:
        if isinstance(self.focused, TextArea) and self.focused.selected_text:
            self.focused.action_copy()

    def action_copy_or_quit(self) -> None:
        if isinstance(self.focused, TextArea) and self.focused.selected_text:
            self.focused.action_copy()
        elif text := self.screen.get_selected_text():
            self.copy_to_clipboard(text)
        else:
            self.action_quit_chat()

    def action_quit_chat(self) -> None:
        if self.turn is None:
            self.exit()
        else:
            self.quit_requested = True
            self.action_stop()

    def action_toggle_theme(self) -> None:
        self.theme = "hayhooks-light" if self.theme == "hayhooks-dark" else "hayhooks-dark"

    def on_unmount(self) -> None:
        if self.turn is not None and self.turn.state == "active":
            self.turn.state = "stopped"
            if self.turn.worker is not None:
                self.turn.worker.cancel()
