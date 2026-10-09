"""Trace presentation using Textual's native tree and scroll widgets."""

from datetime import datetime, timezone
from typing import Any, ClassVar

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import Footer, Static, Tree

from hayhooks.tui.client import literal

WIDE_TERMINAL = 110
MILLISECONDS_PER_SECOND = 1000
COMFORTABLE_TERMINAL_HEIGHT = 30


def trace_tags(item: dict[str, Any]) -> dict[str, str]:
    return {tag["key"]: tag["value"] for tag in item["tags"]}


def trace_state(item: dict[str, Any]) -> str:
    tags = trace_tags(item)
    if tags.get("hayhooks.success") == "false" or "hayhooks.error.type" in tags:
        return "failed"
    return "running" if item["root_span"]["running"] else "completed"


def trace_kind(item: dict[str, Any]) -> str:
    name = item["root_span"]["name"]
    for pattern, kind in (
        (".undeploy", "undeploy"),
        (".deploy", "deploy"),
        (".openai.", "openai"),
        (".mcp.", "mcp"),
        (".a2a.", "a2a"),
        (".durable.", "durable"),
        (".run", "run"),
    ):
        if pattern in name:
            return kind
    return "other"


def duration(milliseconds: float) -> str:
    return f"{milliseconds / 1000:.2f}s" if milliseconds >= MILLISECONDS_PER_SECOND else f"{milliseconds:.0f}ms"


def started_at(milliseconds: int) -> str:
    try:
        return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, ValueError, OSError):
        return "Unknown"


def all_spans(trace: dict[str, Any]) -> list[dict[str, Any]]:
    spans = []
    pending = [trace["root_span"]]
    while pending:
        span = pending.pop()
        spans.append(span)
        pending.extend(reversed(span["children"]))
    return spans


def slowest_component(trace: dict[str, Any]) -> dict[str, Any] | None:
    return max(
        (span for span in all_spans(trace) if span["name"] == "haystack.component.run" and not span["running"]),
        key=lambda span: span["duration_ms"],
        default=None,
    )


def trace_detail(trace: dict[str, Any], theme: Theme, slow_ms: int = 1000) -> Text:
    text = Text()
    shown = len(all_spans(trace))
    text.append("TRACE\n", style=f"bold {theme.accent}")
    text.append(literal(trace["trace_id"]) + "\n\n", style="bold")
    for label, value in (
        ("Entrypoint", trace["entrypoint"] or "—"),
        ("Kind", trace_kind(trace)),
        ("State", trace_state(trace)),
        ("Started (UTC)", started_at(trace["start_time_ms"])),
        ("Duration", "In progress" if trace["root_span"]["running"] else duration(trace["duration_ms"])),
        ("Spans", trace["span_count"] if shown == trace["span_count"] else f"{shown} of {trace['span_count']} shown"),
    ):
        text.append(label.ljust(18), style=theme.variables["text-muted"])
        text.append(literal(value) + "\n", style=theme.foreground or "")
    slow = slowest_component(trace)
    if slow and slow["duration_ms"] >= slow_ms:
        name = trace_tags(slow).get("haystack.component.name", slow["span_id"])
        text.append("\nSLOWEST COMPONENT\n", style=f"bold {theme.warning}")
        text.append(literal(name) + "  ·  " + duration(slow["duration_ms"]) + "\n", style=theme.foreground or "")
    text.append("\nTAGS\n", style=f"bold {theme.accent}")
    for key, value in trace_tags(trace).items():
        text.append(literal(key) + "\n", style=theme.variables["text-muted"])
        text.append(literal(value) + "\n", style=theme.foreground or "")
    text.append(
        "\nEnter opens the span tree and waterfall.\nRecent process traces; not execution history.",
        style=theme.variables["text-muted"],
    )
    return text


class TraceScreen(Screen):
    BINDINGS: ClassVar = [Binding("escape", "app.close_detail", "Back")]

    def __init__(self, trace: dict[str, Any], slow_ms: int) -> None:
        super().__init__()
        self.trace = trace
        self.slow_ms = slow_ms
        self.nodes: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Static(Text("HAYHOOKS   /   Trace inspector"), classes="detail-heading")
        yield Static("", id="trace-heading", markup=False)
        yield Static("", id="trace-warning", markup=False)
        with Horizontal(id="trace-panes"):
            yield Tree(Text("Spans"), id="span-tree")
            with VerticalScroll(id="span-pane", can_focus=True):
                yield Static("", id="span-detail", markup=False)
        yield Static("↑↓ Select span   Enter Expand / collapse   Tab Tags   Esc Back", id="trace-hint")
        yield Footer(compact=True)

    def on_mount(self) -> None:
        self.set_class(self.size.width < WIDE_TERMINAL, "narrow-trace")
        self.set_class(self.size.height < COMFORTABLE_TERMINAL_HEIGHT, "short-trace")
        self.update_trace(self.trace, force=True)
        self.query_one(Tree).focus()

    def update_trace(self, trace: dict[str, Any], *, force: bool = False) -> None:
        if not force and trace == self.trace:
            return
        tree = self.query_one(Tree)
        selected = tree.cursor_node.data["span_id"] if tree.cursor_node and tree.cursor_node.data else None
        initial = not self.nodes
        expanded = {key for key, node in self.nodes.items() if node.is_expanded}
        scroll = tree.scroll_offset
        self.trace = trace
        heading = Text(
            literal(trace["entrypoint"] or "No entrypoint")
            + "  ·  "
            + trace_state(trace)
            + "  ·  "
            + ("In progress" if trace["root_span"]["running"] else duration(trace["duration_ms"]))
            + "  ·  "
            + literal(trace["trace_id"])
        )
        theme = self.app.current_theme
        slow = slowest_component(trace)
        slow_id = slow["span_id"] if slow and slow["duration_ms"] >= self.slow_ms else None
        tree.reset(Text("Spans"))
        tree.show_root = False
        self.nodes = {}
        pending = [(trace["root_span"], tree.root)]
        while pending:
            span, parent = pending.pop()
            name = trace_tags(span).get("haystack.component.name", span["name"])
            offset = max(0, span["start_time_ms"] - trace["start_time_ms"])
            track = "·" * 16 if span["running"] else self._waterfall(span)
            label = Text(literal(name, 100), style=theme.foreground or "")
            label.append(
                "  " + ("live" if span["running"] else duration(span["duration_ms"])),
                style=theme.accent if span["running"] else theme.variables["text-muted"],
            )
            label.append("  " + track + "  +" + duration(offset), style=theme.variables["text-muted"])
            if span["span_id"] == slow_id:
                label.append("  SLOW", style=f"bold {theme.warning}")
            node = parent.add(
                label, data=span, expand=initial or span["span_id"] in expanded, allow_expand=bool(span["children"])
            )
            self.nodes[span["span_id"]] = node
            pending.extend((child, node) for child in reversed(span["children"]))
        if len(self.nodes) < trace["span_count"]:
            heading.append(f"  ·  {len(self.nodes)} of {trace['span_count']} spans shown")
        self.query_one("#trace-heading", Static).update(heading)
        node = (self.nodes.get(selected) if selected is not None else None) or next(iter(self.nodes.values()))
        tree.call_after_refresh(tree.move_cursor, node)
        tree.call_after_refresh(tree.scroll_to, scroll.x, scroll.y, animate=False)
        self._show_span(node.data)

    def _waterfall(self, span: dict[str, Any]) -> str:
        width = 16
        total = max(1, self.trace["duration_ms"])
        offset = min(width - 1, max(0, int((span["start_time_ms"] - self.trace["start_time_ms"]) / total * width)))
        length = min(width - offset, max(1, round(span["duration_ms"] / total * width)))
        return "·" * offset + "━" * length + "·" * (width - offset - length)

    @on(Tree.NodeHighlighted)
    def span_highlighted(self, event: Tree.NodeHighlighted) -> None:
        if event.node.data:
            self._show_span(event.node.data)

    def _show_span(self, span: dict[str, Any]) -> None:
        theme = self.app.current_theme
        text = Text()
        text.append("SPAN\n", style=f"bold {theme.accent}")
        text.append(literal(span["name"]) + "\n" + literal(span["span_id"]) + "\n\n", style=theme.foreground or "")
        for label, value in (
            ("Started (UTC)", started_at(span["start_time_ms"])),
            ("Offset", duration(max(0, span["start_time_ms"] - self.trace["start_time_ms"]))),
            ("Duration", "In progress" if span["running"] else duration(span["duration_ms"])),
        ):
            text.append(label.ljust(18), style=theme.variables["text-muted"])
            text.append(value + "\n", style=theme.foreground or "")
        text.append("\nTAGS / ERRORS\n", style=f"bold {theme.accent}")
        for key, value in trace_tags(span).items():
            text.append(literal(key) + "\n", style=theme.variables["text-muted"])
            text.append(literal(value) + "\n\n", style=theme.error if "error" in key else theme.foreground)
        self.query_one("#span-detail", Static).update(text)

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < WIDE_TERMINAL, "narrow-trace")
        self.set_class(event.size.height < COMFORTABLE_TERMINAL_HEIGHT, "short-trace")
