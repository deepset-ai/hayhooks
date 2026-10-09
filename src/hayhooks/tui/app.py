"""Browsable, read-only monitoring with one Requests worker in flight."""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import Button, DataTable, Footer, Input, Select, Static, Tab, Tabs

from hayhooks.colors import BRAND_COLOR
from hayhooks.tui.client import DashboardClient, DashboardError, literal
from hayhooks.tui.traces import (
    COMFORTABLE_TERMINAL_HEIGHT,
    WIDE_TERMINAL,
    TraceScreen,
    all_spans,
    duration,
    started_at,
    trace_detail,
    trace_kind,
    trace_state,
    trace_tags,
)

HELP = """Keyboard and mouse

Arrows       Move rows or scroll focused details
Tab / Shift+Tab   Move focus
Enter        Read selected item details
Esc          Return from details/help or leave search
/            Search deployment names or indexed execution IDs
d            Deployments tab
e            Recent runs and stored durable executions
v            Traces tab
Space        Pause / resume refresh
r            Refresh now
[ / ]        Previous / next page
?            This help
t            Toggle light / dark theme
q / Ctrl+C   Exit

Click rows to select; use the wheel to scroll.
Search and filters stay in place when returning from details.
Execution pages scan at most 200 candidates; a sparse page may have a next page.
All deployments: up to 50 durable executions per deployment on each page.
Recent request-bound runs appear on page 1 and disappear after buffer eviction/restart.
c                  Hide the currently buffered traces locally (Traces tab)
Traces: filter entrypoint/kind/state, sort newest/slowest, Enter for span tree.
Span tree: arrows select, Enter expands/collapses, Tab scrollable tags/errors.
Recent traces describe the responding process, not durable execution history.
All network requests are GET. Public results are discarded.
"""


def state_color(state: str, theme: Theme) -> str:
    role = {
        "Healthy": "success",
        "Connected": "success",
        "completed": "success",
        "Degraded": "warning",
        "waiting": "warning",
        "queued": "warning",
        "failed": "error",
        "running": "accent",
    }.get(state)
    return (getattr(theme, role) if role else theme.variables["text-muted"]) or BRAND_COLOR


def detail_text(item: dict[str, Any], *, execution: bool = False, theme: Theme) -> Text:  # noqa: C901, PLR0912
    """Explicit display fields, always literal and bounded; results never appear."""
    fields = (
        (
            ("deployment", "Deployment"),
            ("kind", "Type"),
            ("revision", "Stored revision"),
            ("status", "Status"),
            ("attempt", "Attempt"),
            ("created_at", "Created"),
            ("updated_at", "Updated"),
            ("cancellation_pending", "Cancel pending"),
            ("available_at", "Retry at"),
            ("cancellation_requested_at", "Cancel requested"),
        )
        if execution
        else (
            ("type", "Type"),
            ("execution_mode", "Execution"),
            ("state", "State"),
            ("revision", "Served revision"),
        )
    )
    muted = theme.variables["text-muted"]
    text = Text()
    text.append("EXECUTION\n" if execution else "DEPLOYMENT\n", style=f"bold {theme.accent}")
    text.append(literal(item.get("execution_id" if execution else "name", "")) + "\n", style="bold")
    if not execution and item.get("description"):
        text.append(literal(item["description"]) + "\n", style=muted)
    text.append("\n")
    for field, label in fields:
        if field in item and item[field] is not None:
            value = item[field]
            text.append(label.ljust(18), style=muted)
            text.append(
                literal(value) + "\n", style=state_color(str(value), theme) if field in {"state", "status"} else ""
            )
    text.append("\n")
    if execution:
        for field, label in (("waiting", "Waiting"), ("progress", "Progress"), ("error", "Error")):
            if item.get(field):
                text.append(label.upper() + "\n", style=f"bold {theme.accent}")
                text.append(literal(json.dumps(item[field], ensure_ascii=False, indent=2)) + "\n\n")
        if message := item.get("detail_message"):
            text.append(literal(message) + "\n\n")
        text.append(
            "Creation/update times are not runtime duration.\nResults are not displayed or retained.", style=muted
        )
    elif item.get("health") is not None:
        text.append("PROCESS\n", style=f"bold {theme.accent}")
        text.append(literal(item.get("worker_scope", "Responding process")) + "\n", style=muted)
        health = item["health"]
        for field, label in (
            ("active_executions", "Active executions"),
            ("configured_slots", "Configured slots"),
            ("running_slots", "Running slots"),
            ("draining_slots", "Draining slots"),
            ("draining_runs", "Draining runs"),
            ("maintenance_running", "Maintenance"),
            ("accepting", "Accepting"),
            ("store_error_streak", "Store error streak"),
            ("operational_error", "Operational error"),
        ):
            if field in health:
                text.append(label.ljust(18), style=muted)
                text.append(literal(health[field]) + "\n")
        text.append("\nSTORE\n", style=f"bold {theme.accent}")
        if item.get("store_scope"):
            text.append(literal(item["store_scope"]) + "\n", style=muted)
        for field, value in health.get("counts", {}).items():
            text.append(literal(field).replace("_", " ").capitalize().ljust(18), style=muted)
            text.append(literal(value) + "\n")
        text.append(
            "\nRevision runnable includes queued delayed retries.\nWorker totals describe this process.", style=muted
        )
    else:
        text.append("Loaded confirms registration.\nExternal services are not probed.", style=muted)
    if not execution and "capabilities" in item:
        text.append("\n\nAPI / ELIGIBILITY\n", style=f"bold {theme.accent}")
        labels = {
            "run_api": "Run API",
            "responses": "Responses",
            "chat_completion": "Chat completions",
            "file_upload": "File upload",
            "mcp_eligible": "MCP eligible",
            "a2a_eligible": "A2A eligible",
        }
        values = item["capabilities"]
        value = (
            ", ".join(labels.get(str(part), str(part).replace("_", " ")) for part in values) or "None advertised"
            if isinstance(values, list)
            else values
        )
        text.append(literal(value))
    return text


class DetailScreen(Screen):
    BINDINGS: ClassVar = [Binding("escape", "app.close_detail", "Back")]

    def __init__(self, content: Text, *, help_screen: bool = False) -> None:
        super().__init__()
        self.help_screen = help_screen
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static("HAYHOOKS   /   " + ("Shortcuts" if self.help_screen else "Inspector"), classes="detail-heading")
        with VerticalScroll(id="full-detail", can_focus=True):
            yield Static(self.content, id="full-detail-text", markup=False)
        yield Footer(compact=True)

    def on_mount(self) -> None:
        self.query_one(VerticalScroll).focus()


class DashboardApp(App):
    CSS_PATH = "dashboard.tcss"
    TITLE = "Hayhooks"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "exit", "Exit", priority=True, show=False),
        Binding("slash", "search", "Search"),
        Binding("space", "pause", "Pause", key_display="Space", priority=True),
        Binding("r", "refresh_now", "Refresh"),
        Binding("question_mark", "help", "Help"),
        Binding("d", "view('deployments')", "Deployments", show=False),
        Binding("e", "executions", "Executions"),
        Binding("left_square_bracket", "previous", "Prev", show=False),
        Binding("right_square_bracket", "next", "Next", show=False),
        Binding("escape", "escape", "Back", show=False),
        Binding("t", "theme", "Theme"),
        Binding("v", "view('traces')", "Traces", show=False),
        Binding("c", "clear_traces", "Clear locally", show=False),
    ]

    def __init__(
        self, client: DashboardClient, *, pipeline: str | None = None, interval: float = 2.0, traces: bool = False
    ) -> None:
        super().__init__()
        for name, dark in (("hayhooks-dark", True), ("hayhooks-light", False)):
            self.register_theme(
                Theme(
                    name=name,
                    primary=BRAND_COLOR,
                    accent="#8BAAFF" if dark else BRAND_COLOR,
                    success="#72CCA5" if dark else "#007D35",
                    error="#F18D99" if dark else "#C12636",
                    warning="#E4BD78" if dark else "#855600",
                    foreground="#DBE3EF" if dark else "#252C3C",
                    background="#0F141E" if dark else "#F6F5F1",
                    surface="#171E2B" if dark else "#EEEEE9",
                    panel="#202B3D" if dark else "#E3E8F1",
                    dark=dark,
                    variables={
                        "text-muted": "#91A0B6" if dark else "#636D7D",
                        "text-disabled": "#6E7C91" if dark else "#717B89",
                        "block-cursor-background": "#263C62" if dark else "#D9E3FA",
                        "block-cursor-foreground": "#F4F7FD" if dark else "#182D57",
                        "block-cursor-blurred-background": "#1E2D46" if dark else "#E1E7F2",
                        "block-cursor-text-style": "bold",
                        "footer-key-foreground": "#8BAAFF" if dark else "#315FCE",
                        "footer-description-foreground": "#91A0B6" if dark else "#636D7D",
                    },
                )
            )
        self.theme = "hayhooks-dark"
        self.client = client
        self.pipeline = pipeline
        self.interval = interval
        self.view = "traces" if traces else "deployments"
        self.snapshot: dict[str, Any] = {}
        self.page: dict[str, Any] = {}
        self.trace_page: dict[str, Any] = {}
        self.hidden_traces: set[str] = set()
        self.trace_entrypoints: tuple[str, ...] = ()
        self.items: dict[str, dict[str, Any]] = {}
        self.selected: dict[str, str | None] = {"deployments": pipeline, "executions": None, "traces": None}
        self.details: dict[str, Any] = {}
        self.cursor: str | dict[str, str] | None = None
        self.history: list[str | dict[str, str] | None] = []
        self.searches = {"deployments": "", "executions": "", "traces": ""}
        self.paused = False
        self.in_flight = False
        self.generation = 0
        self.last_good: float | None = None
        self.error = ""
        self.failures = 0
        self.next_poll = 0.0
        self.updating = False
        self.previous_focus = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="masthead"):
            yield Static(Text("HAYHOOKS\nTerminal monitor", style="bold"), id="brand")
            yield Static("Connecting…", id="connection", markup=False)
        yield Tabs(
            Tab("Deployments (d)", id="deployments"),
            Tab("Executions (e)", id="executions"),
            Tab("Traces (v)", id="traces"),
            id="views",
        )
        with Horizontal(id="filters"):
            yield Input(placeholder="/  Search deployments", id="search", compact=True)
            yield Select(
                [("All types", "all"), ("Agent", "agent"), ("Pipeline", "pipeline"), ("Unknown", "unknown")],
                value="all",
                allow_blank=False,
                id="type-filter",
                compact=True,
            )
            yield Select(
                [
                    ("All statuses", "all"),
                    *[(s.title(), s) for s in ("queued", "running", "waiting", "completed", "failed", "canceled")],
                ],
                value="all",
                allow_blank=False,
                id="status-filter",
                compact=True,
            )
            yield Select(
                [
                    ("All kinds", "all"),
                    *[
                        (kind.title(), kind)
                        for kind in ("run", "deploy", "undeploy", "openai", "mcp", "a2a", "durable", "other")
                    ],
                ],
                value="all",
                allow_blank=False,
                id="trace-kind",
                compact=True,
            )
        with Horizontal(id="trace-controls"):
            yield Select([], prompt="All entrypoints", id="trace-entrypoint", compact=True)
            yield Select(
                [("All states", "all"), ("Running", "running"), ("Completed", "completed"), ("Failed", "failed")],
                value="all",
                allow_blank=False,
                id="trace-state",
                compact=True,
            )
            yield Select(
                [("Newest", "newest"), ("Slowest", "slowest")],
                value="newest",
                allow_blank=False,
                id="trace-sort",
                compact=True,
            )
        with Horizontal(id="execution-controls"):
            yield Select([], prompt="All deployments", id="deployment-filter", compact=True)
            yield Input(placeholder="Revision · all retained", id="revision-filter", compact=True)
        yield Static("Connecting…", id="message", markup=False)
        with Horizontal(id="panes"):
            yield DataTable(id="rows", cursor_type="row", zebra_stripes=False)
            with VerticalScroll(id="detail-pane", can_focus=True):
                yield Static("Select a deployment", id="detail", markup=False)
        with Horizontal(id="pagination"):
            yield Button("← Previous", id="previous", disabled=True, compact=True)
            yield Static("Page 1", id="page-label", markup=False)
            yield Button("Next →", id="next", disabled=True, compact=True)
        yield Static("Tab Focus   Enter Details   Arrows Navigate", id="navigation-hint", markup=False)
        yield Footer(compact=True)

    def on_mount(self) -> None:
        self.main_screen = self.screen
        self._widget("#views").active = self.view
        self.main_screen.set_class(self.size.height < COMFORTABLE_TERMINAL_HEIGHT, "compact")
        self._widget("#detail-pane").display = self.size.width >= WIDE_TERMINAL
        self._configure_view()
        self.query_one(DataTable).focus()
        self.set_interval(0.25, self._tick)
        self._tick()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if action == "clear_traces" and self.view != "traces":
            return False
        return not (isinstance(self.focused, Input) and action not in {"escape", "close_detail", "exit"})

    def _widget(self, selector: str):
        return self.main_screen.query_one(selector)

    def _configure_view(self) -> None:
        self.updating = True
        executions = self.view == "executions"
        traces = self.view == "traces"
        self._widget("#type-filter").display = not executions and not traces
        self._widget("#trace-kind").display = traces
        self._widget("#trace-controls").display = traces
        self._widget("#status-filter").display = executions
        self._widget("#execution-controls").display = executions
        self._widget("#pagination").display = executions
        search = self._widget("#search")
        search.value = self.searches[self.view]
        search.placeholder = (
            "/  Search traces / tags"
            if traces
            else "/  Search indexed IDs"
            if executions
            else "/  Search deployment names"
        )
        table = self._widget("#rows")
        table.clear(columns=True)
        columns = (
            ("ID", "Entrypoint", "Kind", "State", "Duration", "Spans", "Stream")
            if traces
            else ("ID", "Deployment", "Mode", "Status", "Duration", "Updated (UTC)")
            if executions
            else ("Name", "Type", "Execution", "State")
        )
        for label in columns:
            width = (
                18 if (executions or traces) and label == "ID" else 8 if executions and label == "Revision" else None
            )
            table.add_column(label, width=width)
        self.items = {}
        self._update_table()
        self.updating = False
        self._invalidate()

    def _invalidate(self) -> None:
        self.generation += 1
        self.next_poll = 0

    def _tick(self) -> None:
        if not self.is_running:
            return
        self._header()
        if isinstance(self.focused, Input):
            navigation = "Tab Next control   Esc Return   Ctrl+C Exit"
        elif self.focused is self._widget("#detail-pane"):
            navigation = "Tab Focus   Esc List   Arrows Scroll"
        else:
            navigation = "Tab Focus   Enter Inspect   ↑↓ Rows   ←→ Columns"
        self._widget("#navigation-hint").update(Text(navigation))
        if not self.paused and not self.in_flight and time.monotonic() >= self.next_poll:
            self._request_refresh()

    def _header(self) -> None:
        age = f"Updated {int(time.monotonic() - self.last_good)}s ago" if self.last_good is not None else "No snapshot"
        state = (
            "Degraded"
            if self.snapshot.get("status") == "Degraded"
            else ("Connected" if self.last_good else "Connecting")
        )
        annotation = "  PAUSED" if self.paused else ""
        if self.error:
            annotation += "  STALE"
        text = Text(
            literal(self.client.label) + "\n",
            style=self.current_theme.variables["text-muted"],
            no_wrap=True,
            overflow="ellipsis",
        )
        text.append("● " + state, style=state_color(state, self.current_theme))
        text.append("  ·  " + age)
        text.append(annotation, style=f"bold {self.current_theme.warning}")
        self._widget("#connection").update(text)
        self._widget("#brand").update(
            Text.assemble(
                ("▎ HAYHOOKS\n", f"bold {self.current_theme.accent}"),
                ("  Terminal monitor", self.current_theme.variables["text-muted"]),
            )
        )

    def _query(self) -> tuple:
        return (
            self.view,
            self.pipeline,
            self.searches["executions"],
            self._widget("#status-filter").value,
            self._widget("#revision-filter").value,
            self.cursor,
            self.selected["executions"],
            self._reading_detail(),
        )

    def _reading_detail(self) -> bool:
        return any(
            isinstance(screen, TraceScreen) or (isinstance(screen, DetailScreen) and not screen.help_screen)
            for screen in self.screen_stack
        ) or (self.focused is self._widget("#detail-pane"))

    def _request_refresh(self) -> None:
        if self.paused or self.in_flight:
            return
        self.in_flight = True
        selected = self.selected["executions"]
        self._fetch(
            self.generation, self._query(), not bool(self.snapshot), self.items.get(selected or "", self.details)
        )

    @work(thread=True, exit_on_error=False)
    def _fetch(self, generation: int, query: tuple, need_status: bool, selected_item: dict) -> None:  # noqa: C901, PLR0912
        view, pipeline, search, status, revision, cursor, selected, reading = query
        snapshot = page = detail = trace_page = None
        error = detail_error = ""
        try:
            if view == "deployments" or need_status:
                snapshot = self.client.status()
            if view == "executions" and not need_status:
                # Independent sources: an operator denial must not hide owner-scoped history.
                try:
                    trace_page = self.client.traces()
                except DashboardError as failure:
                    error = str(failure)
                deployments = [
                    item["name"]
                    for item in self._deployments()
                    if item.get("execution_enumeration")
                    and (pipeline is None or item["name"] == pipeline)
                    and (not isinstance(cursor, dict) or item["name"] in cursor)
                ]
                collected: dict[str, Any] = {
                    "executions": [],
                    "has_more": False,
                    "next_cursor": None,
                    "skipped_controls": 0,
                }
                cursors = {}
                for name in deployments:
                    try:
                        batch = self.client.executions(
                            name,
                            search=search,
                            status=None if status == "all" else status,
                            revision=revision or None,
                            cursor=cursor.get(name) if isinstance(cursor, dict) else cursor,
                        )
                    except DashboardError as failure:
                        error = "; ".join(part for part in (error, str(failure)) if part)
                        collected["executions"].extend(
                            row for row in self.page.get("executions", []) if row["deployment"] == name
                        )
                        previous = self.page.get("next_cursor")
                        previous = previous.get(name) if isinstance(previous, dict) else previous
                        if previous:
                            cursors[name] = previous
                        continue
                    for row in batch["executions"]:
                        item = {**row, "deployment": name}
                        item["_key"] = (
                            item["execution_id"] if pipeline else json.dumps(["durable", name, item["execution_id"]])
                        )
                        collected["executions"].append(item)
                    collected["skipped_controls"] += batch.get("skipped_controls", 0)
                    if batch["has_more"]:
                        cursors[name] = batch["next_cursor"]
                collected["has_more"] = bool(cursors)
                collected["next_cursor"] = cursors.get(pipeline) if pipeline else cursors or None
                page = collected
                ids = [item["_key"] for item in page["executions"]]
                # Local trace selections are resolved by the merged table, never by durable detail routes.
                if selected not in ids and not (selected and reading) and not selected_item.get("_trace"):
                    selected = ids[0] if ids else None
                summary = next((item for item in page["executions"] if item["_key"] == selected), selected_item)
                if selected and summary and not summary.get("_trace"):
                    try:
                        detail = self.client.detail(summary["deployment"], summary["execution_id"])
                    except DashboardError as failure:
                        detail_error = str(failure)
            if view == "traces":
                page = self.client.traces()
        except DashboardError as failure:
            error = "; ".join(part for part in (error, str(failure)) if part)
        finally:
            if self.is_running:
                self.call_from_thread(
                    self._apply_refresh,
                    generation,
                    query,
                    snapshot,
                    page,
                    selected,
                    detail,
                    error,
                    detail_error,
                    trace_page,
                )

    def _apply_refresh(  # noqa: PLR0913, PLR0912, C901
        self,
        generation: int,
        query: tuple,
        snapshot: dict | None,
        page: dict | None,
        selected: str | None,
        detail: dict | None,
        error: str,
        detail_error: str,
        trace_page: dict | None = None,
    ) -> None:
        self.in_flight = False
        if not self.is_running:
            return
        if self.paused or generation != self.generation or query != self._query():
            self.next_poll = 0
            self._tick()
            return
        if error:
            self.error = error
            self.failures += 1
            self._widget("#message").update(Text(error + ". Retaining last snapshot; retrying."))
        else:
            self.error = ""
            self.failures = 0
            self.last_good = time.monotonic()
        if snapshot is not None or page is not None or trace_page is not None:
            if trace_page is not None:
                self.trace_page = trace_page
            if snapshot is not None:
                self.snapshot = snapshot
                deployments = self._deployments()
                options_list = [(literal(item["name"]), item["name"]) for item in deployments]
                options = self._widget("#deployment-filter")
                self.updating = True
                options.set_options(options_list)
                options.value = self.pipeline if self.pipeline in [value for _, value in options_list] else Select.NULL
                self.updating = False
            if page is not None and self.view == "traces":
                self.trace_page = page
                self.hidden_traces.intersection_update(item["trace_id"] for item in page["traces"])
            if page is not None and self.view == "executions":
                self.page = page
                self.selected["executions"] = selected
                summary = next((item for item in page["executions"] if item["_key"] == selected), {})
                if not summary and self.details.get("_key") == selected:
                    summary = self.details
                if detail is not None:
                    self.details = {**summary, **detail}
                    if selected not in [item["_key"] for item in page["executions"]]:
                        self.details["detail_message"] = "Selected execution is outside this page/filter."
                elif detail_error:
                    previous = self.details if self.details.get("_key") == selected else {}
                    self.details = {**summary, **previous, "detail_message": detail_error + ". Detail snapshot stale."}
                else:
                    self.details = summary
            self._update_table()
            if snapshot is not None and self.view == "executions" and page is None:
                self._invalidate()
        self._show_detail()
        self.next_poll = time.monotonic() + min(30.0, self.interval * (2 ** min(self.failures, 4)))
        self._header()

    def _deployments(self) -> list[dict[str, Any]]:
        if "deployments" in self.snapshot:
            return self.snapshot["deployments"]
        return [
            {
                "name": name,
                "type": "unknown",
                "execution_mode": "unknown",
                "state": "Registered",
                "description": "Older server: typed deployment summaries unavailable",
            }
            for name in self.snapshot.get("pipelines", [])
        ]

    def _can_enumerate(self) -> bool:
        return any(
            (self.pipeline is None or item["name"] == self.pipeline) and item.get("execution_enumeration")
            for item in self._deployments()
        )

    def _execution_rows(self) -> list[dict[str, Any]]:
        rows = list(self.page.get("executions", []))
        if self.cursor is not None or self._widget("#revision-filter").value:
            return rows
        for trace in self.trace_page.get("traces", []):
            tags = trace_tags(trace)
            if tags.get("hayhooks.durable.execution_id") or tags.get("hayhooks.openai.execution_mode") == "durable":
                continue
            if self.pipeline is not None and trace["entrypoint"] != self.pipeline:
                continue
            if not any(
                span["name"]
                in {
                    "hayhooks.pipeline.run",
                    "hayhooks.openai.run",
                    "hayhooks.mcp.run_pipeline_tool",
                    "hayhooks.a2a.run_agent",
                }
                for span in all_spans(trace)
            ):
                continue
            state = trace_state(trace)
            if (
                self._widget("#status-filter").value not in ("all", state)
                or self.searches["executions"].lower() not in trace["trace_id"].lower()
            ):
                continue
            rows.append(
                {
                    "_key": json.dumps(["trace", trace["trace_id"]]),
                    "execution_id": trace["trace_id"],
                    "deployment": trace["entrypoint"] or "—",
                    "status": state,
                    "_trace": trace,
                    "updated_at": started_at(trace["start_time_ms"] + trace["duration_ms"]),
                }
            )
        return sorted(rows, key=lambda item: item.get("created_at", item["updated_at"]), reverse=True)

    def _update_table(self, *, recolor: bool = False) -> None:  # noqa: C901, PLR0912, PLR0915
        if not hasattr(self, "main_screen"):
            return
        execution = self.view == "executions"
        traces = self.view == "traces"
        if traces:
            rows, message = self._trace_rows()
        elif execution:
            rows = self._execution_rows()
            message = (
                f"{literal(self.pipeline or 'All deployments')} · {len(rows)} runs · Recent process + durable history"
                if rows
                else "No filter matches on this bounded page; try Next."
                if self.page.get("has_more")
                else "No executions match these filters."
                if self.searches["executions"]
                or self._widget("#status-filter").value != "all"
                or self._widget("#revision-filter").value
                else "No executions yet. Run a pipeline to see it here."
            )
            if not self.page and not self.trace_page:
                message = "Loading execution page…"
            if self.trace_page.get("has_more"):
                message += " Recent sample capped."
            if self.page.get("skipped_controls"):
                message += " Invalid controls skipped; collection may be incomplete."
        else:
            rows = self._deployments()
            type_filter = self._widget("#type-filter").value
            rows = [
                item
                for item in rows
                if self.searches["deployments"].lower() in item["name"].lower()
                and (type_filter == "all" or item["type"] == type_filter)
            ]
            message = f"{len(rows)} deployments  ·  Loaded = registered  ·  Process-scoped health"
            if not rows:
                message = (
                    "No deployments match these filters."
                    if self.searches["deployments"] or type_filter != "all"
                    else "No deployments loaded."
                )
            if not self.snapshot:
                message = "Connecting; no snapshot yet."
        if not self.error:
            self._widget("#message").update(Text(message))
        table = self._widget("#rows")
        key_field = "trace_id" if traces else "_key" if execution else "name"
        wanted = {item[key_field]: item for item in rows}
        was_updating = self.updating
        self.updating = True
        for key in self.items.keys() - wanted.keys():
            table.remove_row(key)
        for key, item in wanted.items():
            values = (
                [
                    item["trace_id"],
                    item["entrypoint"] or "—",
                    trace_kind(item),
                    trace_state(item),
                    "live" if item["root_span"]["running"] else duration(item["duration_ms"]),
                    item["span_count"],
                    "yes"
                    if any(
                        trace_tags(item).get(key) == "true"
                        for key in ("hayhooks.openai.stream_requested", "hayhooks.response.streaming")
                    )
                    else "—",
                ]
                if traces
                else [
                    item["execution_id"],
                    item.get("deployment", "—"),
                    "request-bound" if item.get("_trace") else "durable",
                    item["status"]
                    + (" · cancel pending" if item.get("cancellation_pending") else "")
                    + (" · retry" if item.get("available_at") else ""),
                    ("live" if item["_trace"]["root_span"]["running"] else duration(item["_trace"]["duration_ms"]))
                    if item.get("_trace")
                    else "—",
                    item.get("updated_at", "")[:19],
                ]
                if execution
                else [item["name"], item["type"].title(), item["execution_mode"].title(), item["state"]]
            )
            cells = [Text(literal(value, 512), no_wrap=True, overflow="ellipsis") for value in values]
            state = trace_state(item) if traces else str((item.get("status") if execution else item.get("state")) or "")
            cells[0].stylize("bold")
            cells[1].stylize(self.current_theme.variables["text-muted"])
            cells[3].stylize(state_color(state, self.current_theme))
            if key not in self.items:
                table.add_row(*cells, key=key)
            elif recolor or self.items[key] != item:
                for column, value in zip(table.columns, cells, strict=True):
                    table.update_cell(key, column, value)
        self.items = wanted
        if execution or traces:
            # Cell identity preserves row keys when display text is sanitized or truncated.
            positions = {id(table.get_row(key)[0]): index for index, key in enumerate(wanted)}
            table.sort(key=lambda cells: positions[id(cells[0])])
        selected = self.selected[self.view]
        keep_reading = self._reading_detail() and (
            self.details.get("_key") == selected or self.view == "traces" or isinstance(self.screen, TraceScreen)
        )
        if selected not in wanted and not keep_reading:
            selected = next(iter(wanted), None)
            self.selected[self.view] = selected
        if selected is not None and selected in wanted:
            target = table.get_row_index(selected)
            if table.cursor_row != target:
                table.move_cursor(row=target, animate=False, scroll=False)
        self.updating = was_updating
        self._show_detail()
        self._widget("#previous").disabled = not self.history
        self._widget("#next").disabled = not self.page.get("has_more", False) or not self._can_enumerate()
        self._widget("#page-label").update(Text(f"Page {len(self.history) + 1}"))

    def _trace_rows(self) -> tuple[list[dict[str, Any]], str]:
        traces = self.trace_page.get("traces", [])
        entrypoints = tuple(
            sorted(
                {item["entrypoint"] for item in traces if item["entrypoint"]}
                | {item["name"] for item in self._deployments()}
            )
        )
        if entrypoints != self.trace_entrypoints:
            self.trace_entrypoints = entrypoints
            selector = self._widget("#trace-entrypoint")
            selected = selector.value
            selector.set_options([(literal(name), name) for name in entrypoints])
            selector.value = selected if selected in entrypoints else Select.NULL
        rows = []
        search = self.searches["traces"].lower()
        for item in traces:
            if item["trace_id"] in self.hidden_traces:
                continue
            entrypoint = self._widget("#trace-entrypoint").value
            if entrypoint is not Select.NULL and entrypoint != item["entrypoint"]:
                continue
            if any(
                self._widget(selector).value not in ("all", value)
                for selector, value in (
                    ("#trace-kind", trace_kind(item)),
                    ("#trace-state", trace_state(item)),
                )
            ):
                continue
            if (
                search
                and search
                not in (
                    item["trace_id"]
                    + " "
                    + (item["entrypoint"] or "")
                    + " "
                    + item["root_span"]["name"]
                    + " "
                    + str(trace_tags(item))
                ).lower()
            ):
                continue
            rows.append(item)
        rows.sort(
            key=lambda item: (
                item["duration_ms"] if self._widget("#trace-sort").value == "slowest" else item["start_time_ms"],
                item["trace_id"],
            ),
            reverse=True,
        )
        finished = [item for item in rows if not item["root_span"]["running"]]
        average = duration(sum(item["duration_ms"] for item in finished) / len(finished)) if finished else "—"
        failures = sum(trace_state(item) == "failed" for item in rows)
        cap = " · capped" if self.trace_page.get("has_more") else ""
        message = (
            f"{len(rows)} of {len(traces)} recent traces{cap} · {failures} failed · Avg finished {average}\n"
            "Responding process · Enter Span tree · c Clear locally"
        )
        if not self.trace_page:
            message = "Loading local traces… Monitoring is enabled by default; authenticate for operator access."
        return rows, message

    def _selected_detail(self) -> Text:
        selected = self.selected[self.view]
        item = self.items.get(selected, {}) if selected is not None else {}
        if self.view == "executions" and self.details.get("_key") == selected:
            item = self.details
        if item.get("_trace"):
            item = item["_trace"]
            return trace_detail(
                item, self.current_theme, self.trace_page.get("config", {}).get("slow_component_ms", 1000)
            )
        if self.view == "traces" and item:
            return trace_detail(
                item, self.current_theme, self.trace_page.get("config", {}).get("slow_component_ms", 1000)
            )
        return (
            detail_text(item, execution=self.view == "executions", theme=self.current_theme)
            if item
            else Text("Select an item to read its details.")
        )

    def _show_detail(self) -> None:
        content = self._selected_detail()
        self._widget("#detail").update(content)
        if isinstance(self.screen, TraceScreen):
            selected = self.selected[self.view]
            trace = self.items.get(selected) if selected is not None else None
            if trace and self.view == "executions":
                trace = trace.get("_trace")
            warning = (
                self.error + ". Inspector snapshot stale."
                if self.error
                else ("Trace outside the current buffer/filter. Inspector snapshot retained." if trace is None else "")
            )
            self.screen.query_one("#trace-warning", Static).update(Text(warning))
            self.screen.query_one("#trace-warning").display = bool(warning)
            if trace is not None:
                self.screen.update_trace(trace)
        if isinstance(self.screen, DetailScreen) and not self.screen.help_screen:
            self.screen.query_one("#full-detail-text", Static).update(content)

    @on(DataTable.RowHighlighted)
    def row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.updating:
            return
        if self._reading_detail() and self.selected[self.view] not in self.items:
            return
        key = event.row_key.value
        table = self._widget("#rows")
        if not table.row_count or table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value != key:
            return
        if key not in self.items or self.selected[self.view] == key:
            return
        self.selected[self.view] = key
        if self.view == "deployments":
            item = self.items[key]
            if item["execution_mode"] == "durable" and self.pipeline != key:
                self.pipeline = key
                self._reset_page()
        else:
            self._invalidate()
        self._show_detail()

    @on(DataTable.RowSelected)
    def open_detail(self) -> None:
        if not self.items:
            return
        if self.view == "traces" or self.items.get(self.selected[self.view] or "", {}).get("_trace"):
            selected = self.selected[self.view]
            trace = self.items.get(selected) if selected is not None else None
            if trace and self.view == "executions":
                trace = trace["_trace"]
            if trace:
                self.push_screen(TraceScreen(trace, self.trace_page.get("config", {}).get("slow_component_ms", 1000)))
        elif self.size.width >= WIDE_TERMINAL:
            self._widget("#detail-pane").focus()
        else:
            self.push_screen(DetailScreen(self._selected_detail()))

    @on(Tabs.TabActivated, "#views")
    def switch_view(self, event: Tabs.TabActivated) -> None:
        if event.tab.id == self.view or not hasattr(self, "main_screen"):
            return
        self.view = event.tab.id
        self._configure_view()

    @on(Input.Changed)
    def search_changed(self, event: Input.Changed) -> None:
        if self.updating or not hasattr(self, "main_screen"):
            return
        if event.input.id == "search":
            self.searches[self.view] = event.value
        if self.view == "executions":
            self._reset_page()
        else:
            self._update_table()

    @on(Select.Changed)
    def filter_changed(self, event: Select.Changed) -> None:
        if self.updating or not hasattr(self, "main_screen"):
            return
        if event.select.id == "deployment-filter":
            value = None if event.value is Select.NULL else str(event.value)
            if value == self.pipeline:
                return
            self.pipeline = value
        if self.view == "executions":
            self._reset_page()
        else:
            self._update_table()

    def _reset_page(self) -> None:
        self.cursor = None
        self.history.clear()
        self.page = {}
        self.details = {}
        self.selected["executions"] = None
        self.error = ""
        self._invalidate()
        self._update_table()

    def action_search(self) -> None:
        self.previous_focus = self.focused
        self._widget("#search").focus()

    def action_exit(self) -> None:
        self.exit()

    def action_escape(self) -> None:
        if isinstance(self.screen, (DetailScreen, TraceScreen)):
            self.action_close_detail()
        elif isinstance(self.focused, Input):
            (self.previous_focus or self._widget("#rows")).focus()
        else:
            self._widget("#rows").focus()
            self.call_later(self._update_table)
            self._invalidate()

    def action_close_detail(self) -> None:
        self.pop_screen()
        if isinstance(self.screen, TraceScreen):
            self.screen.query_one("#span-tree").focus()
        elif isinstance(self.screen, DetailScreen):
            self.screen.query_one("#full-detail").focus()
        else:
            self._widget("#rows").focus()
        self.call_later(self._update_table)
        self._invalidate()

    def action_pause(self) -> None:
        self.paused = not self.paused
        self._invalidate()
        self._header()
        self._tick()

    def action_refresh_now(self) -> None:
        self._invalidate()
        self._tick()

    def action_help(self) -> None:
        self.push_screen(DetailScreen(Text(HELP), help_screen=True))

    def action_theme(self) -> None:
        self.theme = "hayhooks-light" if self.theme == "hayhooks-dark" else "hayhooks-dark"
        self._update_table(recolor=True)
        self._header()
        if isinstance(self.screen, TraceScreen):
            self.screen.update_trace(self.screen.trace, force=True)

    def action_view(self, view: str) -> None:
        while self.screen is not self.main_screen:
            self.pop_screen()
        self._widget("#views").active = view
        self._widget("#rows").focus()

    def action_clear_traces(self) -> None:
        if self.view == "traces":
            self.hidden_traces.update(item["trace_id"] for item in self.trace_page.get("traces", []))
            self._update_table()

    def action_executions(self) -> None:
        self.action_view("executions")

    @on(Button.Pressed, "#next")
    def action_next(self) -> None:
        if self.view != "executions" or not self.page.get("has_more") or self.paused:
            return
        self.history.append(self.cursor)
        self.cursor = self.page.get("next_cursor")
        self.page = {}
        self.details = {}
        self.selected["executions"] = None
        self._invalidate()
        self._update_table()

    @on(Button.Pressed, "#previous")
    def action_previous(self) -> None:
        if self.view != "executions" or not self.history or self.paused:
            return
        self.cursor = self.history.pop()
        self.page = {}
        self.details = {}
        self.selected["executions"] = None
        self._invalidate()
        self._update_table()

    def on_resize(self, event: events.Resize) -> None:
        if hasattr(self, "main_screen"):
            self.main_screen.set_class(event.size.height < COMFORTABLE_TERMINAL_HEIGHT, "compact")
            if event.size.width < WIDE_TERMINAL and self.focused is self._widget("#detail-pane"):
                self.push_screen(DetailScreen(self._selected_detail()))
            self._widget("#detail-pane").display = event.size.width >= WIDE_TERMINAL
