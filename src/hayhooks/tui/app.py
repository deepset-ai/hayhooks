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

from hayhooks.colors import BRAND_COLOR, ERROR_COLOR, SUCCESS_COLOR, WARNING_COLOR
from hayhooks.tui.client import DashboardClient, DashboardError, literal

HELP = """Keyboard and mouse

Arrows       Move rows or scroll focused details
Tab / Shift+Tab   Move focus
Enter        Read selected item details
Esc          Return from details/help or leave search
/            Search deployment names or indexed execution IDs
e            Executions for the selected durable deployment
Space        Pause / resume refresh
r            Refresh now
[ / ]        Previous / next page
?            This help
t            Toggle light / dark theme
q / Ctrl+C   Exit

Click rows to select; use the wheel to scroll.
Search and filters stay in place when returning from details.
Execution pages scan at most 200 candidates; a sparse page may have a next page.
All network requests are GET. Public results are discarded.
"""


WIDE_TERMINAL = 110


def detail_text(item: dict[str, Any], *, execution: bool = False) -> Text:  # noqa: C901
    """Explicit display fields, always literal and bounded; results never appear."""
    fields = (
        (
            ("execution_id", "Execution"),
            ("deployment", "Deployment"),
            ("kind", "Type"),
            ("revision", "Stored revision"),
            ("status", "Status"),
            ("attempt", "Attempt"),
            ("created_at", "Created"),
            ("updated_at", "Updated"),
            ("cancellation_pending", "Cancellation pending"),
            ("available_at", "Retry available at"),
            ("cancellation_requested_at", "Cancellation requested"),
        )
        if execution
        else (
            ("name", "Deployment"),
            ("type", "Type"),
            ("execution_mode", "Execution"),
            ("state", "State"),
            ("description", "Description"),
            ("capabilities", "API capabilities / eligibility"),
            ("revision", "Served revision"),
            ("worker_scope", "Worker scope"),
            ("store_scope", "Store scope"),
        )
    )
    text = Text()
    for field, label in fields:
        if field in item and item[field] is not None:
            value = item[field]
            if isinstance(value, list):
                value = ", ".join(str(part) for part in value)
            text.append(label + ": ", style="bold")
            text.append(literal(value) + "\n")
    text.append("\n")
    if execution:
        for field, label in (("waiting", "Waiting"), ("progress", "Progress"), ("error", "Error")):
            if item.get(field):
                text.append(label + "\n", style="bold")
                text.append(literal(json.dumps(item[field], ensure_ascii=False, indent=2)) + "\n\n")
        if message := item.get("detail_message"):
            text.append(literal(message) + "\n\n")
        text.append(
            "Creation/update timestamps do not measure runtime duration.\nResults are not displayed or retained."
        )
    elif item.get("health") is not None:
        text.append("Responding process\n", style="bold")
        health = item["health"]
        for field in (
            "active_executions",
            "configured_slots",
            "running_slots",
            "draining_slots",
            "draining_runs",
            "maintenance_running",
            "accepting",
            "store_error_streak",
            "operational_error",
        ):
            if field in health:
                text.append(f"{field}: {literal(health[field])}\n")
        text.append("\nStore counts\n", style="bold")
        for field, value in health.get("counts", {}).items():
            text.append(f"{literal(field)}: {literal(value)}\n")
        text.append("\nrevision_runnable includes queued delayed retries.\nWorker totals describe this process.")
    else:
        text.append("Loaded confirms the wrapper is registered. External services are not probed.")
    return text


class DetailScreen(Screen):
    BINDINGS: ClassVar = [Binding("escape", "app.close_detail", "Back")]

    def __init__(self, content: Text, *, help_screen: bool = False) -> None:
        super().__init__()
        self.help_screen = help_screen
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static("Hayhooks  /  " + ("Shortcuts" if self.help_screen else "Details"), classes="detail-heading")
        with VerticalScroll(id="full-detail", can_focus=True):
            yield Static(self.content, id="full-detail-text", markup=False)
        yield Footer()

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
        Binding("space", "pause", "Pause", priority=True),
        Binding("r", "refresh_now", "Refresh"),
        Binding("question_mark", "help", "Help"),
        Binding("e", "executions", "Executions"),
        Binding("left_square_bracket", "previous", "Prev", show=False),
        Binding("right_square_bracket", "next", "Next", show=False),
        Binding("escape", "escape", "Back", show=False),
        Binding("t", "theme", "Theme", show=False),
    ]

    def __init__(self, client: DashboardClient, *, pipeline: str | None = None, interval: float = 2.0) -> None:
        super().__init__()
        for name, dark in (("hayhooks-dark", True), ("hayhooks-light", False)):
            self.register_theme(
                Theme(
                    name=name,
                    primary=BRAND_COLOR,
                    accent=BRAND_COLOR,
                    success=SUCCESS_COLOR if dark else "#007D35",
                    error=ERROR_COLOR if dark else "#C12636",
                    warning=WARNING_COLOR if dark else "#855600",
                    dark=dark,
                )
            )
        self.theme = "hayhooks-dark"
        self.client = client
        self.pipeline = pipeline
        self.interval = interval
        self.view = "deployments"
        self.snapshot: dict[str, Any] = {}
        self.page: dict[str, Any] = {}
        self.items: dict[str, dict[str, Any]] = {}
        self.selected: dict[str, str | None] = {"deployments": pipeline, "executions": None}
        self.details: dict[str, Any] = {}
        self.cursor: str | None = None
        self.history: list[str | None] = []
        self.searches = {"deployments": "", "executions": ""}
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
        yield Static("Hayhooks  /  Connecting…", id="connection", markup=False)
        yield Tabs(Tab("Deployments", id="deployments"), Tab("Executions", id="executions"), id="views")
        with Horizontal(id="filters"):
            yield Input(placeholder="Search deployments", id="search")
            yield Select(
                [("All types", "all"), ("Agent", "agent"), ("Pipeline", "pipeline"), ("Unknown", "unknown")],
                value="all",
                allow_blank=False,
                id="type-filter",
            )
            yield Select(
                [
                    ("All statuses", "all"),
                    *[(s.title(), s) for s in ("queued", "running", "waiting", "completed", "failed", "canceled")],
                ],
                value="all",
                allow_blank=False,
                id="status-filter",
            )
        with Horizontal(id="execution-controls"):
            yield Select([], prompt="Durable deployment", id="deployment-filter")
            yield Input(placeholder="Revision (all retained)", id="revision-filter")
        yield Static("Connecting…", id="message", markup=False)
        with Horizontal(id="panes"):
            yield DataTable(id="rows", cursor_type="row", zebra_stripes=False)
            with VerticalScroll(id="detail-pane", can_focus=True):
                yield Static("Select a deployment", id="detail", markup=False)
        with Horizontal(id="pagination"):
            yield Button("Previous", id="previous", disabled=True)
            yield Static("Page 1", id="page-label", markup=False)
            yield Button("Next", id="next", disabled=True)
        yield Static("Tab Focus   Enter Details   Arrows Navigate", id="navigation-hint", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.main_screen = self.screen
        self._widget("#detail-pane").display = self.size.width >= WIDE_TERMINAL
        self._configure_view()
        self.query_one(DataTable).focus()
        self.set_interval(0.25, self._tick)
        self._tick()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        return not (isinstance(self.focused, Input) and action not in {"escape", "close_detail", "exit"})

    def _widget(self, selector: str):
        return self.main_screen.query_one(selector)

    def _configure_view(self) -> None:
        self.updating = True
        executions = self.view == "executions"
        self._widget("#type-filter").display = not executions
        self._widget("#status-filter").display = executions
        self._widget("#execution-controls").display = executions
        self._widget("#pagination").display = executions
        search = self._widget("#search")
        search.value = self.searches[self.view]
        search.placeholder = "Search indexed execution IDs" if executions else "Search deployment names"
        table = self._widget("#rows")
        table.clear(columns=True)
        columns = (
            ("ID", "Type", "Revision", "Status", "Attempt", "Updated (UTC)")
            if executions
            else ("Name", "Type", "Execution", "State")
        )
        table.add_columns(*columns)
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
            navigation = "Tab Focus   Enter Details   Arrows Navigate"
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
        self._widget("#connection").update(Text(f"Hayhooks  /  {self.client.label}  /  {state}  /  {age}{annotation}"))

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
        return (isinstance(self.screen, DetailScreen) and not self.screen.help_screen) or (
            self.focused is self._widget("#detail-pane")
        )

    def _request_refresh(self) -> None:
        if self.paused or self.in_flight:
            return
        if self.view == "executions" and self.snapshot and not self._can_enumerate():
            self._update_table()
            self.next_poll = time.monotonic() + self.interval
            return
        self.in_flight = True
        self._fetch(self.generation, self._query(), not bool(self.snapshot))

    @work(thread=True, exit_on_error=False)
    def _fetch(self, generation: int, query: tuple, need_status: bool) -> None:
        view, pipeline, search, status, revision, cursor, selected, reading = query
        snapshot = page = detail = None
        error = detail_error = ""
        try:
            if view == "deployments" or need_status:
                snapshot = self.client.status()
            if view == "executions" and pipeline and not need_status:
                page = self.client.executions(
                    pipeline,
                    search=search,
                    status=None if status == "all" else status,
                    revision=revision or None,
                    cursor=cursor,
                )
                ids = [item["execution_id"] for item in page["executions"]]
                selected = selected if selected in ids or (selected and reading) else (ids[0] if ids else None)
                if selected:
                    try:
                        detail = self.client.detail(pipeline, selected)
                    except DashboardError as failure:
                        detail_error = str(failure)
        except DashboardError as failure:
            error = str(failure)
        finally:
            if self.is_running:
                self.call_from_thread(
                    self._apply_refresh, generation, query, snapshot, page, selected, detail, error, detail_error
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
            if snapshot is not None:
                self.snapshot = snapshot
                deployments = self._deployments()
                durable = [
                    (literal(item["name"]), item["name"]) for item in deployments if item["execution_mode"] == "durable"
                ]
                options = self._widget("#deployment-filter")
                self.updating = True
                options.set_options(durable)
                if self.pipeline is None and durable:
                    self.pipeline = durable[0][1]
                if self.pipeline in [value for _, value in durable]:
                    options.value = self.pipeline
                self.updating = False
            if page is not None:
                self.page = page
                self.selected["executions"] = selected
                summary = next((item for item in page["executions"] if item["execution_id"] == selected), {})
                if not summary and self.details.get("execution_id") == selected:
                    summary = self.details
                if detail is not None:
                    self.details = {**summary, **detail}
                    if selected not in [item["execution_id"] for item in page["executions"]]:
                        self.details["detail_message"] = "Selected execution is outside this page/filter."
                elif detail_error:
                    previous = self.details if self.details.get("execution_id") == selected else {}
                    self.details = {**summary, **previous, "detail_message": detail_error + ". Detail snapshot stale."}
                else:
                    self.details = summary
            self._update_table()
            if snapshot is not None and self.view == "executions" and page is None:
                self._invalidate()
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
        return any(item["name"] == self.pipeline and item.get("execution_enumeration") for item in self._deployments())

    def _update_table(self, *, recolor: bool = False) -> None:  # noqa: C901, PLR0912, PLR0915
        if not hasattr(self, "main_screen"):
            return
        execution = self.view == "executions"
        if execution:
            rows = self.page.get("executions", []) if self._can_enumerate() else []
            message = (
                f"{literal(self.pipeline)}  /  Stored kind and revision  /  Indexed collection search"
                if rows
                else "No filter matches on this bounded page; try Next."
                if self.page.get("has_more")
                else "No executions match these filters."
                if self.searches["executions"]
                or self._widget("#status-filter").value != "all"
                or self._widget("#revision-filter").value
                else "No indexed executions yet."
            )
            if not self._can_enumerate():
                message = (
                    "Select a durable deployment. Enable HAYHOOKS_MONITORING_ENABLED=true "
                    "on every writer and configure authenticated access."
                )
            elif not self.page:
                message = "Loading execution page…"
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
            message = "Loaded = registered wrapper. Durable health and workers describe the responding process."
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
        key_field = "execution_id" if execution else "name"
        wanted = {item[key_field]: item for item in rows}
        was_updating = self.updating
        self.updating = True
        for key in self.items.keys() - wanted.keys():
            table.remove_row(key)
        for key, item in wanted.items():
            values = (
                [
                    item["execution_id"],
                    item.get("kind", "unknown"),
                    item.get("revision", ""),
                    item["status"]
                    + (" · cancel pending" if item.get("cancellation_pending") else "")
                    + (" · retry" if item.get("available_at") else ""),
                    item.get("attempt", 0),
                    item.get("updated_at", "")[:19],
                ]
                if execution
                else [item["name"], item["type"].title(), item["execution_mode"].title(), item["state"]]
            )
            cells = [Text(literal(value, 512)) for value in values]
            state = str((item.get("status") if execution else item.get("state")) or "")
            role = {
                "Healthy": "success",
                "completed": "success",
                "Degraded": "warning",
                "waiting": "warning",
                "queued": "warning",
                "failed": "error",
                "running": "primary",
            }.get(state)
            if role:
                cells[3].stylize(getattr(self.current_theme, role) or BRAND_COLOR)
            if key not in self.items:
                table.add_row(*cells, key=key)
            elif recolor or self.items[key] != item:
                for column, value in zip(table.columns, cells, strict=True):
                    table.update_cell(key, column, value)
        self.items = wanted
        if execution:
            positions = {key: index for index, key in enumerate(wanted)}
            table.sort(key=lambda cells: positions[cells[0].plain])
        selected = self.selected[self.view]
        keep_reading = self._reading_detail() and self.details.get("execution_id") == selected
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

    def _selected_detail(self) -> Text:
        selected = self.selected[self.view]
        item = self.items.get(selected, {}) if selected is not None else {}
        if self.view == "executions" and self.details.get("execution_id") == selected:
            item = self.details
        return (
            detail_text(item, execution=self.view == "executions")
            if item
            else Text("Select an item to read its details.")
        )

    def _show_detail(self) -> None:
        content = self._selected_detail()
        self._widget("#detail").update(content)
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
        if self.size.width >= WIDE_TERMINAL:
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
            if event.value in (Select.NULL, self.pipeline):
                return
            self.pipeline = str(event.value)
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
        if isinstance(self.screen, DetailScreen):
            self.action_close_detail()
        elif isinstance(self.focused, Input):
            (self.previous_focus or self._widget("#rows")).focus()
        else:
            self._widget("#rows").focus()
            self.call_later(self._update_table)
            self._invalidate()

    def action_close_detail(self) -> None:
        self.pop_screen()
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

    def action_executions(self) -> None:
        self._widget("#views").active = "executions"
        self._widget("#rows").focus()

    @on(Button.Pressed, "#next")
    def action_next(self) -> None:
        if not self.page.get("has_more") or self.paused:
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
        if not self.history or self.paused:
            return
        self.cursor = self.history.pop()
        self.page = {}
        self.details = {}
        self.selected["executions"] = None
        self._invalidate()
        self._update_table()

    def on_resize(self, event: events.Resize) -> None:
        if hasattr(self, "main_screen"):
            if event.size.width < WIDE_TERMINAL and self.focused is self._widget("#detail-pane"):
                self.push_screen(DetailScreen(self._selected_detail()))
            self._widget("#detail-pane").display = event.size.width >= WIDE_TERMINAL
