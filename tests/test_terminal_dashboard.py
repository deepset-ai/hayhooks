"""Read-only dashboard behavior, responsive navigation and worker races."""

import copy
import json
import threading
import time
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hayhooks.cli import hayhooks_cli
from hayhooks.tui.client import DashboardClient, DashboardError, literal

textual = pytest.importorskip("textual")
from textual.widgets import DataTable, Input, Static
from hayhooks.tui.app import DashboardApp, DetailScreen


SNAPSHOT = {"status": "Degraded", "pipelines": ["agent", "rag", "custom"], "deployments": [
    {"name": "agent", "type": "agent", "execution_mode": "durable", "state": "Degraded", "revision": "v3",
     "description": "Research with approval", "capabilities": [], "execution_enumeration": True,
     "health": {"healthy": False, "configured_slots": 4, "running_slots": 4, "active_executions": 2,
                "counts": {"nonterminal": 12, "revision_runnable": 5}}},
    {"name": "rag", "type": "pipeline", "execution_mode": "request-bound", "state": "Loaded"},
    {"name": "custom", "type": "unknown", "execution_mode": "request-bound", "state": "Loaded"},
]}
EXECUTIONS = [{"execution_id": f"run_{i:03}", "deployment": "agent", "kind": "agent", "revision": "v2",
               "status": "waiting", "attempt": 1, "created_at": "2026-10-07T08:00:00Z",
               "updated_at": "2026-10-07T08:01:00Z", "cancellation_pending": False, "available_at": None}
              for i in range(65)]


class FakeClient:
    label = "localhost:1416"

    def __init__(self):
        self.calls = []
        self.rows = copy.deepcopy(EXECUTIONS)
        self.failure = False
        self.block = None
        self.long_detail = False
        self.hidden_ids = set()
        self.expired_ids = set()
        self.started = threading.Event()

    def status(self):
        self.calls.append(("GET", "status"))
        self.started.set()
        if self.block:
            self.block.wait(2)
        if self.failure:
            raise DashboardError("Connection unavailable")
        return copy.deepcopy(SNAPSHOT)

    def executions(self, pipeline, **params):
        self.calls.append(("GET", pipeline, params))
        rows = [row for row in self.rows if params.get("search", "") in row["execution_id"] and row["execution_id"] not in self.hidden_ids]
        start = int(params.get("cursor") or 0)
        return {"executions": copy.deepcopy(rows[start:start+50]), "has_more": start+50 < len(rows),
                "next_cursor": str(start+50) if start+50 < len(rows) else None}

    def detail(self, pipeline, execution_id):
        self.calls.append(("GET", pipeline, execution_id))
        if execution_id in self.expired_ids:
            raise DashboardError("Execution expired or no longer accessible (404)")
        return {"execution_id": execution_id, "waiting": {"message": "[bold]literal[/bold]\x1b[2J\x1b]52;c;bad\x07"},
                "progress": [{"message": "long progress line"} for _ in range(50)] if self.long_detail else [], "error": None}


async def settled(app, pilot):
    for _ in range(80):
        await pilot.pause(0.025)
        app._tick()
        if not app.in_flight and app.last_good is not None and (app.view != "executions" or app.page):
            return
    assert False, "refresh did not settle"


@pytest.mark.parametrize("size", [(80, 24), (120, 36), (160, 48)])
async def test_keyboard_browsing_text_inputs_pages_and_resize(size):
    client = FakeClient()
    app = DashboardApp(client, pipeline="agent", interval=100)
    async with app.run_test(size=size) as pilot:
        await settled(app, pilot)
        assert len(app.items) == 3
        assert app.selected["deployments"] == "agent"
        await pilot.press("slash", "q", "r", "space", "question_mark")
        assert app._widget("#search").value == "qr ?"
        assert not app.paused
        app._widget("#search").value = ""
        await pilot.press("escape", "e")
        app.action_refresh_now()
        await settled(app, pilot)
        assert len(app.items) == 50
        if size == (80, 24):
            assert app._widget("#rows").content_size.height >= 8
        assert app.pipeline == "agent"
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.selected["executions"] == "run_001"
        if size[0] < 110:
            assert isinstance(app.screen, DetailScreen)
        else:
            assert app.focused.id == "detail-pane"
        app.action_refresh_now()
        await settled(app, pilot)
        assert app.selected["executions"] == "run_001"
        await pilot.press("escape")
        app.action_next()
        await settled(app, pilot)
        assert len(app.items) == 15
        app.action_previous()
        await settled(app, pilot)
        assert len(app.items) == 50
        await pilot.press("slash")
        app._widget("#search").value = "064"
        app.action_refresh_now()
        await settled(app, pilot)
        assert list(app.items) == ["run_064"]
        assert isinstance(app.focused, Input)
        await pilot.press("escape", "enter")
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.selected["executions"] == "run_064"
        assert {call[0] for call in client.calls} == {"GET"}


async def test_pause_stale_reply_backoff_and_retained_snapshot():
    client = FakeClient()
    app = DashboardApp(client, interval=100)
    async with app.run_test() as pilot:
        await settled(app, pilot)
        before = copy.deepcopy(app.snapshot)
        client.block = threading.Event()
        client.started.clear()
        app.action_refresh_now()
        await pilot.pause(0.05)
        assert client.started.is_set()
        app.action_pause()
        client.block.set()
        await pilot.pause(0.1)
        assert app.snapshot == before and app.paused
        calls = len(client.calls)
        app._tick()
        assert len(client.calls) == calls
        client.block = None
        client.failure = True
        app.action_pause()
        for _ in range(20):
            await pilot.pause(0.02)
            if not app.in_flight:
                break
        assert app.error and app.items and app.snapshot == before
        assert app.next_poll > time.monotonic()
        client.failure = False
        app.action_refresh_now()
        await settled(app, pilot)
        assert not app.error


def test_client_prefix_credentials_safe_errors_and_literal_text(monkeypatch):
    monkeypatch.setenv("HAYHOOKS_DASHBOARD_TOKEN", "private-token")
    client = DashboardClient("https://example.com/hayhooks/")
    response = Mock(status_code=200)
    response.json.return_value = {"result": {"private": "result"}, "execution_id": "run_1"}
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    client.session.get = Mock(return_value=response)
    assert "result" not in client.detail("agent", "run_1")
    args, kwargs = client.session.get.call_args
    assert args[0] == "https://example.com/hayhooks/agent/executions/run_1"
    assert kwargs["verify"] and kwargs["timeout"] == (3, 5)
    response.status_code = 403
    with pytest.raises(DashboardError, match="Authentication") as failure:
        client.status()
    assert "private-token" not in str(failure.value)
    assert literal("[bold]hi[/bold]\x1b[2J\x1b]52;c;secret\x07\u202e!") == "[bold]hi[/bold]!"
    client.close()


def test_once_json_stdout_and_error_exit(monkeypatch):
    monkeypatch.setattr(DashboardClient, "status", lambda _: copy.deepcopy(SNAPSHOT))
    runner = CliRunner()
    result = runner.invoke(hayhooks_cli, ["dashboard", "--once", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "Degraded"
    assert "degraded" in result.stderr


@pytest.mark.parametrize("code,path,detail,message", [
    (404, "status", None, "Status API unavailable"),
    (404, "agent/executions", None, "monitoring disabled"),
    (404, "agent/executions/run_1", None, "expired"),
    (503, "agent/executions", "Durable execution store is unavailable", "Execution store unavailable"),
    (302, "status", None, "HTTP 302"),
])
def test_client_failures_are_distinct_and_never_become_empty_pages(code, path, detail, message):
    client = DashboardClient("https://example.com/prefix")
    response = Mock(status_code=code)
    response.json.return_value = {"detail": detail}
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    client.session.get = Mock(return_value=response)
    with pytest.raises(DashboardError, match=message):
        client.get(path)
    assert client.session.get.call_args.kwargs["allow_redirects"] is False
    client.close()


def test_invalid_health_counts_are_rejected_before_rendering(monkeypatch):
    client = DashboardClient("https://example.com")
    malformed = copy.deepcopy(SNAPSHOT)
    malformed["deployments"][0]["health"]["counts"] = []
    monkeypatch.setattr(client, "get", lambda _: malformed)
    with pytest.raises(DashboardError, match="invalid deployment"):
        client.status()
    client.close()


async def test_obsolete_selection_single_worker_mouse_and_scroll():
    client = FakeClient()
    client.long_detail = True
    app = DashboardApp(client, interval=100)
    async with app.run_test(size=(120, 36)) as pilot:
        await settled(app, pilot)
        await pilot.click("#rows", offset=(5, 2))
        await pilot.pause()
        assert app.selected["deployments"] == "rag"
        client.block = threading.Event()
        client.started.clear()
        app.action_refresh_now()
        await pilot.pause(0.05)
        assert app.in_flight
        before = len(client.calls)
        for _ in range(5):
            app.action_refresh_now()
        assert len(client.calls) == before
        app.action_executions()
        await pilot.pause()
        client.block.set()
        client.block = None
        app.action_refresh_now()
        await settled(app, pilot)
        await pilot.press("down")
        await settled(app, pilot)
        await pilot.press("enter", "end")
        await pilot.pause()
        pane = app._widget("#detail-pane")
        pane.scroll_end(animate=False)
        await pilot.pause()
        position = pane.scroll_y
        assert position > 0
        selected = app.selected["executions"]
        app.action_refresh_now()
        await settled(app, pilot)
        assert pane.scroll_y == position and app.selected["executions"] == selected
        assert app.focused is pane
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert isinstance(app.screen, DetailScreen)
        await pilot.press("escape", "slash", "ctrl+c")
        await pilot.pause()
        assert not app.is_running


@pytest.mark.parametrize("size", [(80, 24), (120, 36), (160, 48)])
async def test_empty_disabled_stale_and_light_layouts(size):
    client = FakeClient()
    app = DashboardApp(client, interval=100)
    async with app.run_test(size=size) as pilot:
        await settled(app, pilot)
        app.action_theme()
        app._widget("#search").value = "no matches"
        await pilot.pause()
        assert not app.items
        assert "No deployments match" in app._widget("#message").content.plain
        app.snapshot["deployments"][0]["execution_enumeration"] = False
        app.action_executions()
        await pilot.pause()
        assert "HAYHOOKS_MONITORING_ENABLED" in app._widget("#message").content.plain
        app.action_help()
        await pilot.pause()
        await pilot.press("escape")
        app.view = "deployments"
        app._widget("#views").active = "deployments"
        await pilot.pause()
        client.failure = True
        app.action_refresh_now()
        for _ in range(20):
            await pilot.pause(0.025)
            if not app.in_flight:
                break
        assert "STALE" in app._widget("#connection").content.plain


@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_details_stay_open_when_execution_leaves_filter_or_expires(size):
    client = FakeClient()
    app = DashboardApp(client, pipeline="agent", interval=100)
    async with app.run_test(size=size) as pilot:
        await settled(app, pilot)
        app.action_executions()
        app.action_refresh_now()
        await settled(app, pilot)
        await pilot.press("enter")
        await pilot.pause()
        selected = app.selected["executions"]
        client.hidden_ids.add(selected)
        app.action_refresh_now()
        await settled(app, pilot)
        assert app.selected["executions"] == selected
        assert "outside this page" in app.details["detail_message"]
        client.expired_ids.add(selected)
        app.action_refresh_now()
        await settled(app, pilot)
        assert app.selected["executions"] == selected
        assert "expired" in app.details["detail_message"]
        await pilot.press("escape")
        await pilot.pause()
        assert app.selected["executions"] != selected


async def test_new_execution_rows_preserve_selection_focus_and_server_order():
    client = FakeClient()
    app = DashboardApp(client, pipeline="agent", interval=100)
    async with app.run_test(size=(120, 36)) as pilot:
        await settled(app, pilot)
        app.action_executions()
        app.action_refresh_now()
        await settled(app, pilot)
        await pilot.press("down", "down", "slash")
        selected = app.selected["executions"]
        first = dict(client.rows[0], execution_id="inserted_before")
        client.rows.insert(0, first)
        app.action_refresh_now()
        await settled(app, pilot)
        table = app._widget("#rows")
        assert app.selected["executions"] == selected
        assert table.ordered_rows[0].key.value == "inserted_before"
        assert isinstance(app.focused, Input)


async def test_poll_and_late_reply_do_not_touch_unmounted_widgets():
    client = FakeClient()
    app = DashboardApp(client, interval=100)
    async with app.run_test() as pilot:
        await settled(app, pilot)
        generation, query = app.generation, app._query()
    calls = len(client.calls)
    app._tick()
    app._apply_refresh(generation, query, SNAPSHOT, None, None, None, "", "")
    assert len(client.calls) == calls and not app.in_flight


async def test_theme_change_recolors_unchanged_rows_without_losing_selection():
    app = DashboardApp(FakeClient(), interval=100)
    async with app.run_test() as pilot:
        await settled(app, pilot)
        table = app._widget("#rows")
        before = table.get_row("agent")[3].spans
        selected = app.selected["deployments"]
        app.action_theme()
        await pilot.pause()
        after = table.get_row("agent")[3].spans
        assert before != after and after[0].style == "#855600"
        assert app.selected["deployments"] == selected


@pytest.mark.parametrize("size", [(80, 24), (120, 36), (160, 48)])
async def test_compact_filters_and_long_execution_ids_stay_reachable(size):
    client = FakeClient()
    full_id = "a1b2c3d4" * 4
    client.rows[0]["execution_id"] = full_id
    app = DashboardApp(client, pipeline="agent", interval=100)
    async with app.run_test(size=size) as pilot:
        await settled(app, pilot)
        selector = app._widget("#type-filter")
        selector.focus()
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        assert selector.value == "agent" and list(app.items) == ["agent"]
        assert selector.region.right <= size[0] and selector.region.y >= 0
        app.action_executions()
        await settled(app, pilot)
        table = app._widget("#rows")
        assert table.columns[next(iter(table.columns))].width == 18
        await pilot.press("enter")
        await pilot.pause()
        assert full_id in app._selected_detail().plain
        assert app.selected["executions"] == full_id
        await pilot.press("escape")
        selector = app._widget("#status-filter")
        selector.focus()
        await pilot.press("enter", "down", "enter")
        await settled(app, pilot)
        assert selector.value == "queued"
        calls = [call for call in client.calls if len(call) == 3 and isinstance(call[2], dict)]
        assert calls[-1][2]["status"] == "queued"
