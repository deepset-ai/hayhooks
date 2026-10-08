import copy
from unittest.mock import Mock

import pytest

from hayhooks.tui.client import DashboardClient, DashboardError

pytest.importorskip("textual")
from textual.widgets import Static, Tree
from hayhooks.tui.app import DashboardApp
from hayhooks.tui.traces import TraceScreen, trace_state
from tests.test_terminal_dashboard import FakeClient, settled


def sample_trace(trace_id="trace-one", *, state="completed", start=1700000000000, duration=2000):
    tags = [{"key": "hayhooks.success", "value": "false" if state == "failed" else "true"}]
    if state == "failed":
        tags += [
            {"key": "hayhooks.error.type", "value": "ValueError"},
            {"key": "hayhooks.error.message", "value": "[bold]literal[/bold]\x1b[2J"},
        ]
    child = {
        "span_id": "generator",
        "name": "haystack.component.run",
        "start_time_ms": start + 100,
        "duration_ms": 1500,
        "running": False,
        "tags": [{"key": "haystack.component.name", "value": "Answer generator"}],
        "children": [],
    }
    return {
        "trace_id": trace_id,
        "entrypoint": "rag",
        "start_time_ms": start,
        "duration_ms": duration,
        "span_count": 2,
        "tags": tags,
        "root_span": {
            "span_id": "root",
            "name": "hayhooks.pipeline.run",
            "start_time_ms": start,
            "duration_ms": duration,
            "running": state == "running",
            "tags": tags,
            "children": [child],
        },
    }


class TraceClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.trace_rows = [
            sample_trace(),
            sample_trace("trace-two", state="failed", start=1700000001000, duration=3000),
            sample_trace("trace-live", state="running", start=1700000002000, duration=10),
        ]

    def traces(self):
        self.calls.append(("GET", "traces"))
        if self.failure:
            raise DashboardError("Connection unavailable")
        return {
            "traces": copy.deepcopy(self.trace_rows),
            "has_more": False,
            "config": {"fetch_limit": 100, "buffer_capacity": 200, "slow_component_ms": 1000},
        }


async def trace_settled(app, pilot):
    for _ in range(100):
        await pilot.pause(0.025)
        app._tick()
        if not app.in_flight and app.trace_page:
            return
    assert False, "trace refresh did not settle"


@pytest.mark.parametrize("size", [(80, 24), (120, 36), (160, 48)])
async def test_trace_browsing_filters_spans_updates_and_failure(size):
    client = TraceClient()
    app = DashboardApp(client, interval=100)
    async with app.run_test(size=size) as pilot:
        await settled(app, pilot)
        await pilot.press("v")
        await trace_settled(app, pilot)
        assert list(app.items) == ["trace-live", "trace-two", "trace-one"]
        app._widget("#trace-sort").value = "slowest"
        await pilot.pause()
        assert list(app.items)[0] == "trace-two"
        assert "Avg finished 2.50s" in app._widget("#message").render().plain
        app._widget("#trace-state").value = "failed"
        await pilot.pause()
        assert list(app.items) == ["trace-two"]
        app._widget("#rows").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, TraceScreen)
        screen = app.screen
        assert "[bold]literal[/bold]" in screen.query_one("#span-detail", Static).render().plain
        assert "\x1b" not in screen.query_one("#span-detail", Static).render().plain
        tree = screen.query_one(Tree)
        await pilot.press("down")
        await pilot.pause()
        assert tree.cursor_node.data["span_id"] == "generator"
        assert not tree.cursor_node.allow_expand
        assert "Answer generator" in screen.query_one("#span-detail", Static).render().plain
        screen.nodes["root"].collapse()
        client.trace_rows[1]["root_span"]["children"][0]["duration_ms"] = 1800
        app.action_refresh_now()
        await trace_settled(app, pilot)
        assert not screen.nodes["root"].is_expanded
        assert screen.nodes["generator"].data["duration_ms"] == 1800
        client.failure = True
        app.action_refresh_now()
        for _ in range(50):
            await pilot.pause(0.025)
            if app.error:
                break
        assert "stale" in screen.query_one("#trace-warning", Static).render().plain
        await pilot.press("escape")
        assert app.view == "traces" and app._widget("#trace-state").value == "failed"
        assert all(call[0] == "GET" for call in client.calls)


async def test_trace_clear_is_local_pause_and_snapshot_eviction():
    client = TraceClient()
    app = DashboardApp(client, interval=100)
    async with app.run_test(size=(80, 24)) as pilot:
        await settled(app, pilot)
        await pilot.press("v")
        await trace_settled(app, pilot)
        before = len(client.calls)
        await pilot.press("space", "c")
        await pilot.pause()
        assert app.paused and not app.items and len(client.calls) == before
        client.trace_rows = [sample_trace("new-process")]
        await pilot.press("space")
        await trace_settled(app, pilot)
        assert list(app.items) == ["new-process"] and not app.hidden_traces
        assert app._widget("#rows").content_size.height >= 7


def test_finished_child_does_not_make_a_live_root_completed():
    assert trace_state(sample_trace(state="running")) == "running"


@pytest.mark.parametrize("corruption", ["duration", "children", "depth", "tags", "duplicate"])
def test_trace_client_rejects_invalid_snapshots(corruption):
    client = DashboardClient("https://example.test/proxy")
    trace = sample_trace()
    if corruption == "duration":
        trace["duration_ms"] = "oops"
    elif corruption == "children":
        trace["root_span"]["children"] = "oops"
    elif corruption == "depth":
        node = trace["root_span"]
        for i in range(66):
            child = {**copy.deepcopy(node), "span_id": str(i), "children": []}
            node["children"] = [child]
            node = child
        trace["span_count"] = 67
    elif corruption == "tags":
        trace["tags"] = [{"key": "error", "value": {"bad": 1}}]
    else:
        trace["root_span"]["children"][0]["span_id"] = "root"
    client.get = Mock(
        side_effect=[
            {"fetch_limit": 100, "buffer_capacity": 200, "slow_component_ms": 1000},
            {"traces": [trace], "has_more": False},
        ]
    )
    with pytest.raises(DashboardError, match="invalid trace metadata"):
        client.traces()
    client.close()


async def test_start_in_traces_and_keep_root_selection_after_theme_change():
    app = DashboardApp(TraceClient(), interval=100, traces=True)
    async with app.run_test(size=(120, 36)) as pilot:
        await trace_settled(app, pilot)
        assert app.view == "traces" and app._widget("#views").active == "traces"
        await pilot.press("enter", "t")
        await pilot.pause()
        assert isinstance(app.screen, TraceScreen)
        assert app.screen.query_one(Tree).cursor_node.data["span_id"] == "root"
        assert app.theme == "hayhooks-light"


def test_trace_once_json_uses_no_textual_runtime(monkeypatch):
    import json
    from typer.testing import CliRunner
    from hayhooks.cli import hayhooks_cli

    monkeypatch.setattr(DashboardClient, "status", lambda _: {"status": "Up!", "pipelines": []})
    monkeypatch.setattr(DashboardClient, "traces", lambda _: {"traces": [sample_trace()], "has_more": False})
    result = CliRunner().invoke(hayhooks_cli, ["dashboard", "--traces", "--once", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["tracing"]["traces"][0]["trace_id"] == "trace-one"


async def test_entrypoint_named_all_is_distinct_from_no_filter():
    client = TraceClient()
    client.trace_rows[0]["entrypoint"] = "all"
    app = DashboardApp(client, traces=True, interval=100)
    async with app.run_test(size=(80, 24)) as pilot:
        await trace_settled(app, pilot)
        app._widget("#trace-entrypoint").focus()
        await pilot.press("enter", "down", "down", "enter")
        await pilot.pause()
        assert app._widget("#trace-entrypoint").value == "all"
        assert list(app.items) == ["trace-one"]


async def test_help_preserves_trace_inspection_and_view_shortcuts_return_to_list():
    client = TraceClient()
    app = DashboardApp(client, traces=True, interval=100)
    async with app.run_test(size=(120, 36)) as pilot:
        await trace_settled(app, pilot)
        await pilot.press("enter", "down", "?")
        await pilot.pause()
        selected = app.selected["traces"]
        client.trace_rows = []
        app.action_refresh_now()
        await trace_settled(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, TraceScreen)
        assert app.focused is app.screen.query_one(Tree)
        assert app.selected["traces"] == selected
        assert "retained" in app.screen.query_one("#trace-warning", Static).render().plain
        await pilot.press("e")
        await pilot.pause()
        assert app.screen is app.main_screen and app.view == "executions"
