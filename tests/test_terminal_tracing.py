import copy
from unittest.mock import Mock

import pytest

from hayhooks.tui.client import DashboardClient, DashboardError

pytest.importorskip("textual")
from textual.widgets import Select, Static, Tree

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
        return_value={
            "config": {"fetch_limit": 100, "buffer_capacity": 200, "slow_component_ms": 1000},
            "traces": [trace],
            "has_more": False,
        }
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


async def test_trace_sorting_keeps_raw_row_identity_after_sanitization():
    client = TraceClient()
    unsafe = "unsafe\x1b[2J-id"
    client.trace_rows = [
        sample_trace(unsafe),
        sample_trace("unsafe-id", start=1700000001000),
        sample_trace("third\x1b[31m", start=1700000002000),
    ]
    app = DashboardApp(client, traces=True, interval=100)
    async with app.run_test(size=(120, 36)) as pilot:
        await trace_settled(app, pilot)
        table = app._widget("#rows")
        assert [row.key.value for row in table.ordered_rows] == list(app.items)
        assert table.get_row(unsafe)[0].plain == table.get_row("unsafe-id")[0].plain == "unsafe-id"
        table.move_cursor(row=table.get_row_index(unsafe))
        await pilot.pause()
        assert app.selected["traces"] == unsafe
        client.trace_rows[0]["duration_ms"] = 4000
        app.action_refresh_now()
        await trace_settled(app, pilot)
        app._widget("#trace-sort").value = "slowest"
        await pilot.press("t")
        await pilot.pause()
        assert [row.key.value for row in table.ordered_rows] == list(app.items)
        assert table.get_row_index(unsafe) == 0 and app.selected["traces"] == unsafe
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, TraceScreen) and app.screen.trace["trace_id"] == unsafe
        assert "\x1b" not in app.screen.query_one("#trace-heading", Static).render().plain


async def test_partial_span_tree_discloses_omitted_spans():
    client = TraceClient()
    trace = sample_trace()
    trace["span_count"] = 66
    client.trace_rows = [trace]
    app = DashboardApp(client, traces=True, interval=100)
    async with app.run_test(size=(120, 36)) as pilot:
        await trace_settled(app, pilot)
        assert "2 of 66 shown" in app._widget("#detail").render().plain
        await pilot.press("enter")
        await pilot.pause()
        assert "2 of 66 spans shown" in app.screen.query_one("#trace-heading", Static).render().plain


def test_trace_client_refreshes_config_in_one_request_per_snapshot():
    client = DashboardClient("https://example.test/proxy")
    snapshots = [
        {
            "traces": [sample_trace()],
            "has_more": False,
            "config": {"fetch_limit": 100, "buffer_capacity": 200, "slow_component_ms": 1000},
        },
        {"traces": [], "has_more": False, "config": {"fetch_limit": 1, "buffer_capacity": 1, "slow_component_ms": 500}},
    ]
    client.get = Mock(side_effect=snapshots)
    try:
        assert client.traces() == snapshots[0]
        assert client.traces() == snapshots[1]
        assert client.get.call_args_list == [(("traces",), {"limit": 100})] * 2
    finally:
        client.close()


@pytest.mark.parametrize(
    "config",
    [
        None,
        [],
        {},
        {"fetch_limit": True, "buffer_capacity": 200, "slow_component_ms": 1000},
        {"fetch_limit": 101, "buffer_capacity": 200, "slow_component_ms": 1000},
        {"fetch_limit": 100, "buffer_capacity": 1, "slow_component_ms": 1000},
    ],
)
def test_trace_client_rejects_invalid_inline_config(config):
    client = DashboardClient("https://example.test")
    client.get = Mock(return_value={"traces": [], "has_more": False, "config": config})
    try:
        with pytest.raises(DashboardError, match="invalid trace configuration"):
            client.traces()
    finally:
        client.close()


@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_executions_combine_recent_and_durable_runs_by_default(size):
    client = TraceClient()
    durable_attempt = sample_trace("durable-attempt")
    durable_attempt["entrypoint"] = "agent"
    durable_attempt["tags"].append({"key": "hayhooks.durable.execution_id", "value": "run_000"})
    deployment = sample_trace("deploy-trace")
    deployment["root_span"]["name"] = "hayhooks.pipeline.deploy"
    client.trace_rows += [durable_attempt, deployment]
    app = DashboardApp(client, interval=100)
    async with app.run_test(size=size) as pilot:
        await settled(app, pilot)
        app.action_executions()
        await settled(app, pilot)
        assert app.pipeline is None
        assert len(app.items) == 53
        assert sum(bool(item.get("_trace")) for item in app.items.values()) == 3
        assert {row["deployment"] for row in app.items.values()} == {"agent", "rag"}
        assert "durable-attempt" not in [row["execution_id"] for row in app.items.values()]
        assert "deploy-trace" not in [row["execution_id"] for row in app.items.values()]
        app._widget("#deployment-filter").value = "rag"
        await settled(app, pilot)
        assert len(app.items) == 3
        assert all(row.get("_trace") for row in app.items.values())
        selected = app.selected["executions"]
        app.open_detail()
        await pilot.pause()
        assert isinstance(app.screen, TraceScreen)
        assert not any(len(call) > 2 and isinstance(call[2], str) and call[2] in {"trace-one", "trace-two", "trace-live"}
                       for call in client.calls)
        client.trace_rows = [row for row in client.trace_rows if row["trace_id"] != app.screen.trace["trace_id"]]
        app.action_refresh_now()
        await settled(app, pilot)
        assert app.selected["executions"] == selected
        assert "retained" in app.screen.query_one("#trace-warning", Static).content.plain
        await pilot.press("escape")
        app._widget("#status-filter").value = "failed"
        await settled(app, pilot)
        assert all(row["status"] == "failed" for row in app.items.values())
        app._widget("#deployment-filter").value = Select.NULL
        app._widget("#status-filter").value = "all"
        await settled(app, pilot)
        assert app.pipeline is None and any(not row.get("_trace") for row in app.items.values())
        app.action_next()
        await settled(app, pilot)
        assert len(app.items) == 15
        assert all(not row.get("_trace") for row in app.items.values())
        app.action_previous()
        await settled(app, pilot)
        assert any(row.get("_trace") for row in app.items.values())


async def test_trace_denial_keeps_owner_history_and_store_failure_keeps_recent_runs(monkeypatch):
    client = TraceClient()
    app = DashboardApp(client, pipeline="agent", interval=100)
    async with app.run_test() as pilot:
        await settled(app, pilot)
        original_traces = client.traces

        def denied(*args, **kwargs):
            raise DashboardError("Authentication failed or monitoring access denied (401/403)")

        monkeypatch.setattr(client, "traces", denied)
        app.action_executions()
        await settled(app, pilot)
        assert len(app.items) == 50 and app.error
        monkeypatch.setattr(client, "traces", original_traces)
        monkeypatch.setattr(client, "executions", denied)
        app._widget("#deployment-filter").value = Select.NULL
        for _ in range(80):
            await pilot.pause(0.025)
            app._tick()
            if not app.in_flight and app.trace_page:
                break
        assert app.error and len(app.items) == 3
        assert all(row.get("_trace") for row in app.items.values())


@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_all_tab_hotkeys_return_from_details_and_preserve_filter_typing(size):
    app = DashboardApp(TraceClient(), interval=100)
    async with app.run_test(size=size) as pilot:
        await settled(app, pilot)
        for key, view in (("v", "traces"), ("e", "executions"), ("d", "deployments")):
            app.action_help()
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
            assert app.screen is app.main_screen
            assert app.view == view
            assert app.focused is app._widget("#rows")
        await pilot.press("slash", "d", "e", "v")
        assert app.view == "deployments"
        assert app._widget("#search").value == "dev"
