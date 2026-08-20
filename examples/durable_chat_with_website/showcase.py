"""Run two concurrent resumable streams in a three-pane terminal dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_PIPELINE = "chat_with_website"
_REATTACH_DELAY_SECONDS = 3.0
_MIN_TERMINAL_WIDTH = 100
_MIN_TERMINAL_HEIGHT = 28


@dataclass(frozen=True)
class Lane:
    name: str
    color: str
    disconnect_after: float
    question: str


@dataclass
class LaneState:
    lane: Lane
    phase: str = "READY"
    detail: str = "waiting for launch"
    answer: str = ""
    cursor: str | None = None
    connected_at: float | None = None
    attempt: int | None = None
    terminal: dict[str, Any] | None = None
    done: bool = False
    error: str | None = None


@dataclass(frozen=True)
class Activity:
    timestamp: float
    lane: str
    method: str
    status: str
    target: str
    note: str


_LANES = (
    Lane(
        name="ATLAS",
        color="bright_cyan",
        disconnect_after=12.0,
        question=(
            "Write a lively 1,800 to 2,200 word field guide titled THE ATLAS TRANSMISSIONS. "
            "Use exactly 14 numbered dispatches to explain what Haystack is, what people build with it, "
            "how its pipelines and components fit together, and where Redis is useful. Begin with the exact "
            "marker ATLAS-LIVE. Base factual claims on the supplied websites, use vivid but accurate analogies, "
            "and keep going until all 14 dispatches and a short field checklist are complete."
        ),
    ),
    Lane(
        name="COMET",
        color="bright_magenta",
        disconnect_after=16.0,
        question=(
            "Write a lively 1,800 to 2,200 word incident chronicle titled THE COMET LOG. Use exactly 12 "
            "timestamped scenes to explore what Haystack can build, how pipelines organize work, what Redis "
            "provides, and how those ideas can support reliable AI systems. Begin with the exact marker "
            "COMET-LIVE. Base factual claims on the supplied websites, label imaginative examples clearly, "
            "and finish with a six-item launch checklist."
        ),
    ),
)


class StreamEventError(RuntimeError):
    """The SSE endpoint reported an error event."""


def _path(url: str) -> str:
    parsed = urlparse(url)
    parts = (parsed.path or "/").split("/")
    if len(parts) > 3 and parts[2] == "executions":  # noqa: PLR2004 - path segment indexes are the format
        parts[3] = f"{parts[3][:8]}…"
    return "/".join(parts)


def _activity(  # noqa: PLR0913 - one compact call records one HTTP timeline row
    events: list[Activity], started: float, lane: str, method: str, status: str, target: str, note: str = ""
) -> None:
    events.append(Activity(time.monotonic() - started, lane, method, status, target, note))


def _apply_sse_event(state: LaneState, fields: dict[str, str]) -> dict[str, Any] | None:
    event = fields.get("event", "message")
    state.cursor = fields.get("id", state.cursor)
    data = json.loads(fields.get("data", "{}"))
    if event == "chunk":
        state.attempt = data["attempt"]
        state.answer += data["payload"].get("content") or ""
        return None
    if event == "gap":
        state.detail = data["detail"]
        return None
    if event == "error":
        raise StreamEventError(data.get("detail", "execution stream interrupted"))
    return data


async def _read_sse(response: httpx.Response, state: LaneState) -> dict[str, Any] | None:
    fields: dict[str, str] = {}
    async for line in response.aiter_lines():
        if line.startswith(":"):
            continue
        if line:
            name, separator, value = line.partition(":")
            if separator:
                fields[name] = value.lstrip()
            continue
        if fields:
            terminal = _apply_sse_event(state, fields)
            fields = {}
            if terminal is not None:
                return terminal
    return None


async def _open_stream(  # noqa: PLR0913 - stream context is deliberately explicit at the call site
    client: httpx.AsyncClient,
    state: LaneState,
    stream_url: str,
    events: list[Activity],
    started: float,
    *,
    disconnect_after: float | None,
) -> tuple[dict[str, Any] | None, bool]:
    headers = {"Last-Event-ID": state.cursor} if state.cursor else {}
    note = f"Last-Event-ID: {state.cursor}" if state.cursor else "new event stream"
    _activity(events, started, state.lane.name, "GET", "→", _path(stream_url), note)
    async with client.stream("GET", stream_url, headers=headers) as response:
        _activity(
            events, started, state.lane.name, "GET", str(response.status_code), _path(stream_url), "SSE connected"
        )
        response.raise_for_status()
        state.connected_at = time.monotonic()
        try:
            if disconnect_after is None:
                return await _read_sse(response, state), False
            return await asyncio.wait_for(_read_sse(response, state), timeout=disconnect_after), False
        except asyncio.TimeoutError:
            return None, True


async def _run_lane(  # noqa: PLR0915 - the full visible lifecycle reads linearly in one coroutine
    client: httpx.AsyncClient, state: LaneState, base_url: str, events: list[Activity], started: float
) -> None:
    lane = state.lane
    submit_url = f"{base_url}/{_PIPELINE}/run-durable"
    state.phase = "SUBMITTING"
    state.detail = "creating durable execution"
    _activity(events, started, lane.name, "POST", "→", _path(submit_url), "submit durable execution")
    try:
        submitted = await client.post(
            submit_url,
            headers={"Idempotency-Key": f"show-{lane.name.lower()}-{uuid.uuid4().hex[:10]}"},
            json={"question": lane.question},
        )
        body = submitted.json()
        execution_id = body.get("execution_id", "unknown")
        _activity(
            events,
            started,
            lane.name,
            "POST",
            str(submitted.status_code),
            _path(submit_url),
            f"accepted {execution_id[:12]}",
        )
        submitted.raise_for_status()
        stream_url = urljoin(f"{base_url}/", body["links"]["stream"])
        inspect_url = urljoin(f"{base_url}/", body["links"]["self"])

        state.phase = "LIVE"
        state.detail = f"client link will cut at {lane.disconnect_after:.0f}s"
        terminal, interrupted = await _open_stream(
            client,
            state,
            stream_url,
            events,
            started,
            disconnect_after=lane.disconnect_after,
        )
        if not interrupted:
            msg = f"answer completed before the planned {lane.disconnect_after:.0f}s network cut"
            raise RuntimeError(msg)

        state.phase = "LINK CUT"
        state.detail = f"socket closed on purpose; cursor {state.cursor or 'start'} saved"
        _activity(
            events,
            started,
            lane.name,
            "CLOSE",
            "✂",
            _path(stream_url),
            f"client cut at {lane.disconnect_after:.0f}s; cursor {state.cursor or 'start'}",
        )
        await asyncio.sleep(_REATTACH_DELAY_SECONDS / 2)
        _activity(events, started, lane.name, "GET", "→", _path(inspect_url), "check execution without a viewer")
        inspected = await client.get(inspect_url)
        inspected.raise_for_status()
        control = inspected.json()
        _activity(
            events,
            started,
            lane.name,
            "GET",
            str(inspected.status_code),
            _path(inspect_url),
            f"status={control['status']}; attempt={control['attempt']}",
        )
        state.detail = f"control plane: {control['status']}; reattaching from {state.cursor or 'start'}"
        await asyncio.sleep(_REATTACH_DELAY_SECONDS / 2)

        state.phase = "RESUMED"
        state.detail = f"Last-Event-ID: {state.cursor or 'start'}"
        terminal, _ = await _open_stream(
            client,
            state,
            stream_url,
            events,
            started,
            disconnect_after=None,
        )
        if terminal is None:
            msg = "stream ended without a terminal event"
            raise RuntimeError(msg)
        state.terminal = terminal
        exact = terminal.get("result", {}).get("reply") == state.answer
        if terminal.get("status") != "completed" or not exact:
            msg = "terminal result did not exactly match the replayed stream"
            raise RuntimeError(msg)

        state.phase = "EXACT REPLAY"
        state.detail = f"{len(state.answer):,} characters; no gaps; no duplicates"
        state.done = True
        _activity(events, started, lane.name, "SSE", "✓", _path(stream_url), state.detail)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, RuntimeError, StreamEventError) as error:
        state.phase = "FAILED"
        state.detail = str(error)
        state.error = str(error)
        state.done = True
        _activity(events, started, lane.name, "SSE", "!", _path(submit_url), str(error))


def _tail_text(value: str, console: Console, *, width: int, height: int) -> Text:
    if not value:
        return Text("Waiting for the first streamed token…")
    lines = console.render_lines(Text(value), console.options.update(width=width, height=None), pad=False)
    plain_lines = ["".join(segment.text for segment in line).rstrip() for line in lines]
    if len(plain_lines) <= height:
        return Text("\n".join(plain_lines))
    visible_lines = plain_lines[-height:]
    if height == 1:
        return Text.assemble(("… ", "dim"), visible_lines[0])
    truncated = "\n".join(visible_lines[1:])
    return Text.assemble(("… replay transcript continues …\n", "dim"), visible_lines[0], "\n", truncated)


def _lane_panel(state: LaneState, console: Console, *, width: int, height: int) -> Panel:
    color = "red" if state.error else state.lane.color
    elapsed = time.monotonic() - state.connected_at if state.connected_at and state.phase == "LIVE" else None
    progress = ""
    if elapsed is not None:
        width = max(10, min(24, console.width // 6))
        filled = min(width, int(width * elapsed / state.lane.disconnect_after))
        progress = f"  {'━' * filled}{'·' * (width - filled)} {elapsed:4.1f}/{state.lane.disconnect_after:.0f}s"

    status = Text()
    status.append(state.phase, style=f"bold {color}")
    status.append(progress, style=color)
    status.append(f"\n{state.detail}", style="dim")
    if state.attempt is not None:
        status.append(f"  attempt {state.attempt}", style="dim")

    content_width = max(1, width - 4)  # panel border and horizontal padding
    header = Group(status, Text("─" * min(28, max(10, console.width // 4)), style=color))
    header_height = len(
        console.render_lines(header, console.options.update(width=content_width, height=None), pad=False)
    )
    answer = _tail_text(state.answer, console, width=content_width, height=max(1, height - header_height - 2))
    return Panel(
        Group(header, answer),
        title=f" {state.lane.name} · CUT AT {state.lane.disconnect_after:.0f}s ",
        border_style=color,
        padding=(0, 1),
    )


def _activity_panel(events: list[Activity], console: Console) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=7, style="dim", no_wrap=True)
    table.add_column(width=8, no_wrap=True)
    table.add_column(width=7, no_wrap=True)
    table.add_column(width=7, no_wrap=True)
    table.add_column(ratio=1, overflow="ellipsis")
    rows = max(4, int(console.height * 0.3) - 4)
    for event in events[-rows:]:
        lane_color = "bright_cyan" if event.lane == "ATLAS" else "bright_magenta"
        status_color = "green" if event.status.startswith("2") or event.status == "✓" else "red"
        if event.status == "→":
            status_color = "bright_yellow"
        table.add_row(
            f"{event.timestamp:5.1f}s",
            Text(event.lane, style=f"bold {lane_color}"),
            Text(event.method, style="bold"),
            Text(event.status, style=f"bold {status_color}"),
            f"{event.target}  [dim]{event.note}[/]",
        )
    if not events:
        table.add_row("0.0s", "SHOW", "READY", "·", "waiting for both POST requests")
    return Panel(table, title=" HTTP FLIGHT RECORDER · POST → CUT → INSPECT → RESUME ", border_style="bright_yellow")


def _dashboard(states: list[LaneState], events: list[Activity], console: Console) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="streams", ratio=7, minimum_size=12), Layout(name="activity", ratio=3, minimum_size=8)
    )
    layout["streams"].split_row(Layout(name="atlas"), Layout(name="comet"))
    regions = layout.render(console, console.options)
    for state, name in zip(states, ("atlas", "comet"), strict=True):
        region = regions[layout[name]].region
        layout[name].update(_lane_panel(state, console, width=region.width, height=region.height))
    layout["activity"].update(_activity_panel(events, console))
    return layout


def _preflight(base_url: str, console: Console) -> bool:
    try:
        response = httpx.get(f"{base_url}/status", timeout=3)
        response.raise_for_status()
        status = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        console.print(
            Panel(f"Hayhooks is not ready at {base_url}: {error}", title=" SERVER MISSING ", border_style="red")
        )
        return False
    deployment = status.get("durable", {}).get("deployments", {}).get(_PIPELINE)
    if not deployment or not deployment.get("healthy"):
        console.print(
            Panel(f"The durable {_PIPELINE} pipeline is not healthy.", title=" PREFLIGHT FAILED ", border_style="red")
        )
        return False
    if deployment.get("configured_slots", 0) < len(_LANES):
        console.print(
            Panel(
                "The server's durable execution concurrency ceiling is below two, "
                "so this two-lane demo would run serially.\n\n"
                "Set [bold]HAYHOOKS_DURABLE_EXECUTION_CONCURRENCY=2[/] or higher and restart Hayhooks.",
                title=" CONCURRENCY CEILING TOO LOW ",
                border_style="bright_yellow",
            )
        )
        return False
    if console.width < _MIN_TERMINAL_WIDTH or console.height < _MIN_TERMINAL_HEIGHT:
        console.print("[yellow]Tip: enlarge this terminal to at least 100x28 for the best show.[/]")
    return True


async def _show(base_url: str, console: Console) -> int:
    states = [LaneState(lane) for lane in _LANES]
    events: list[Activity] = []
    started = time.monotonic()
    timeout = httpx.Timeout(10, read=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [asyncio.create_task(_run_lane(client, state, base_url, events, started)) for state in states]
        with Live(_dashboard(states, events, console), console=console, refresh_per_second=12) as live:
            while not all(task.done() for task in tasks):
                live.update(_dashboard(states, events, console))
                await asyncio.sleep(0.08)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for state, result in zip(states, results, strict=True):
                if isinstance(result, BaseException):
                    state.phase = "FAILED"
                    state.detail = str(result)
                    state.error = str(result)
                    state.done = True
                    _activity(events, started, state.lane.name, "SSE", "!", "/", str(result))
            live.update(_dashboard(states, events, console), refresh=True)
    console.print(
        "[bold green]Recovery race complete.[/] Both client connections were cut and both answers replayed exactly."
        if not any(state.error for state in states)
        else "[bold red]Recovery race failed.[/] Read the pane marked FAILED above."
    )
    return 1 if any(state.error for state in states) else 0


def _self_test() -> int:
    state = LaneState(_LANES[0])
    assert (
        _apply_sse_event(
            state,
            {"id": "1-0", "event": "chunk", "data": '{"attempt":1,"payload":{"content":"hello"}}'},
        )
        is None
    )
    assert state.answer == "hello"
    assert state.cursor == "1-0"
    terminal = _apply_sse_event(state, {"event": "completed", "data": '{"status":"completed"}'})
    assert terminal == {"status": "completed"}
    assert _path("http://localhost:1416/a?b=1") == "/a"
    assert _path("http://localhost/chat_with_website/executions/123456789/stream").endswith("/12345678…/stream")
    tail = _tail_text("zero one two three four", Console(width=8), width=8, height=3)
    assert tail.plain.endswith("three\nfour") and "zero" not in tail.plain
    test_console = Console(width=100, height=28)
    states = [LaneState(lane, answer=("old text " * 500) + f"LATEST-{lane.name}") for lane in _LANES]
    lines = test_console.render_lines(_dashboard(states, [], test_console), test_console.options)
    rendered = "\n".join("".join(segment.text for segment in line) for line in lines)
    assert all(f"LATEST-{lane.name}" in rendered for lane in _LANES)
    Console().print("showcase self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:1416")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    base_url = args.base_url.rstrip("/")
    console = Console()
    if not _preflight(base_url, console):
        return 1
    return asyncio.run(_show(base_url, console))


if __name__ == "__main__":
    raise SystemExit(main())
