"""Lazy terminal dashboard command and noninteractive JSON snapshot."""

import json
import math
from typing import Annotated

import typer

from hayhooks.cli.utils import get_server_url
from hayhooks.tui.client import DashboardClient, DashboardError, literal


def dashboard(  # noqa: C901, PLR0913
    ctx: typer.Context,
    url: Annotated[str | None, typer.Option(help="HTTP server URL, including any proxy path prefix")] = None,
    pipeline: Annotated[str | None, typer.Option(help="Initially selected deployment")] = None,
    traces: Annotated[
        bool, typer.Option("--traces", help="Open Traces, or include local traces in --once JSON")
    ] = False,
    interval: Annotated[float, typer.Option(help="Refresh interval in seconds", min=0.0)] = 2.0,
    once: Annotated[bool, typer.Option("--once", help="Export one snapshot and exit")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Use structured JSON output with --once")] = False,
) -> None:
    """Browse deployments, executions and recent process traces (read only)."""
    if json_output and not once:
        message = "--json requires --once"
        raise typer.BadParameter(message)
    if not math.isfinite(interval) or interval <= 0:
        message = "--interval must be finite and positive"
        raise typer.BadParameter(message)
    server_url = url or get_server_url(ctx.obj["host"], ctx.obj["port"], https=ctx.obj["use_https"])
    client = None
    try:
        client = DashboardClient(server_url)
        if once:
            snapshot = client.status()
            if pipeline is not None:
                snapshot["executions"] = client.executions(pipeline)
            if traces:
                snapshot["tracing"] = client.traces()
            typer.echo(json.dumps(snapshot, ensure_ascii=True))
            if snapshot["status"] == "Degraded":
                typer.echo("Server reports degraded health", err=True)
                raise typer.Exit(1)
            return
        try:
            from hayhooks.tui.app import DashboardApp
        except ModuleNotFoundError as error:
            if error.name != "textual":
                raise
            typer.echo("Install the terminal dashboard with: pip install 'hayhooks[tui]'", err=True)
            raise typer.Exit(1) from error
        DashboardApp(client, pipeline=pipeline, interval=interval, traces=traces).run()
    except DashboardError as error:
        typer.echo(literal(error), err=True)
        raise typer.Exit(1) from error
    finally:
        if client is not None:
            client.close()
