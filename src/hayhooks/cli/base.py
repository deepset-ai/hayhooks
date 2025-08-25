import sys
from typing import Annotated, Optional

import typer
from fastapi import FastAPI

from hayhooks.cli.mcp import mcp
from hayhooks.cli.pipeline import pipeline
from hayhooks.cli.utils import get_console, get_server_url, make_request, show_success_panel

hayhooks_cli = typer.Typer(name="hayhooks")
hayhooks_cli.add_typer(pipeline, name="pipeline")
hayhooks_cli.add_typer(mcp, name="mcp")


def get_app() -> FastAPI:
    """
    Factory function to create the FastAPI app.
    """
    # Lazy import to avoid importing FastAPI and related dependencies on CLI startup
    from hayhooks.server.app import create_app

    return create_app()


@hayhooks_cli.command()
def run(  # noqa: PLR0913
    host: Annotated[Optional[str], typer.Option("--host", "-h", help="Host to run the server on")] = None,
    port: Annotated[Optional[int], typer.Option("--port", "-p", help="Port to run the server on")] = None,
    pipelines_dir: Annotated[
        Optional[str], typer.Option("--pipelines-dir", "-d", help="Directory containing the pipelines")
    ] = None,
    root_path: Annotated[Optional[str], typer.Option(help="Root path of the server")] = None,
    additional_python_path: Annotated[
        Optional[str], typer.Option(help="Additional Python path to add to sys.path")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of workers to run the server with")] = 1,
    reload: Annotated[
        bool, typer.Option("--reload", "-r", help="Whether to reload the server on file changes")
    ] = False,
) -> None:
    """
    Run the Hayhooks server.
    """
    # Lazy imports to avoid heavy deps on CLI startup
    import uvicorn

    from hayhooks.server.logger import log
    from hayhooks.settings import settings

    # Fill defaults from settings only when command is executed
    host = host or settings.host
    port = port or settings.port
    pipelines_dir = pipelines_dir or settings.pipelines_dir
    root_path = root_path or settings.root_path

    settings.host = host
    settings.port = port
    settings.pipelines_dir = pipelines_dir
    settings.root_path = root_path

    if additional_python_path:
        settings.additional_python_path = additional_python_path
        sys.path.append(additional_python_path)
        log.trace(f"Added {additional_python_path} to sys.path")

    # Use string import path so server modules load only within uvicorn context
    uvicorn.run("hayhooks.server.app:create_app", host=host, port=port, workers=workers, reload=reload, factory=True)


@hayhooks_cli.command()
def status(ctx: typer.Context) -> None:
    """Get the status of the Hayhooks server."""
    response = make_request(
        host=ctx.obj["host"],
        port=ctx.obj["port"],
        endpoint="status",
        use_https=ctx.obj["use_https"],
        disable_ssl=ctx.obj["disable_ssl"],
    )

    server_url = get_server_url(host=ctx.obj["host"], port=ctx.obj["port"], https=ctx.obj["use_https"])

    show_success_panel(
        f"[bold]Hayhooks server is up and running at: {server_url}[/bold]",
        title="",
    )

    if pipes := response.get("pipelines"):
        # Lazy import rich only when needed to render the table
        import rich
        from rich import box

        table = rich.table.Table(
            title="[bold]Deployed Pipelines[/bold]", box=box.ROUNDED, show_header=True, header_style="bold cyan"
        )
        table.add_column("№", style="dim")
        table.add_column("Pipeline Name", style="bright_blue")
        table.add_column("Status", style="green")

        for idx, pipeline in enumerate(pipes, 1):
            table.add_row(str(idx), pipeline, "🟢 Active")

        get_console().print("\n", table)
    else:
        get_console().print("\n[yellow]No pipelines currently deployed[/yellow]")


@hayhooks_cli.callback()
def callback(ctx: typer.Context) -> None:
    # Lazy import settings so it's only loaded on actual CLI invocation
    from hayhooks.settings import settings

    ctx.obj = {
        "host": settings.host,
        "port": settings.port,
        "disable_ssl": settings.disable_ssl,
        "use_https": settings.use_https,
    }
