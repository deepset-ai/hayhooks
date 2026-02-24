import os
import sys
from typing import Annotated

import typer
from fastapi import FastAPI

from hayhooks.cli.mcp import mcp
from hayhooks.cli.pipeline import pipeline
from hayhooks.cli.utils import get_console, get_server_url, make_request, show_success_panel

hayhooks_cli = typer.Typer(name="hayhooks")
hayhooks_cli.add_typer(pipeline, name="pipeline")
hayhooks_cli.add_typer(mcp, name="mcp")


def _set_env(key: str, value: str | None) -> None:
    """Set an environment variable if the value is truthy, so worker child processes inherit it."""
    if value:
        os.environ[key] = value


def get_app() -> FastAPI:
    """
    Factory function to create the FastAPI app.
    """
    # Lazy import to avoid importing FastAPI and related dependencies on CLI startup
    from hayhooks.server.app import create_app

    return create_app()


@hayhooks_cli.command()
def run(  # noqa: PLR0913
    host: Annotated[str | None, typer.Option("--host", "-h", help="Host to run the server on")] = None,
    port: Annotated[int | None, typer.Option("--port", "-p", help="Port to run the server on")] = None,
    pipelines_dir: Annotated[
        str | None, typer.Option("--pipelines-dir", "-d", help="Directory containing the pipelines")
    ] = None,
    root_path: Annotated[str | None, typer.Option(help="Root path of the server")] = None,
    additional_python_path: Annotated[
        str | None, typer.Option(help="Additional Python path to add to sys.path")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of workers to run the server with")] = 1,
    reload: Annotated[
        bool, typer.Option("--reload", "-r", help="Whether to reload the server on file changes")
    ] = False,
    with_chainlit: Annotated[
        bool, typer.Option("--with-chainlit", help="Enable embedded Chainlit UI (requires hayhooks[chainlit])")
    ] = False,
    chainlit_path: Annotated[
        str | None, typer.Option("--chainlit-path", help="URL path for the Chainlit UI (default: /chat)")
    ] = None,
    chainlit_custom_elements_dir: Annotated[
        str | None,
        typer.Option(
            "--chainlit-custom-elements-dir",
            help="Directory with custom .jsx element files for the Chainlit UI",
        ),
    ] = None,
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

    # Propagate CLI overrides via env vars so that worker child processes
    # (spawned by uvicorn when workers > 1) pick them up when they create
    # their own AppSettings instance.
    _set_env("HAYHOOKS_HOST", host)
    _set_env("HAYHOOKS_PORT", str(port))
    _set_env("HAYHOOKS_PIPELINES_DIR", pipelines_dir)
    _set_env("HAYHOOKS_ROOT_PATH", root_path)

    settings.host = host
    settings.port = port
    settings.pipelines_dir = pipelines_dir
    settings.root_path = root_path

    if additional_python_path:
        _set_env("HAYHOOKS_ADDITIONAL_PYTHON_PATH", additional_python_path)
        settings.additional_python_path = additional_python_path
        sys.path.append(additional_python_path)
        log.trace("Added '{}' to sys.path", additional_python_path)

    # Handling Chainlit CLI flags
    if chainlit_path and not with_chainlit:
        log.warning("--chainlit-path was provided but --with-chainlit is not set. The UI will not be mounted.")

    if chainlit_custom_elements_dir and not with_chainlit:
        log.warning(
            "--chainlit-custom-elements-dir was provided but --with-chainlit is not set. "
            "Custom elements will not be loaded."
        )

    if with_chainlit:
        if workers > 1:
            log.warning(
                "Chainlit UI uses WebSockets (socket.io) which requires sticky sessions. "
                "With --workers {}, requests may hit different worker processes, causing WebSocket failures. "
                "Use --workers 1 when running with --with-chainlit, "
                "or place a reverse proxy with sticky sessions in front.",
                workers,
            )
        _set_env("HAYHOOKS_CHAINLIT_ENABLED", "true")
        settings.chainlit_enabled = True

        if chainlit_path:
            _set_env("HAYHOOKS_CHAINLIT_PATH", chainlit_path)
            settings.chainlit_path = chainlit_path

        if chainlit_custom_elements_dir:
            _set_env("HAYHOOKS_CHAINLIT_CUSTOM_ELEMENTS_DIR", chainlit_custom_elements_dir)
            settings.chainlit_custom_elements_dir = chainlit_custom_elements_dir

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

    assert isinstance(response, dict), "Status endpoint must return JSON"

    if pipes := response.get("pipelines"):
        from rich import box
        from rich.table import Table

        table = Table(
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
