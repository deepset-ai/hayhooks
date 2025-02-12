import typer
import sys
import uvicorn
import rich
from typing import Annotated, Optional
from hayhooks.cli.pipeline import pipeline
from hayhooks.cli.utils import get_server_url, make_request
from hayhooks.server.app import create_app
from hayhooks.settings import settings
from hayhooks.server.logger import log
from rich.console import Console
from rich.panel import Panel
from rich import box

hayhooks_cli = typer.Typer(name="hayhooks")
hayhooks_cli.add_typer(pipeline, name="pipeline")

console = Console()


def get_app():
    """
    Factory function to create the FastAPI app.
    """
    return create_app()


@hayhooks_cli.command()
def run(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to run the server on")] = settings.host,
    port: Annotated[int, typer.Option("--port", "-p", help="Port to run the server on")] = settings.port,
    pipelines_dir: Annotated[str, typer.Option("--pipelines-dir", "-d", help="Directory containing the pipelines")] = settings.pipelines_dir,
    root_path: Annotated[str, typer.Option(help="Root path of the server")] = settings.root_path,
    additional_python_path: Annotated[Optional[str], typer.Option(help="Additional Python path to add to sys.path")] = settings.additional_python_path,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of workers to run the server with")] = 1,
    reload: Annotated[bool, typer.Option("--reload", "-r", help="Whether to reload the server on file changes")] = False,
):
    """
    Run the Hayhooks server.
    """
    settings.host = host
    settings.port = port
    settings.pipelines_dir = pipelines_dir
    settings.root_path = root_path

    if additional_python_path:
        settings.additional_python_path = additional_python_path
        sys.path.append(additional_python_path)
        log.trace(f"Added {additional_python_path} to sys.path")

    uvicorn.run("hayhooks.cli.base:get_app", host=host, port=port, workers=workers, reload=reload, factory=True)


@hayhooks_cli.command()
def status(ctx: typer.Context):
    """Get the status of the Hayhooks server."""
    response = make_request(
        host=ctx.obj["host"], port=ctx.obj["port"], endpoint="status", disable_ssl=ctx.obj["disable_ssl"]
    )

    console.print(
        Panel.fit(
            f"[green]✓[/green] [bold]Hayhooks server is up and running at: {get_server_url(ctx.obj['host'], ctx.obj['port'])}[/bold]",
            border_style="green",
        )
    )

    if pipes := response.get("pipelines"):
        table = rich.table.Table(
            title="[bold]Deployed Pipelines[/bold]", box=box.ROUNDED, show_header=True, header_style="bold cyan"
        )
        table.add_column("№", style="dim")
        table.add_column("Pipeline Name", style="bright_blue")
        table.add_column("Status", style="green")

        for idx, pipeline in enumerate(pipes, 1):
            table.add_row(str(idx), pipeline, "🟢 Active")

        console.print("\n", table)
    else:
        console.print("\n[yellow]No pipelines currently deployed[/yellow]")


@hayhooks_cli.callback()
def callback(ctx: typer.Context):
    ctx.obj = {
        "disable_ssl": settings.disable_ssl,
        "host": settings.host,
        "port": settings.port,
    }
