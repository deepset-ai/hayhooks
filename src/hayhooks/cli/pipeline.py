import typer
from typing import Optional, Annotated
from pathlib import Path
from hayhooks.cli.utils import make_request
from hayhooks.server.utils.deploy_utils import read_pipeline_files_from_dir
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()

pipeline = typer.Typer()


def _show_error_and_abort(message: str, highlight: str = "") -> None:
    """Display error message in a panel and abort."""
    if highlight:
        message = message.replace(highlight, f"[red]{highlight}[/red]")
    console.print(Panel.fit(message, border_style="red", title="Error"))
    raise typer.Abort()


def _deploy_with_progress(ctx: typer.Context, name: str, endpoint: str, payload: dict) -> None:
    """Handle deployment with progress spinner and response handling."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description=f"Deploying pipeline '{name}'...", total=None)

        response = make_request(
            host=ctx.obj["host"],
            port=ctx.obj["port"],
            endpoint=endpoint,
            method="POST",
            json=payload,
            disable_ssl=ctx.obj["disable_ssl"],
        )

    if response.get("name") == name:
        console.print(
            Panel.fit(
                f"Pipeline '[bold]{name}[/bold]' successfully deployed! 🚀", border_style="green", title="Success"
            )
        )
    else:
        _show_error_and_abort(f"Pipeline '[bold]{name}[/bold]' already exists! ⚠️")


@pipeline.command()
def deploy(
    ctx: typer.Context,
    name: Annotated[Optional[str], typer.Option("--name", "-n", help="The name of the pipeline to deploy.")],
    pipeline_file: Path = typer.Argument(help="The path to the pipeline file to deploy."),
):
    """Deploy a pipeline to the Hayhooks server."""
    if not pipeline_file.exists():
        _show_error_and_abort("Pipeline file does not exist.", str(pipeline_file))

    if name is None:
        name = pipeline_file.stem

    payload = {"name": name, "source_code": pipeline_file.read_text()}
    _deploy_with_progress(ctx=ctx, name=name, endpoint="deploy", payload=payload)


@pipeline.command()
def deploy_files(
    ctx: typer.Context,
    name: Annotated[Optional[str], typer.Option("--name", "-n", help="The name of the pipeline to deploy.")],
    pipeline_dir: Path = typer.Argument(help="The path to the directory containing the pipeline files to deploy."),
    overwrite: Annotated[
        bool, typer.Option("--overwrite", "-o", help="Whether to overwrite the pipeline if it already exists.")
    ] = False,
    skip_saving_files: Annotated[
        bool, typer.Option("--skip-saving-files", "-s", help="Whether to skip saving the files to the server.")
    ] = False,
):
    """Deploy all pipeline files from a directory to the Hayhooks server."""
    if not pipeline_dir.exists():
        _show_error_and_abort("Directory does not exist.", str(pipeline_dir))

    files_dict = read_pipeline_files_from_dir(pipeline_dir)

    if not files_dict:
        console.print(
            Panel.fit(
                "No valid pipeline files found in the specified directory.", border_style="yellow", title="Warning"
            )
        )
        raise typer.Abort()

    if name is None:
        name = pipeline_dir.stem

    payload = {"name": name, "files": files_dict, "save_files": not skip_saving_files, "overwrite": overwrite}
    _deploy_with_progress(ctx=ctx, name=name, endpoint="deploy_files", payload=payload)


@pipeline.command()
def undeploy(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The name of the pipeline to undeploy.")],
):
    """Undeploy a pipeline from the Hayhooks server."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description=f"Undeploying pipeline '{name}'...", total=None)

        response = make_request(
            host=ctx.obj["host"],
            port=ctx.obj["port"],
            endpoint=f"undeploy/{name}",
            method="POST",
            disable_ssl=ctx.obj["disable_ssl"],
        )

    # Check if the response indicates success
    if response and response.get("success"):
        console.print(
            Panel.fit(
                f"Pipeline '[bold]{name}[/bold]' successfully undeployed! 🚀",
                border_style="green",
                title="Success"
            )
        )
    else:
        error_message = response.get("detail", f"Failed to undeploy pipeline '{name}'") if response else f"Pipeline '{name}' not found"
        _show_error_and_abort(error_message)
