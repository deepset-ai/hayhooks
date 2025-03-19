import typer
import json
from typing import Optional, Annotated, Dict, List, Any
from pathlib import Path
from hayhooks.cli.utils import (
    make_request,
    show_error_and_abort,
    show_success_panel,
    show_warning_panel,
    with_progress_spinner,
    console,
    get_server_url,
    upload_files_with_progress,
)
from hayhooks.server.utils.deploy_utils import read_pipeline_files_from_dir


pipeline = typer.Typer()


def _deploy_with_progress(ctx: typer.Context, name: str, endpoint: str, payload: dict) -> None:
    """Handle deployment with progress spinner and response handling."""
    response = with_progress_spinner(
        f"Deploying pipeline '{name}'...",
        make_request,
        host=ctx.obj["host"],
        port=ctx.obj["port"],
        endpoint=endpoint,
        method="POST",
        json=payload,
        disable_ssl=ctx.obj["disable_ssl"],
    )

    if response.get("name") == name:
        show_success_panel(f"Pipeline '[bold]{name}[/bold]' successfully deployed! 🚀")
    else:
        show_error_and_abort(f"Pipeline '[bold]{name}[/bold]' already exists! ⚠️")


@pipeline.command()
def deploy(
    ctx: typer.Context,
    name: Annotated[Optional[str], typer.Option("--name", "-n", help="The name of the pipeline to deploy.")],
    pipeline_file: Path = typer.Argument(help="The path to the pipeline file to deploy."),
):
    """Deploy a pipeline to the Hayhooks server."""
    if not pipeline_file.exists():
        show_error_and_abort("Pipeline file does not exist.", str(pipeline_file))

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
        show_error_and_abort("Directory does not exist.", str(pipeline_dir))

    files_dict = read_pipeline_files_from_dir(pipeline_dir)

    if not files_dict:
        show_warning_panel("No valid pipeline files found in the specified directory.")
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
    response = with_progress_spinner(
        f"Undeploying pipeline '{name}'...",
        make_request,
        host=ctx.obj["host"],
        port=ctx.obj["port"],
        endpoint=f"undeploy/{name}",
        method="POST",
        disable_ssl=ctx.obj["disable_ssl"],
    )

    # Check if the response indicates success
    if response and response.get("success"):
        show_success_panel(f"Pipeline '[bold]{name}[/bold]' successfully undeployed! 🚀")
    else:
        error_message = (
            response.get("detail", f"Failed to undeploy pipeline '{name}'")
            if response
            else f"Pipeline '{name}' not found"
        )
        show_error_and_abort(error_message)


@pipeline.command()
def run(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The name of the pipeline to run.")],
    file: Annotated[
        Optional[List[Path]], typer.Option("--file", "-f", help="Files to upload (can be specified multiple times)")
    ] = None,
    directory: Annotated[
        Optional[List[Path]],
        typer.Option("--dir", "-d", help="Directories to upload (all files within will be uploaded)"),
    ] = None,
    param: Annotated[
        Optional[List[str]],
        typer.Option("--param", "-p", help="Parameters in format key=value (value can be string or JSON)"),
    ] = None,
):
    """Run a pipeline with the given files and parameters."""
    # Initialize collections
    files_to_upload = {}
    params_dict = {}

    # Parse parameters
    if param:
        for p in param:
            if "=" not in p:
                show_error_and_abort(f"Invalid parameter format: {p}. Use key=value")

            key, value = p.split("=", 1)

            # Try to parse as JSON first
            try:
                # First check if it looks like it might be JSON
                if (
                    (value.startswith('{') and value.endswith('}'))
                    or (value.startswith('[') and value.endswith(']'))
                    or value.lower() in ('true', 'false', 'null')
                    or (value.replace('.', '', 1).isdigit())
                ):
                    params_dict[key] = json.loads(value)
                else:
                    params_dict[key] = value
            except json.JSONDecodeError:
                # If JSON parsing fails, use as a string
                params_dict[key] = value

    # Collect individual files
    if file:
        for f in file:
            if not f.exists():
                show_error_and_abort(f"File {f} does not exist")
            if f.is_file():
                files_to_upload[f.name] = f
            else:
                show_error_and_abort(f"{f} is not a file")

    # Collect files from directories
    if directory:
        for dir_path in directory:
            if not dir_path.exists():
                show_error_and_abort(f"Directory {dir_path} does not exist")
            if not dir_path.is_dir():
                show_error_and_abort(f"{dir_path} is not a directory")

            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    # Use relative path as key to preserve directory structure
                    rel_path = file_path.relative_to(dir_path)
                    files_to_upload[str(rel_path)] = file_path

    # Run pipeline
    run_pipeline_with_files(ctx=ctx, pipeline_name=name, files=files_to_upload, params=params_dict)


def run_pipeline_with_files(
    ctx: typer.Context, pipeline_name: str, files: Dict[str, Path], params: Dict[str, Any]
) -> None:
    """Run a pipeline with files and parameters."""
    server_url = get_server_url(ctx.obj["host"], ctx.obj["port"], ctx.obj["disable_ssl"])
    endpoint = f"{server_url}/{pipeline_name}/run"

    # For files or no files, handle differently
    if files:
        # Prepare form data (parameters)
        form_data = {}
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                form_data[key] = json.dumps(value)
            else:
                form_data[key] = str(value)

        # Use the utility function to upload files with progress tracking
        console.print(f"Running pipeline '[bold]{pipeline_name}[/bold]'...")
        result, _ = upload_files_with_progress(
            url=endpoint, files=files, form_data=form_data, verify_ssl=not ctx.obj["disable_ssl"]
        )
    else:
        # No files - use regular JSON request
        response = with_progress_spinner(
            f"Running pipeline '{pipeline_name}'...",
            make_request,
            host=ctx.obj["host"],
            port=ctx.obj["port"],
            endpoint=f"{pipeline_name}/run",
            method="POST",
            json=params,
            disable_ssl=ctx.obj["disable_ssl"],
        )
        result = response

    # Display results
    show_success_panel(f"Pipeline '[bold]{pipeline_name}[/bold]' executed successfully!")

    # Display the result
    if "result" in result:
        console.print("\n[bold cyan]Result:[/bold cyan]")
        if isinstance(result["result"], dict) or isinstance(result["result"], list):
            console.print_json(json.dumps(result["result"]))
        else:
            console.print(result["result"])
    else:
        console.print_json(json.dumps(result))
