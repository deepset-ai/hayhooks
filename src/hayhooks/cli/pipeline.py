import json
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from hayhooks.cli.utils import (
    get_console,
    get_server_url,
    make_request,
    show_error_and_abort,
    show_success_panel,
    show_warning_panel,
    upload_files_with_progress,
    with_progress_spinner,
)

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
        use_https=ctx.obj["use_https"],
        disable_ssl=ctx.obj["disable_ssl"],
    )

    if response.get("name") == name:
        show_success_panel(f"Pipeline '[bold]{name}[/bold]' successfully deployed! 🚀")
    else:
        show_error_and_abort(f"Pipeline '[bold]{name}[/bold]' already exists! ⚠️")


@pipeline.command(name="deploy-yaml")
def deploy_yaml(  # noqa: PLR0913
    ctx: typer.Context,
    pipeline_file: Path = typer.Argument(  # noqa: B008
        help="The path to the YAML pipeline file to deploy."
    ),
    name: Annotated[Optional[str], typer.Option("--name", "-n", help="The name of the pipeline to deploy.")] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", "-o", help="Whether to overwrite the pipeline if it already exists.")
    ] = False,
    description: Annotated[
        Optional[str], typer.Option("--description", help="Optional description for the pipeline.")
    ] = None,
    skip_mcp: Annotated[
        bool, typer.Option("--skip-mcp", help="If set, skip MCP integration for this pipeline.")
    ] = False,
    save_file: Annotated[
        bool,
        typer.Option(
            "--save-file/--no-save-file",
            help="Whether to save the YAML under pipelines/{name}.yml on the server.",
        ),
    ] = True,
) -> None:
    """Deploy a YAML pipeline using the /deploy-yaml endpoint."""
    if not pipeline_file.exists():
        show_error_and_abort("Pipeline file does not exist.", str(pipeline_file))

    if name is None:
        name = pipeline_file.stem

    payload = {
        "name": name,
        "source_code": pipeline_file.read_text(),
        "overwrite": overwrite,
        "save_file": save_file,
        "skip_mcp": skip_mcp,
    }

    if description is not None:
        payload["description"] = description

    _deploy_with_progress(ctx=ctx, name=name, endpoint="deploy-yaml", payload=payload)


@pipeline.command()
def deploy_files(
    ctx: typer.Context,
    name: Annotated[Optional[str], typer.Option("--name", "-n", help="The name of the pipeline to deploy.")],
    pipeline_dir: Path = typer.Argument(  # noqa: B008
        help="The path to the directory containing the pipeline files to deploy."
    ),
    overwrite: Annotated[
        bool, typer.Option("--overwrite", "-o", help="Whether to overwrite the pipeline if it already exists.")
    ] = False,
    skip_saving_files: Annotated[
        bool, typer.Option("--skip-saving-files", "-s", help="Whether to skip saving the files to the server.")
    ] = False,
) -> None:
    """Deploy all pipeline files from a directory to the Hayhooks server."""
    if not pipeline_dir.exists():
        show_error_and_abort("Directory does not exist.", str(pipeline_dir))

    # Lazy import to avoid importing heavy server dependencies on CLI startup
    from hayhooks.server.utils.deploy_utils import read_pipeline_files_from_dir

    files_dict = read_pipeline_files_from_dir(pipeline_dir)

    if not files_dict:
        show_warning_panel("No valid pipeline files found in the specified directory.")
        raise typer.Abort()

    if name is None:
        name = pipeline_dir.stem

    payload = {"name": name, "files": files_dict, "save_files": not skip_saving_files, "overwrite": overwrite}
    _deploy_with_progress(ctx=ctx, name=name, endpoint="deploy_files", payload=payload)


@pipeline.command(name="deploy", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def deploy(_ctx: typer.Context) -> None:
    """Removed command; use 'deploy-yaml' or 'deploy-files' instead."""
    show_warning_panel(
        "[bold yellow]`hayhooks pipeline deploy` has been removed.[/bold yellow]\n"
        "Use: \n"
        "`hayhooks pipeline deploy-yaml <pipeline.yml>` for YAML-based deployments or\n"
        "`hayhooks pipeline deploy-files <pipeline_dir>` for PipelineWrapper-based deployments."
    )


@pipeline.command()
def undeploy(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The name of the pipeline to undeploy.")],
) -> None:
    """Undeploy a pipeline from the Hayhooks server."""
    response = with_progress_spinner(
        f"Undeploying pipeline '{name}'...",
        make_request,
        host=ctx.obj["host"],
        port=ctx.obj["port"],
        endpoint=f"undeploy/{name}",
        method="POST",
        use_https=ctx.obj["use_https"],
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
def run(  # noqa: PLR0912, C901
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The name of the pipeline to run.")],
    file: Annotated[
        Optional[list[Path]], typer.Option("--file", "-f", help="Files to upload (can be specified multiple times)")
    ] = None,
    directory: Annotated[
        Optional[list[Path]],
        typer.Option("--dir", "-d", help="Directories to upload (all files within will be uploaded)"),
    ] = None,
    param: Annotated[
        Optional[list[str]],
        typer.Option("--param", "-p", help="Parameters in format key=value (value can be string or JSON)"),
    ] = None,
) -> None:
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
                    (value.startswith("{") and value.endswith("}"))
                    or (value.startswith("[") and value.endswith("]"))
                    or value.lower() in ("true", "false", "null")
                    or (value.replace(".", "", 1).isdigit())
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
                if file_path.is_file() and not file_path.name.startswith("."):
                    # Use relative path as key to preserve directory structure
                    rel_path = file_path.relative_to(dir_path)
                    files_to_upload[str(rel_path)] = file_path

    # Run pipeline
    run_pipeline_with_files(ctx=ctx, pipeline_name=name, files=files_to_upload, params=params_dict)


def run_pipeline_with_files(
    ctx: typer.Context, pipeline_name: str, files: dict[str, Path], params: dict[str, Any]
) -> None:
    """Run a pipeline with files and parameters."""
    server_url = get_server_url(host=ctx.obj["host"], port=ctx.obj["port"], https=ctx.obj["use_https"])
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
        get_console().print(f"Running pipeline '[bold]{pipeline_name}[/bold]'...")
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
            use_https=ctx.obj["use_https"],
        )
        result = response

    # Display results
    show_success_panel(f"Pipeline '[bold]{pipeline_name}[/bold]' executed successfully!")

    # Display the result
    if "result" in result:
        get_console().print("\n[bold cyan]Result:[/bold cyan]")
        if isinstance(result["result"], (dict, list)):
            get_console().print_json(json.dumps(result["result"]))
        else:
            get_console().print(result["result"])
    else:
        get_console().print_json(json.dumps(result))
