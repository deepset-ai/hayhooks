import sys
from typing import Annotated, Optional

import typer
from haystack.lazy_imports import LazyImport

mcp = typer.Typer()

with LazyImport("Run 'pip install \"mcp\"' to install MCP.") as mcp_import:
    from mcp.server import Server


@mcp.command()
def run(  # noqa: PLR0913
    host: Annotated[Optional[str], typer.Option("--host", "-h", help="Host to run the MCP server on")] = None,
    port: Annotated[Optional[int], typer.Option("--port", "-p", help="Port to run the MCP server on")] = None,
    pipelines_dir: Annotated[
        Optional[str], typer.Option("--pipelines-dir", "-d", help="Directory containing the pipelines")
    ] = None,
    additional_python_path: Annotated[
        Optional[str], typer.Option(help="Additional Python path to add to sys.path")
    ] = None,
    json_response: Annotated[
        bool, typer.Option("--json-response", "-j", help="Enable JSON responses instead of SSE streams")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="If true, tracebacks should be returned on errors")
    ] = False,
) -> None:
    """
    Run the MCP server.
    """
    # Lazy imports of settings, logger and uvicorn
    import uvicorn

    from hayhooks.server.logger import log
    from hayhooks.server.utils.mcp_utils import create_mcp_server, create_starlette_app, deploy_pipelines
    from hayhooks.settings import settings

    mcp_import.check()

    # Fill defaults from settings when command executes
    host = host or settings.mcp_host
    port = port or settings.mcp_port
    pipelines_dir = pipelines_dir or settings.pipelines_dir

    settings.mcp_host = host
    settings.mcp_port = port
    settings.pipelines_dir = pipelines_dir

    if additional_python_path:
        settings.additional_python_path = additional_python_path
        sys.path.append(additional_python_path)
        log.trace(f"Added {additional_python_path} to sys.path")

    # Deploy the pipelines
    deploy_pipelines()

    # Setup the MCP server
    server: Server = create_mcp_server()

    # Setup the Starlette app
    app = create_starlette_app(server, debug=debug, json_response=json_response)

    # Run the MCP server
    # NOTE: reload and workers options are not supported in this context
    uvicorn.run(app, host=host, port=port)
