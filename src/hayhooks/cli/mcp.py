import asyncio
import typer
import uvicorn
import sys
from typing import Annotated, Optional
from hayhooks.settings import settings
from hayhooks.server.utils.mcp_utils import (
    create_mcp_server,
    deploy_pipelines,
)
from hayhooks.server.logger import log
from haystack.lazy_imports import LazyImport

mcp = typer.Typer()

with LazyImport("Run 'pip install \"mcp\"' to install MCP.") as mcp_import:
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route


@mcp.command()
def run(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to run the MCP server on")] = settings.mcp_host,
    port: Annotated[int, typer.Option("--port", "-p", help="Port to run the MCP server on")] = settings.mcp_port,
    pipelines_dir: Annotated[
        str, typer.Option("--pipelines-dir", "-d", help="Directory containing the pipelines")
    ] = settings.pipelines_dir,
    additional_python_path: Annotated[
        Optional[str], typer.Option(help="Additional Python path to add to sys.path")
    ] = settings.additional_python_path,
):
    """
    Run the MCP server.
    """
    mcp_import.check()

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
    server = create_mcp_server()

    # Setup the SSE server
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    # Run the MCP server
    # NOTE: reload and workers options are not supported in this context
    uvicorn.run(app, host=host, port=port)
