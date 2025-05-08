import asyncio
import typer
import uvicorn
import sys
from typing import Annotated, List, Optional
from hayhooks.server.pipelines import registry
from hayhooks.server.utils.deploy_utils import deploy_pipeline_files, undeploy_pipeline
from hayhooks.settings import settings
from hayhooks.server.utils.mcp_utils import (
    CoreTools,
    deploy_pipelines,
    list_core_tools,
    list_pipelines_as_tools,
    run_pipeline_as_tool,
)
from hayhooks.server.logger import log
from haystack.lazy_imports import LazyImport

mcp = typer.Typer()

with LazyImport("Run 'pip install \"mcp\"' to install MCP.") as mcp_import:
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    from mcp.server import Server
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
    server: Server = Server("hayhooks-mcp-server")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        try:
            core_tools = await list_core_tools()
            log.debug(f"Listing {len(core_tools)} core tools")

            pipelines_tools = await list_pipelines_as_tools()
            log.debug(f"Listing {len(pipelines_tools)} pipelines as tools")

            return core_tools + pipelines_tools
        except Exception as e:
            log.error(f"Error listing tools: {e}")
            return []

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent | ImageContent | EmbeddedResource]:
        try:
            if name == CoreTools.DEPLOY_PIPELINE:
                result = await asyncio.to_thread(
                    deploy_pipeline_files,
                    pipeline_name=arguments["name"],
                    files=arguments["files"],
                    app=None,
                    save_files=arguments["save_files"],
                    overwrite=arguments["overwrite"],
                )
                return [TextContent(type="text", text=f"Pipeline '{result['name']}' deployed successfully")]

            elif name == CoreTools.GET_ALL_PIPELINE_STATUSES:
                pipelines = registry.get_names()
                pipelines_str = "\n".join(pipelines)
                return [TextContent(type="text", text=f"Available pipelines:\n{pipelines_str}")]

            elif name == CoreTools.GET_PIPELINE_STATUS:
                pipeline_name = arguments["pipeline_name"]
                is_deployed = pipeline_name in registry.get_names()
                return [TextContent(type="text", text=f"Pipeline '{pipeline_name}' is deployed: {is_deployed}")]

            elif name == CoreTools.UNDEPLOY_PIPELINE:
                pipeline_name = arguments["pipeline_name"]
                undeploy_pipeline(pipeline_name=pipeline_name)
                return [TextContent(type="text", text=f"Pipeline '{pipeline_name}' undeployed")]

            else:
                log.debug(f"Attempting to run tool '{name}' as a pipeline with arguments: {arguments}")
                try:
                    return await run_pipeline_as_tool(name, arguments)
                except Exception as e_pipeline:
                    log.error(f"Error calling pipeline tool '{name}': {e_pipeline}")
                    return []

        except KeyError as e_args:
            log.error(f"Missing argument for tool '{name}': {e_args}")
            return [TextContent(type="text", text=f"Error calling tool '{name}': Missing argument {e_args}.")]

        except Exception as e_general:
            log.error(f"General unhandled error in call_tool for tool '{name}': {e_general}")
            return [
                TextContent(
                    type="text", text=f"An unexpected error occurred while processing tool '{name}': {e_general}."
                )
            ]

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
