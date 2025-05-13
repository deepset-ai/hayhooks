import asyncio
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Union
from hayhooks import log
from hayhooks.server.app import init_pipeline_dir
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.deploy_utils import (
    add_pipeline_to_registry,
    deploy_pipeline_files,
    read_pipeline_files_from_dir,
    undeploy_pipeline,
)
from hayhooks.settings import settings
from hayhooks.server.pipelines import registry
from haystack.lazy_imports import LazyImport
from hayhooks.server.routers.deploy import PipelineFilesRequest


with LazyImport("Run 'pip install \"mcp\"' to install MCP.") as mcp_import:
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    from mcp.server import Server


PIPELINE_NAME_SCHEMA = {
    "type": "object",
    "properties": {"pipeline_name": {"type": "string", "description": "Name of the pipeline"}},
    "required": ["pipeline_name"],
}


class CoreTools(str, Enum):
    GET_ALL_PIPELINE_STATUSES = "get_all_pipeline_statuses"
    GET_PIPELINE_STATUS = "get_pipeline_status"
    UNDEPLOY_PIPELINE = "undeploy_pipeline"
    DEPLOY_PIPELINE = "deploy_pipeline"


if TYPE_CHECKING:
    from mcp.types import TextContent, ImageContent, EmbeddedResource, Tool

with LazyImport("Run 'pip install \"mcp\"' to install MCP.") as mcp_import:
    from mcp.types import TextContent, ImageContent, EmbeddedResource, Tool


def deploy_pipelines() -> None:
    """Deploy pipelines from the configured directory"""
    pipelines_dir = init_pipeline_dir(settings.pipelines_dir)

    log.info(f"Pipelines dir set to: {pipelines_dir}")
    pipelines_path = Path(pipelines_dir)

    pipeline_dirs = [d for d in pipelines_path.iterdir() if d.is_dir()]
    log.debug(f"Found {len(pipeline_dirs)} pipeline directories")

    for pipeline_dir in pipeline_dirs:
        log.debug(f"Deploying pipeline from {pipeline_dir}")

        try:
            add_pipeline_to_registry(pipeline_name=pipeline_dir.name, files=read_pipeline_files_from_dir(pipeline_dir))
        except Exception as e:
            log.warning(f"Skipping pipeline directory {pipeline_dir}: {str(e)}")
            continue


async def list_core_tools() -> List["Tool"]:
    """List available Hayhooks core tools"""
    mcp_import.check()

    tools = [
        Tool(
            name=CoreTools.GET_ALL_PIPELINE_STATUSES,
            description="Get the status of all pipelines and list available pipeline names.",
            inputSchema={
                "type": "object",
            },
        ),
        Tool(
            name=CoreTools.GET_PIPELINE_STATUS,
            description="Get status of a specific pipeline.",
            inputSchema=PIPELINE_NAME_SCHEMA,
        ),
        Tool(
            name=CoreTools.UNDEPLOY_PIPELINE,
            description="Undeploy a pipeline. Removes a pipeline from the registry, its API routes, and deletes its files.",
            inputSchema=PIPELINE_NAME_SCHEMA,
        ),
        Tool(
            name=CoreTools.DEPLOY_PIPELINE,
            description="Deploy a pipeline from files (pipeline_wrapper.py and other files).",
            inputSchema=PipelineFilesRequest.model_json_schema(),
        ),
    ]

    return tools


async def list_pipelines_as_tools() -> List["Tool"]:
    """List available pipelines as MCP tools"""
    mcp_import.check()

    tools = []

    for pipeline_name in registry.get_names():
        metadata = registry.get_metadata(name=pipeline_name) or {}
        log.trace(f"Metadata for pipeline '{pipeline_name}': {metadata}")

        if not metadata.get("request_model"):
            log.warning(f"Skipping pipeline '{pipeline_name}' as it has no request model")
            continue

        if metadata.get("skip_mcp"):
            log.debug(f"Skipping pipeline '{pipeline_name}' as it has skip_mcp set to True")
            continue

        tools.append(
            Tool(
                name=pipeline_name,
                description=metadata.get("description", ""),
                inputSchema=metadata["request_model"].model_json_schema(),
            )
        )
        log.debug(f"Added pipeline as MCP tool '{pipeline_name}' with description: '{metadata['description']}'")

    log.debug(f"Pipelines listed as MCP tools: {[tool.name for tool in tools]}")

    return tools


async def run_pipeline_as_tool(
    name: str, arguments: dict
) -> List[Union["TextContent", "ImageContent", "EmbeddedResource"]]:
    mcp_import.check()

    log.debug(f"Calling pipeline as tool '{name}' with arguments: {arguments}")
    pipeline_wrapper: Union[BasePipelineWrapper, None] = registry.get(name)

    if not pipeline_wrapper:
        raise ValueError(f"Pipeline '{name}' not found")

    result = await asyncio.to_thread(pipeline_wrapper.run_api, **arguments)
    log.trace(f"Pipeline '{name}' returned result: {result}")

    return [TextContent(text=result, type="text")]


async def notify_client(server: "Server") -> None:
    await server.request_context.session.send_tool_list_changed()


async def create_mcp_server(name: str = "hayhooks-mcp-server") -> "Server":
    server: Server = Server(name)

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
                log.debug(f"Attempting to run pipeline '{name}' as MCP Tool with arguments: {arguments}")

                try:
                    return await run_pipeline_as_tool(name, arguments)
                except Exception as e_pipeline:
                    log.error(f"Error calling pipeline as MCP Tool '{name}': {e_pipeline}")
                    return [TextContent(type="text", text=f"Error calling tool '{name}': {e_pipeline}")]

        except Exception as exc:
            log.error(f"General unhandled error in call_tool for tool '{name}': {exc}")
            return [
                TextContent(type="text", text=f"An unexpected error occurred while processing tool '{name}': {exc}.")
            ]

        finally:
            if name in [CoreTools.DEPLOY_PIPELINE, CoreTools.UNDEPLOY_PIPELINE]:
                log.debug(f"Sending 'tools/list_changed' notification after deploy/undeploy")
                await notify_client(server)

    return server
