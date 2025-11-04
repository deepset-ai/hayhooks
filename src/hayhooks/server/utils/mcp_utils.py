import asyncio
import json
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Union

from fastapi.concurrency import run_in_threadpool
from haystack import AsyncPipeline
from haystack.lazy_imports import LazyImport
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from hayhooks.server.app import init_pipeline_dir
from hayhooks.server.logger import log
from hayhooks.server.pipelines import registry
from hayhooks.server.pipelines.registry import PipelineType
from hayhooks.server.routers.deploy import PipelineFilesRequest
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.deploy_utils import (
    add_pipeline_wrapper_to_registry,
    deploy_pipeline_files,
    read_pipeline_files_from_dir,
    undeploy_pipeline,
)
from hayhooks.settings import settings

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


# Lazily import MCP modules so the optional dependency is only required when used
with LazyImport("Run 'pip install \"mcp\"' to install MCP.") as mcp_import:
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.types import TextContent, Tool


def deploy_pipelines() -> None:
    """Deploy pipelines from the configured directory"""
    pipelines_dir = init_pipeline_dir(settings.pipelines_dir)

    log.info("Pipelines dir set to: '{}'", pipelines_dir)
    pipelines_path = Path(pipelines_dir)

    pipeline_dirs = [d for d in pipelines_path.iterdir() if d.is_dir()]
    log.debug("Found {} pipeline directories", len(pipeline_dirs))

    for pipeline_dir in pipeline_dirs:
        log.debug("Deploying pipeline from '{}'", pipeline_dir)

        try:
            add_pipeline_wrapper_to_registry(
                pipeline_name=pipeline_dir.name, files=read_pipeline_files_from_dir(pipeline_dir)
            )
        except Exception as e:
            log.warning("Skipping pipeline directory '{}': {}", pipeline_dir, e)
            continue


async def list_core_tools() -> list["Tool"]:
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
            description=(
                "Undeploy a pipeline. Removes a pipeline from the registry, its API routes, and deletes its files."
            ),
            inputSchema=PIPELINE_NAME_SCHEMA,
        ),
        Tool(
            name=CoreTools.DEPLOY_PIPELINE,
            description="Deploy a pipeline from files (pipeline_wrapper.py and other files).",
            inputSchema=PipelineFilesRequest.model_json_schema(),
        ),
    ]

    return tools


async def list_pipelines_as_tools() -> list["Tool"]:
    """List available pipelines as MCP tools"""
    mcp_import.check()

    tools = []

    for pipeline_name in registry.get_names():
        metadata = registry.get_metadata(name=pipeline_name) or {}
        log.trace("Metadata for pipeline '{}': {}", pipeline_name, metadata)

        if not metadata.get("request_model"):
            log.warning("Skipping pipeline '{}' as it has no request model", pipeline_name)
            continue

        if metadata.get("skip_mcp"):
            log.debug("Skipping pipeline '{}' as it has skip_mcp set to True", pipeline_name)
            continue

        tools.append(
            Tool(
                name=pipeline_name,
                description=metadata.get("description", ""),
                inputSchema=metadata["request_model"].model_json_schema(),
            )
        )
        log.debug("Added pipeline as MCP tool '{}' with description: '{}'", pipeline_name, metadata["description"])

    log.debug("Pipelines listed as MCP tools: {}", [tool.name for tool in tools])

    return tools


async def run_pipeline_as_tool(name: str, arguments: dict) -> list["TextContent"]:
    mcp_import.check()

    log.debug("Calling pipeline as tool '{}' with arguments: {}", name, arguments)
    pipeline: Union[PipelineType, None] = registry.get(name)

    if not pipeline:
        msg = f"Pipeline '{name}' not found"
        raise ValueError(msg)

    if isinstance(pipeline, BasePipelineWrapper):
        if pipeline._is_run_api_async_implemented:
            result = await pipeline.run_api_async(**arguments)
        else:
            result = await run_in_threadpool(pipeline.run_api, **arguments)

        log.trace("Pipeline '{}' returned result: {}", name, result)
        return [TextContent(text=result, type="text")]

    if isinstance(pipeline, AsyncPipeline):
        # Get include_outputs_from from metadata if available
        metadata = registry.get_metadata(name) or {}
        outputs_to_include = metadata.get("include_outputs_from")
        kwargs: dict[str, Any] = {
            "data": arguments,
        }

        if outputs_to_include is not None:
            kwargs["include_outputs_from"] = outputs_to_include

        result = await pipeline.run_async(**kwargs)

        log.trace("YAML Pipeline '{}' returned result: {}", name, result)
        return [TextContent(text=json.dumps(result), type="text")]

    msg = (
        f"Pipeline '{name}' is not a supported type for MCP tools. "
        "Expected a BasePipelineWrapper or AsyncPipeline instance."
    )
    raise ValueError(msg)


async def notify_client(server: "Server") -> None:
    await server.request_context.session.send_tool_list_changed()


def create_mcp_server(name: str = "hayhooks-mcp-server") -> "Server":  # noqa: C901
    mcp_import.check()

    server: Server = Server(name)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        try:
            core_tools = await list_core_tools()
            log.debug("Listing {} core tools", len(core_tools))

            pipelines_tools = await list_pipelines_as_tools()
            log.debug("Listing {} pipelines as tools", len(pipelines_tools))

            return core_tools + pipelines_tools
        except Exception as e:
            log.error("Error listing tools: {}", e)
            return []

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list["TextContent"]:
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
                log.debug(
                    "Attempting to run pipeline '{}' as MCP Tool with arguments: {}",
                    name,
                    arguments,
                )

                try:
                    return await run_pipeline_as_tool(name, arguments)
                except Exception as e_pipeline:
                    msg = "Error calling pipeline as MCP Tool '{name}': {e_pipeline}"
                    log.error(msg, name=name, e_pipeline=e_pipeline, exc_info=True)
                    raise Exception(msg.format(name=name, e_pipeline=e_pipeline)) from e_pipeline

        except Exception as exc:
            msg = "General unhandled error in call_tool for tool '{name}': {exc}"
            if settings.show_tracebacks:
                msg += f"\n{traceback.format_exc()}"
            log.error(msg, name=name, exc=exc, exc_info=True)
            raise Exception(msg.format(name=name, exc=exc)) from exc

        finally:
            if name in [CoreTools.DEPLOY_PIPELINE, CoreTools.UNDEPLOY_PIPELINE]:
                log.debug("Sending 'tools/list_changed' notification after deploy/undeploy")
                await notify_client(server)

    return server


def create_starlette_app(server: "Server", *, debug: bool = False, json_response: bool = False) -> "Starlette":
    """
    Create a Starlette app for the MCP server.
    """
    mcp_import.check()

    # Setup the Streamable HTTP session manager
    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=json_response,
        stateless=True,
    )

    # Setup the SSE server
    sse = SseServerTransport("/messages/")

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:  # noqa: ARG001
        async with session_manager.run():
            log.info("Hayhooks MCP server started")
            try:
                yield
            finally:
                log.info("Hayhooks MCP server shutting down...")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    async def handle_status(request):  # noqa: ARG001
        return JSONResponse({"status": "ok"})

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/status", endpoint=handle_status),
            Mount("/messages/", app=sse.handle_post_message),
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )
