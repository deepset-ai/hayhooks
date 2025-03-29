import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, List, Union
from hayhooks import log
from hayhooks.server.app import init_pipeline_dir
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.deploy_utils import add_pipeline_to_registry, read_pipeline_files_from_dir
from hayhooks.settings import settings
from hayhooks.server.pipelines import registry
from haystack.lazy_imports import LazyImport


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
