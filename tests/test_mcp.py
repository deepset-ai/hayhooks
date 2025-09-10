import importlib.util
from pathlib import Path

import pytest

from hayhooks.server.pipelines import registry
from hayhooks.server.routers.deploy import PipelineFilesRequest
from hayhooks.server.utils.deploy_utils import add_pipeline_wrapper_to_registry
from hayhooks.server.utils.mcp_utils import (
    PIPELINE_NAME_SCHEMA,
    CoreTools,
    list_core_tools,
    list_pipelines_as_tools,
    run_pipeline_as_tool,
)

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None

# NOTE: Skip all tests in this file if MCP is not available
pytestmark = [
    pytest.mark.skipif(not MCP_AVAILABLE, reason="'mcp' package not installed"),
    pytest.mark.mcp,
]


@pytest.fixture(autouse=True)
def cleanup_test_pipelines():
    yield
    registry.clear()


@pytest.fixture
def deploy_chat_with_website_mcp():
    pipeline_wrapper_path = Path("tests/test_files/files/chat_with_website_mcp/pipeline_wrapper.py")
    pipeline_yml_path = Path("tests/test_files/files/chat_with_website_mcp/chat_with_website.yml")
    files = {
        "pipeline_wrapper.py": pipeline_wrapper_path.read_text(),
        "chat_with_website.yml": pipeline_yml_path.read_text(),
    }
    add_pipeline_wrapper_to_registry(pipeline_name="chat_with_website", files=files)


@pytest.fixture
def deploy_chat_with_website_mcp_skip():
    pipeline_wrapper_path = Path("tests/test_files/files/chat_with_website_mcp_skip/pipeline_wrapper.py")
    files = {
        "pipeline_wrapper.py": pipeline_wrapper_path.read_text(),
    }
    add_pipeline_wrapper_to_registry(pipeline_name="chat_with_website_mcp_skip", files=files)


@pytest.mark.asyncio
async def test_list_pipelines_as_tools_no_pipelines():
    tools = await list_pipelines_as_tools()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_list_pipelines_as_tools(deploy_chat_with_website_mcp):
    tools = await list_pipelines_as_tools()

    assert len(tools) == 1
    assert tools[0].name == "chat_with_website"
    assert tools[0].description == "Ask a question about one or more websites using a Haystack pipeline."
    assert tools[0].inputSchema == {
        "properties": {
            "urls": {"items": {"type": "string"}, "title": "Urls", "type": "array", "description": "Parameter 'urls'"},
            "question": {"title": "Question", "type": "string", "description": "Parameter 'question'"},
        },
        "required": ["urls", "question"],
        "title": "chat_with_websiteRunRequest",
        "type": "object",
    }


@pytest.mark.asyncio
async def test_list_pipeline_without_description():
    files = {
        "pipeline_wrapper.py": Path("tests/test_files/files/chat_with_website/pipeline_wrapper.py").read_text(),
        "chat_with_website.yml": Path("tests/test_files/files/chat_with_website/chat_with_website.yml").read_text(),
    }
    add_pipeline_wrapper_to_registry(pipeline_name="chat_with_website", files=files)

    tools = await list_pipelines_as_tools()

    assert len(tools) == 1
    assert tools[0].name == "chat_with_website"
    assert tools[0].description == ""


@pytest.mark.asyncio
async def test_fail_to_run_pipeline_as_tool():
    with pytest.raises(ValueError):
        await run_pipeline_as_tool("non_existent_pipeline", {})


@pytest.mark.asyncio
async def test_run_pipeline_as_tool_returns_text_content(deploy_chat_with_website_mcp):
    from mcp.types import TextContent

    result = await run_pipeline_as_tool(
        "chat_with_website", {"urls": ["https://www.google.com"], "question": "What is the capital of France?"}
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "This is a mock response from the pipeline"


@pytest.mark.asyncio
async def test_skip_pipeline_from_mcp_listing(deploy_chat_with_website_mcp_skip):
    tools = await list_pipelines_as_tools()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_list_core_tools():
    tools = await list_core_tools()
    assert len(tools) == len(CoreTools)

    core_tools_map = {tool.name: tool for tool in tools}

    tool_get_all_statuses = core_tools_map.get(CoreTools.GET_ALL_PIPELINE_STATUSES.value)
    assert tool_get_all_statuses is not None
    assert tool_get_all_statuses.description == "Get the status of all pipelines and list available pipeline names."
    assert tool_get_all_statuses.inputSchema == {"type": "object"}

    tool_get_status = core_tools_map.get(CoreTools.GET_PIPELINE_STATUS.value)
    assert tool_get_status is not None
    assert tool_get_status.description == "Get status of a specific pipeline."
    assert tool_get_status.inputSchema == PIPELINE_NAME_SCHEMA

    tool_undeploy = core_tools_map.get(CoreTools.UNDEPLOY_PIPELINE.value)
    assert tool_undeploy is not None
    assert (
        tool_undeploy.description
        == "Undeploy a pipeline. Removes a pipeline from the registry, its API routes, and deletes its files."
    )
    assert tool_undeploy.inputSchema == PIPELINE_NAME_SCHEMA

    tool_deploy = core_tools_map.get(CoreTools.DEPLOY_PIPELINE.value)
    assert tool_deploy is not None
    assert tool_deploy.description == "Deploy a pipeline from files (pipeline_wrapper.py and other files)."
    assert tool_deploy.inputSchema == PipelineFilesRequest.model_json_schema()
