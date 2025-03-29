import pytest
import importlib.util
from hayhooks.server.pipelines import registry
from pathlib import Path
from hayhooks.server.utils.deploy_utils import add_pipeline_to_registry
from hayhooks.server.utils.mcp_utils import list_pipelines_as_tools, run_pipeline_as_tool


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
    add_pipeline_to_registry(pipeline_name="chat_with_website", files=files)


@pytest.fixture
def deploy_chat_with_website_mcp_skip():
    pipeline_wrapper_path = Path("tests/test_files/files/chat_with_website_mcp_skip/pipeline_wrapper.py")
    files = {
        "pipeline_wrapper.py": pipeline_wrapper_path.read_text(),
    }
    add_pipeline_to_registry(pipeline_name="chat_with_website_mcp_skip", files=files)


@pytest.mark.asyncio
async def test_list_pipelines_as_tools_no_pipelines():
    tools = await list_pipelines_as_tools()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_list_pipelines_as_tools(deploy_chat_with_website_mcp):
    # List the pipelines as tools
    tools = await list_pipelines_as_tools()

    assert len(tools) == 1
    assert tools[0].name == "chat_with_website"
    assert tools[0].description == "Ask a question about one or more websites using a Haystack pipeline."
    assert tools[0].inputSchema == {
        'properties': {
            'urls': {'items': {'type': 'string'}, 'title': 'Urls', 'type': 'array'},
            'question': {'title': 'Question', 'type': 'string'},
        },
        'required': ['urls', 'question'],
        'title': 'chat_with_websiteRunRequest',
        'type': 'object',
    }


@pytest.mark.asyncio
async def test_list_pipeline_without_description():
    files = {
        "pipeline_wrapper.py": Path("tests/test_files/files/chat_with_website/pipeline_wrapper.py").read_text(),
        "chat_with_website.yml": Path("tests/test_files/files/chat_with_website/chat_with_website.yml").read_text(),
    }
    add_pipeline_to_registry(pipeline_name="chat_with_website", files=files)

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
