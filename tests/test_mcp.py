import pytest
from hayhooks.server.pipelines import registry
from pathlib import Path
from hayhooks.server.utils.deploy_utils import add_pipeline_to_registry
from hayhooks.server.utils.mcp_utils import list_pipelines_as_tools, run_pipeline_as_tool


@pytest.fixture(autouse=True)
def cleanup_test_pipelines():
    yield
    registry.clear()


@pytest.mark.asyncio
async def test_list_pipelines_as_tools_no_pipelines():
    tools = await list_pipelines_as_tools()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_list_pipelines_as_tools():
    # Add a pipeline to the registry (no need to create API routes for MCP)
    pipeline_wrapper_path = Path("tests/test_files/files/chat_with_website_mcp/pipeline_wrapper.py")
    pipeline_yml_path = Path("tests/test_files/files/chat_with_website_mcp/chat_with_website.yml")
    files = {
        "pipeline_wrapper.py": pipeline_wrapper_path.read_text(),
        "chat_with_website.yml": pipeline_yml_path.read_text(),
    }

    add_pipeline_to_registry(pipeline_name="chat_with_website", files=files)

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
async def test_skip_pipeline_without_description():
    # Add a pipeline without a docstring for run_api method
    pipeline_wrapper_path = Path("tests/test_files/files/chat_with_website/pipeline_wrapper.py")
    pipeline_yml_path = Path("tests/test_files/files/chat_with_website/chat_with_website.yml")

    files = {
        "pipeline_wrapper.py": pipeline_wrapper_path.read_text(),
        "chat_with_website.yml": pipeline_yml_path.read_text(),
    }

    add_pipeline_to_registry(pipeline_name="chat_with_website", files=files)

    tools = await list_pipelines_as_tools()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_fail_to_run_pipeline_as_tool():
    with pytest.raises(ValueError):
        await run_pipeline_as_tool("non_existent_pipeline", {})
