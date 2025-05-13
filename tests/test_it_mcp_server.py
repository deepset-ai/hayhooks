import pytest
import importlib.util
from unittest.mock import patch
from pathlib import Path
from hayhooks.server.pipelines import registry
from hayhooks.server.utils.deploy_utils import add_pipeline_to_registry
from hayhooks.server.utils.mcp_utils import create_mcp_server, CoreTools
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)


MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None

# NOTE: Skip all tests in this file if MCP is not available
pytestmark = [
    pytest.mark.skipif(not MCP_AVAILABLE, reason="'mcp' package not installed"),
    pytest.mark.mcp,
    pytest.mark.integration,
]

# Conditional import for mcp types if needed, though skipif should guard tests
if MCP_AVAILABLE:
    from mcp.server import Server
    from mcp.types import TextContent


@pytest.fixture(autouse=True)
def cleanup_test_pipelines():
    registry.clear()
    yield


@pytest.fixture
def deploy_chat_with_website_mcp_pipeline():
    pipeline_name = "chat_with_website"
    pipeline_wrapper_path = Path("tests/test_files/files/chat_with_website_mcp/pipeline_wrapper.py")
    pipeline_yml_path = Path("tests/test_files/files/chat_with_website_mcp/chat_with_website.yml")
    files = {
        "pipeline_wrapper.py": pipeline_wrapper_path.read_text(),
        "chat_with_website.yml": pipeline_yml_path.read_text(),
    }
    add_pipeline_to_registry(pipeline_name=pipeline_name, files=files)
    return pipeline_name


@pytest.fixture
async def mcp_server_instance() -> "Server":
    return await create_mcp_server()


@pytest.mark.asyncio
async def test_list_only_core_tools(mcp_server_instance):

    async with client_session(mcp_server_instance) as client:
        list_tools_result = await client.list_tools()

        # With no pipelines deployed, the server should only have the core tools
        assert len(list_tools_result.tools) == len(CoreTools)


@pytest.mark.asyncio
async def test_list_tools_with_one_pipeline_deployed(mcp_server_instance, deploy_chat_with_website_mcp_pipeline):
    async with client_session(mcp_server_instance) as client:
        list_tools_result = await client.list_tools()

        # With one pipeline deployed, the server should have the core tools plus the pipeline tool
        assert len(list_tools_result.tools) == len(CoreTools) + 1

        # Find the tool for the deployed pipeline
        pipeline_tool = next(
            (tool for tool in list_tools_result.tools if tool.name == deploy_chat_with_website_mcp_pipeline), None
        )
        assert pipeline_tool is not None

        # Check if the tool is the correct one
        assert pipeline_tool.name == deploy_chat_with_website_mcp_pipeline
        assert pipeline_tool.description == "Ask a question about one or more websites using a Haystack pipeline."
        assert pipeline_tool.inputSchema == {
            'properties': {
                'urls': {
                    'description': "Parameter 'urls'",
                    'items': {'type': 'string'},
                    'title': 'Urls',
                    'type': 'array',
                },
                'question': {'description': "Parameter 'question'", 'title': 'Question', 'type': 'string'},
            },
            'required': ['urls', 'question'],
            'title': 'chat_with_websiteRunRequest',
            'type': 'object',
        }


@pytest.mark.asyncio
async def test_call_pipeline_as_tool(mcp_server_instance, deploy_chat_with_website_mcp_pipeline):
    async with client_session(mcp_server_instance) as client:
        result = await client.call_tool(
            deploy_chat_with_website_mcp_pipeline,
            {"urls": ["https://www.google.com"], "question": "What is the capital of France?"},
        )

        # In the deployed pipeline, the response is mocked
        assert result.content == [TextContent(type="text", text="This is a mock response from the pipeline")]


@pytest.mark.asyncio
async def test_call_pipeline_as_tool_with_invalid_arguments(mcp_server_instance, deploy_chat_with_website_mcp_pipeline):
    async with client_session(mcp_server_instance) as client:
        result = await client.call_tool(
            deploy_chat_with_website_mcp_pipeline,
            {"urls": ["https://www.google.com"]},
        )

        text_response = result.content[0].text
        assert "missing 1 required positional argument: 'question'" in text_response


@pytest.mark.asyncio
async def test_call_pipeline_as_tool_with_invalid_pipeline_name(mcp_server_instance):
    async with client_session(mcp_server_instance) as client:
        result = await client.call_tool(
            "invalid_pipeline_name",
            {"urls": ["https://www.google.com"], "question": "What is the capital of France?"},
        )

        text_response = result.content[0].text
        assert "Pipeline 'invalid_pipeline_name' not found" in text_response


@pytest.mark.asyncio
async def test_ensure_send_tool_list_changed_notification_after_deploy_or_undeploy(mcp_server_instance):
    with patch("hayhooks.server.utils.mcp_utils.notify_client") as mock_notify_client:
        async with client_session(mcp_server_instance) as client:

            result = await client.call_tool(
                CoreTools.DEPLOY_PIPELINE,
                {
                    "name": "chat_with_website",
                    "files": {
                        "pipeline_wrapper.py": Path(
                            "tests/test_files/files/chat_with_website_mcp/pipeline_wrapper.py"
                        ).read_text(),
                        "chat_with_website.yml": Path(
                            "tests/test_files/files/chat_with_website_mcp/chat_with_website.yml"
                        ).read_text(),
                    },
                    "save_files": True,
                    "overwrite": False,
                },
            )

            assert result.content[0].text == "Pipeline 'chat_with_website' deployed successfully"

        mock_notify_client.assert_called_once()
