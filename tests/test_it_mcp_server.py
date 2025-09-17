import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from hayhooks.server.pipelines import registry
from hayhooks.server.utils.deploy_utils import add_pipeline_wrapper_to_registry, add_yaml_pipeline_to_registry
from hayhooks.server.utils.mcp_utils import CoreTools, create_mcp_server

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None

# NOTE: Skip all tests in this file if MCP is not available
pytestmark = [
    pytest.mark.skipif(not MCP_AVAILABLE, reason="'mcp' package not installed"),
    pytest.mark.mcp,
]

# Conditional import for mcp types if needed, though skipif should guard tests
if MCP_AVAILABLE:
    from mcp.server import Server
    from mcp.shared.memory import create_connected_server_and_client_session as client_session
    from mcp.types import CallToolResult, TextContent


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
    add_pipeline_wrapper_to_registry(pipeline_name=pipeline_name, files=files)
    return pipeline_name


@pytest.fixture
def deploy_async_question_answer_mcp_pipeline():
    pipeline_name = "async_question_answer"
    pipeline_wrapper_path = Path("tests/test_files/files/async_question_answer/pipeline_wrapper.py")
    pipeline_yml_path = Path("tests/test_files/files/async_question_answer/question_answer.yml")
    files = {
        "pipeline_wrapper.py": pipeline_wrapper_path.read_text(),
        "question_answer.yml": pipeline_yml_path.read_text(),
    }
    add_pipeline_wrapper_to_registry(pipeline_name=pipeline_name, files=files)
    return pipeline_name


@pytest.fixture
def mcp_server_instance() -> "Server":
    return create_mcp_server()


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
            "properties": {
                "urls": {
                    "description": "Parameter 'urls'",
                    "items": {"type": "string"},
                    "title": "Urls",
                    "type": "array",
                },
                "question": {"description": "Parameter 'question'", "title": "Question", "type": "string"},
            },
            "required": ["urls", "question"],
            "title": "chat_with_websiteRunRequest",
            "type": "object",
        }


@pytest.mark.asyncio
async def test_call_pipeline_as_tool(mcp_server_instance, deploy_chat_with_website_mcp_pipeline):
    async with client_session(mcp_server_instance) as client:
        result = await client.call_tool(
            deploy_chat_with_website_mcp_pipeline,
            {"urls": ["https://www.google.com"], "question": "What is the capital of France?"},
        )

        assert isinstance(result, CallToolResult)

        # In the deployed pipeline, the response is mocked
        assert result.content == [TextContent(type="text", text="This is a mock response from the pipeline")]


@pytest.mark.asyncio
async def test_call_async_pipeline_as_tool(mcp_server_instance, deploy_async_question_answer_mcp_pipeline):
    async with client_session(mcp_server_instance) as client:
        result = await client.call_tool(
            deploy_async_question_answer_mcp_pipeline, {"question": "What is the capital of France?"}
        )

        assert isinstance(result, CallToolResult)

        # In the deployed pipeline, the response is mocked
        assert result.content == [TextContent(type="text", text="This is a mock response from the pipeline")]


@pytest.mark.asyncio
async def test_call_pipeline_as_tool_with_invalid_arguments(mcp_server_instance, deploy_chat_with_website_mcp_pipeline):
    async with client_session(mcp_server_instance) as client:
        result = await client.call_tool(
            deploy_chat_with_website_mcp_pipeline,
            {"urls": ["https://www.google.com"]},
        )

        assert isinstance(result, CallToolResult)
        assert result.isError is True

        text_response = result.content[0].text
        assert "Input validation error: 'question' is a required property" in text_response


@pytest.mark.asyncio
async def test_call_pipeline_as_tool_with_invalid_pipeline_name(mcp_server_instance):
    async with client_session(mcp_server_instance) as client:
        result = await client.call_tool(
            "invalid_pipeline_name",
            {"urls": ["https://www.google.com"], "question": "What is the capital of France?"},
        )

        assert isinstance(result, CallToolResult)
        assert result.isError is True

        text_response = result.content[0].text
        assert "Pipeline 'invalid_pipeline_name' not found" in text_response


@pytest.fixture
def deploy_yaml_calc_pipeline():
    pipeline_name = "calc"
    yaml_path = Path("tests/test_files/yaml/sample_calc_pipeline.yml")
    add_yaml_pipeline_to_registry(pipeline_name=pipeline_name, source_code=yaml_path.read_text())
    return pipeline_name


@pytest.mark.asyncio
async def test_list_tools_with_yaml_pipeline_deployed(mcp_server_instance, deploy_yaml_calc_pipeline):
    async with client_session(mcp_server_instance) as client:
        list_tools_result = await client.list_tools()

        # Core tools + 1 YAML pipeline tool
        assert len(list_tools_result.tools) == len(CoreTools) + 1

        # Find YAML pipeline tool and verify basic schema
        pipeline_tool = next((t for t in list_tools_result.tools if t.name == deploy_yaml_calc_pipeline), None)
        assert pipeline_tool is not None
        assert pipeline_tool.inputSchema["type"] == "object"
        assert "value" in pipeline_tool.inputSchema["properties"]


@pytest.mark.asyncio
async def test_call_yaml_pipeline_as_tool(mcp_server_instance, deploy_yaml_calc_pipeline):
    async with client_session(mcp_server_instance) as client:
        result = await client.call_tool(deploy_yaml_calc_pipeline, {"value": 3})

        assert isinstance(result, CallToolResult)
        assert result.isError is False

        # YAML pipelines return JSON text content; parse and assert
        payload = result.content[0].text
        import json

        parsed = json.loads(payload)
        assert parsed == {"double": {"value": 10}}


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


@pytest.mark.asyncio
async def test_call_tool_general_exception_handler(mcp_server_instance):
    async with client_session(mcp_server_instance) as client:
        result = await client.call_tool(
            CoreTools.DEPLOY_PIPELINE,
            {"name": "chat_with_website", "files": {}, "save_files": False, "overwrite": False},
        )

        assert isinstance(result, CallToolResult)
        assert result.isError is True

        assert result.content[0].text.startswith("General unhandled error in call_tool for tool '")
        assert "Failed to load pipeline module" in result.content[0].text


def test_mcp_http_transport_initialize(test_mcp_client):
    with test_mcp_client as client:
        init_response = client.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            },
            headers={"Content-Type": "application/json", "Accept": "application/json,text/event-stream"},
        )

        assert init_response.status_code == 200
        init_data = init_response.json()

        # Verify JSON-RPC 2.0 response format
        assert init_data["jsonrpc"] == "2.0"
        assert init_data["id"] == 1
        assert "result" in init_data

        # Verify server info in response
        result = init_data["result"]
        assert "serverInfo" in result and result["serverInfo"]["name"] == "hayhooks-mcp-server"
        assert "capabilities" in result and result["capabilities"]["tools"]["listChanged"] is False


def test_sse_transport(test_mcp_client):
    with test_mcp_client as client:
        # Test that the SSE endpoints exist and don't return 404
        # We're not testing full SSE functionality since it's deprecated and can hang

        # Test the SSE connection endpoint exists
        sse_response = client.post("/sse/")
        assert sse_response.status_code != 404

        # Test the messages endpoint with a simple request (not SSE)
        # This verifies the endpoint exists and can handle requests
        messages_response = client.get("/messages/")
        assert messages_response.status_code != 404


def test_mcp_server_status_endpoint(test_mcp_client):
    with test_mcp_client as client:
        response = client.get("/status")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
