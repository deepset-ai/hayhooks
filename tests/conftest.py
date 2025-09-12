import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hayhooks.server.app import create_app
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.mcp_utils import create_mcp_server, create_starlette_app
from hayhooks.settings import settings


def pytest_configure(config):
    config.addinivalue_line("markers", "mcp: mark tests that require the MCP package")


@pytest.fixture(scope="session", autouse=True)
def test_settings():
    settings.pipelines_dir = str(Path(__file__).parent / "pipelines")
    return settings


@pytest.fixture(scope="session", autouse=True)
def test_app():
    return create_app()


@pytest.fixture
def test_mcp_server():
    return create_starlette_app(create_mcp_server(), debug=True, json_response=True)


@pytest.fixture
def test_mcp_client(test_mcp_server):
    return TestClient(test_mcp_server)


@pytest.fixture
def client(test_app: FastAPI):
    return TestClient(test_app)


@pytest.fixture(scope="module", autouse=True)
def cleanup_pipelines(test_settings):
    """
    This fixture is used to cleanup the pipelines directory
    and the registry after each test module.
    """
    yield
    registry.clear()
    if Path(test_settings.pipelines_dir).exists():
        shutil.rmtree(test_settings.pipelines_dir)


@pytest.fixture
def deploy_yaml_pipeline():
    def _deploy_yaml_pipeline(client: TestClient, pipeline_name: str, pipeline_source_code: str):
        deploy_response = client.post("/deploy-yaml", json={"name": pipeline_name, "source_code": pipeline_source_code})
        return deploy_response

    return _deploy_yaml_pipeline


@pytest.fixture
def undeploy_pipeline():
    def _undeploy_pipeline(client: TestClient, pipeline_name: str):
        undeploy_response = client.post(f"/undeploy/{pipeline_name}")
        return undeploy_response

    return _undeploy_pipeline


@pytest.fixture
def draw_pipeline():
    def _draw_pipeline(client: TestClient, pipeline_name: str):
        draw_response = client.get(f"/draw/{pipeline_name}")
        return draw_response

    return _draw_pipeline


@pytest.fixture
def status_pipeline():
    def _status_pipeline(client: TestClient, pipeline_name: str):
        status_response = client.get(f"/status/{pipeline_name}")
        return status_response

    return _status_pipeline


@pytest.fixture
def chat_completion():
    def _chat_completion(client: TestClient, pipeline_name: str, messages: list):
        chat_response = client.post(f"/{pipeline_name}/chat", json={"messages": messages, "model": pipeline_name})
        return chat_response

    return _chat_completion


@pytest.fixture
def deploy_files():
    def _deploy_files(
        client: TestClient, pipeline_name: str, pipeline_files: dict, overwrite: bool = False, save_files: bool = True
    ):
        deploy_response = client.post(
            "/deploy_files",
            json={"name": pipeline_name, "files": pipeline_files, "overwrite": overwrite, "save_files": save_files},
        )
        return deploy_response

    return _deploy_files
