import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hayhooks.server.app import create_app
from hayhooks.server.pipelines.registry import registry
from hayhooks.settings import settings
from tests.pipeline_sources import CALC_YAML, ORDINARY_WRAPPER, write_tree


@pytest.fixture
def startup_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pipelines_dir = write_tree(tmp_path / "pipelines", {"double/pipeline_wrapper.py": ORDINARY_WRAPPER})
    monkeypatch.setattr(settings, "pipelines_dir", str(pipelines_dir))
    monkeypatch.setattr(settings, "durable_mode", False)
    registry.clear()
    yield pipelines_dir
    registry.clear()


@pytest.mark.parametrize("enabled", [True, False])
def test_runtime_deployment_routes(startup_pipeline: Path, tmp_path: Path, monkeypatch, enabled):
    monkeypatch.setattr(settings, "runtime_deploy_enabled", enabled)
    marker = tmp_path / "uploaded_code_executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + ORDINARY_WRAPPER

    with TestClient(create_app()) as client:
        assert client.post("/double/run", json={"value": 21}).json() == {"result": 42}
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/deploy_files", "/deploy-yaml", "/undeploy/{pipeline_name}"):
            assert (path in paths) is enabled

        response = client.post("/deploy_files", json={"name": "uploaded", "files": {"pipeline_wrapper.py": source}})
        assert response.status_code == (200 if enabled else 404)
        assert marker.exists() is enabled
        assert (startup_pipeline / "uploaded").exists() is enabled

        response = client.post("/deploy-yaml", json={"name": "calc", "source_code": CALC_YAML})
        assert response.status_code == (200 if enabled else 404)
        assert (startup_pipeline / "calc.yml").exists() is enabled

        response = client.post("/undeploy/double")
        assert response.status_code == (200 if enabled else 404)
        assert (startup_pipeline / "double/pipeline_wrapper.py").exists() is not enabled


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="MCP is not installed")
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_disabled_runtime_deployment_rejects_mcp_uploads(startup_pipeline: Path, tmp_path: Path, monkeypatch):
    from mcp.shared.memory import create_connected_server_and_client_session

    from hayhooks.server.utils import mcp_utils

    monkeypatch.setattr(settings, "runtime_deploy_enabled", False)
    marker = tmp_path / "uploaded_code_executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + ORDINARY_WRAPPER

    with TestClient(create_app()):
        server = mcp_utils.create_mcp_server()
        monkeypatch.setattr(settings, "runtime_deploy_enabled", True)
        async with create_connected_server_and_client_session(server) as session:
            tools = {tool.name for tool in (await session.list_tools()).tools}
            assert tools == {"double", "get_all_pipeline_statuses", "get_pipeline_status"}
            result = await session.call_tool("double", {"value": 21})
            assert result.content[0].text == "42"

            for tool, arguments in (
                ("deploy_pipeline", {"name": "uploaded", "files": {"pipeline_wrapper.py": source}}),
                ("undeploy_pipeline", {"pipeline_name": "double"}),
            ):
                result = await session.call_tool(tool, arguments)
                assert result.isError
                assert "HAYHOOKS_RUNTIME_DEPLOY_ENABLED=false" in result.content[0].text

    assert not marker.exists()
    assert not (startup_pipeline / "uploaded").exists()
    assert (startup_pipeline / "double/pipeline_wrapper.py").exists()
