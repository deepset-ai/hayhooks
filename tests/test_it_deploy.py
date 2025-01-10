import pytest
from fastapi.testclient import TestClient
from hayhooks.server import app
from pathlib import Path
from hayhooks.server.pipelines.registry import registry

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()


# Load pipeline definitions from test_files
test_files = Path(__file__).parent / "test_files" / "working_pipelines"
pipeline_data = [{"name": file.stem, "source_code": file.read_text()} for file in test_files.glob("*.yml")]


def deploy_pipeline(pipeline_data: dict):
    deploy_response = client.post(
        "/deploy", json={"name": pipeline_data["name"], "source_code": pipeline_data["source_code"]}
    )

    return deploy_response


def undeploy_pipeline(pipeline_data: dict):
    undeploy_response = client.post(f"/undeploy/{pipeline_data['name']}")
    return undeploy_response


@pytest.mark.parametrize("pipeline_data", pipeline_data)
def test_deploy_pipeline_def(pipeline_data: dict):
    deploy_response = deploy_pipeline(pipeline_data)
    assert deploy_response.status_code == 200

    status_response = client.get("/status")
    assert pipeline_data["name"] in status_response.json()["pipelines"]

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200


def test_undeploy_pipeline_def():
    pipeline_file = Path(__file__).parent / "test_files" / "working_pipelines/test_pipeline_01.yml"
    pipeline_data = {"name": pipeline_file.stem, "source_code": pipeline_file.read_text()}

    deploy_response = deploy_pipeline(pipeline_data)
    assert deploy_response.status_code == 200

    undeploy_response = undeploy_pipeline(pipeline_data)
    assert undeploy_response.status_code == 200

    status_response = client.get("/status")
    assert pipeline_data["name"] not in status_response.json()["pipelines"]
