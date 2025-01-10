import pytest
from fastapi.testclient import TestClient
from hayhooks.server import app
from hayhooks.server.pipelines import registry
from tests.test_it_deploy import deploy_pipeline
from pathlib import Path

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()


def test_status_all_pipelines():
    status_response = client.get("/status")
    assert status_response.status_code == 200
    assert "pipelines" in status_response.json()


def test_status_single_pipeline():
    pipeline_file = Path(__file__).parent / "test_files" / "working_pipelines/test_pipeline_01.yml"
    pipeline_data = {"name": pipeline_file.stem, "source_code": pipeline_file.read_text()}

    deploy_pipeline(pipeline_data)

    status_response = client.get(f"/status", params={"pipeline_name": pipeline_data["name"]})
    assert status_response.status_code == 200
    assert status_response.json()["pipeline"] == pipeline_data["name"]


def test_status_non_existent_pipeline():
    status_response = client.get("/status", params={"pipeline_name": "non_existent_pipeline"})
    assert status_response.status_code == 404
    assert status_response.json()["detail"] == f"Pipeline 'non_existent_pipeline' not found"


def test_status_no_pipelines():
    status_response = client.get("/status")
    assert status_response.status_code == 200
    assert "pipelines" in status_response.json()
    assert len(status_response.json()["pipelines"]) == 0
