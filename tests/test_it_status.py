from pathlib import Path

import pytest

from hayhooks.server.pipelines import registry


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()


def test_status_all_pipelines(client, status_pipeline):
    status_response = status_pipeline(client, "")
    assert status_response.status_code == 200
    assert "pipelines" in status_response.json()


def test_status_single_pipeline(client, deploy_yaml_pipeline, status_pipeline):
    pipeline_file = Path(__file__).parent / "test_files/yaml" / "inputs_outputs_pipeline.yml"
    pipeline_data = {"name": pipeline_file.stem, "source_code": pipeline_file.read_text()}

    deploy_response = deploy_yaml_pipeline(client, pipeline_data["name"], pipeline_data["source_code"])
    assert deploy_response.status_code == 200

    status_response = status_pipeline(client, pipeline_data["name"])
    assert status_response.status_code == 200
    assert status_response.json()["pipeline"] == pipeline_data["name"]


def test_status_non_existent_pipeline(client, status_pipeline):
    status_response = status_pipeline(client, "non_existent_pipeline")
    assert status_response.status_code == 404
    assert status_response.json()["detail"] == "Pipeline 'non_existent_pipeline' not found"


def test_status_no_pipelines(client, status_pipeline):
    status_response = status_pipeline(client, "")
    assert status_response.status_code == 200
    assert "pipelines" in status_response.json()
    assert len(status_response.json()["pipelines"]) == 0
