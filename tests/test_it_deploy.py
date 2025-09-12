from pathlib import Path

import pytest

from hayhooks.server.pipelines.registry import registry


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()
    yield


# Load pipeline definitions from test_files
test_files = Path(__file__).parent / "test_files/yaml"
pipeline_data = [{"name": file.stem, "source_code": file.read_text()} for file in test_files.glob("*.yml")]


@pytest.mark.parametrize("pipeline_data", pipeline_data)
def test_deploy_yaml_pipeline(client, deploy_yaml_pipeline, status_pipeline, pipeline_data: dict):
    deploy_response = deploy_yaml_pipeline(client, pipeline_data["name"], pipeline_data["source_code"])
    assert deploy_response.status_code == 200

    status_response = status_pipeline(client, pipeline_data["name"])
    assert pipeline_data["name"] in status_response.json()["pipeline"]

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200


def test_undeploy_yaml_pipeline(client, deploy_yaml_pipeline, undeploy_pipeline, status_pipeline):
    pipeline_file = Path(__file__).parent / "test_files/yaml" / "inputs_outputs_pipeline.yml"
    pipeline_data = {"name": pipeline_file.stem, "source_code": pipeline_file.read_text()}

    deploy_response = deploy_yaml_pipeline(client, pipeline_data["name"], pipeline_data["source_code"])
    assert deploy_response.status_code == 200

    undeploy_response = undeploy_pipeline(client, pipeline_data["name"])
    assert undeploy_response.status_code == 200

    status_response = status_pipeline(client, pipeline_data["name"])
    assert status_response.status_code == 404


def test_undeploy_non_existent_yaml_pipeline(client, undeploy_pipeline):
    undeploy_response = undeploy_pipeline(client, "non_existent_pipeline")
    assert undeploy_response.status_code == 404


def test_undeploy_no_yaml_pipelines(client, undeploy_pipeline):
    undeploy_response = undeploy_pipeline(client, "non_existent_pipeline")
    assert undeploy_response.status_code == 404
