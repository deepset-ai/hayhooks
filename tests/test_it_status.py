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


def test_typed_status_distinguishes_loaded_unknown_and_pipeline(client):
    from haystack import Pipeline
    from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

    class Wrapper(BasePipelineWrapper):
        def setup(self):
            pass

    custom = Wrapper()
    custom._is_run_chat_completion_implemented = True
    registry.add("custom_chat", custom)
    pipeline = Wrapper()
    pipeline.pipeline = Pipeline()
    pipeline._is_run_api_implemented = True
    registry.add("rag", pipeline)
    with client:
        body = client.get("/status").json()
    rows = {row["name"]: row for row in body["deployments"]}
    assert rows["custom_chat"]["type"] == "unknown"
    assert rows["rag"]["type"] == "pipeline"
    assert {row["state"] for row in rows.values()} == {"Loaded"}
    assert {row["execution_mode"] for row in rows.values()} == {"request-bound"}
    assert "chat_completion" in rows["custom_chat"]["capabilities"]
    assert not body["capabilities"]["execution_enumeration"]


def test_invalid_type_hint_is_rejected():
    from hayhooks.server.pipelines.models import create_pipeline_metadata
    from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

    class Wrapper(BasePipelineWrapper):
        deployment_type = "made-up"
        def setup(self):
            pass

    with pytest.raises(ValueError, match="deployment_type"):
        create_pipeline_metadata("custom", Wrapper())
