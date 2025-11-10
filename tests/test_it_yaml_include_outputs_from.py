from pathlib import Path
from unittest.mock import AsyncMock, patch

from hayhooks.server.pipelines import registry

MULTI_OUTPUT_PIPELINE_PATH = Path(__file__).parent / "test_files" / "yaml" / "multi_output_pipeline.yml"
LIST_INPUT_PIPELINE_PATH = Path(__file__).parent / "test_files" / "yaml" / "list_input.yml"


def test_yaml_pipeline_auto_derives_include_outputs_from(client, deploy_yaml_pipeline, undeploy_pipeline):
    yaml_source = MULTI_OUTPUT_PIPELINE_PATH.read_text()

    response = deploy_yaml_pipeline(client, "test_auto_include", yaml_source)
    assert response.status_code == 200

    metadata = registry.get_metadata("test_auto_include")
    assert metadata["include_outputs_from"] == {"double", "second_addition"}

    undeploy_pipeline(client, "test_auto_include")


def test_yaml_pipeline_passes_include_outputs_from_to_haystack(client, deploy_yaml_pipeline, undeploy_pipeline):
    yaml_source = MULTI_OUTPUT_PIPELINE_PATH.read_text()

    response = deploy_yaml_pipeline(client, "test_run_include", yaml_source)
    assert response.status_code == 200

    pipeline = registry.get("test_run_include")
    mock_run_async = AsyncMock(return_value={"double": {"value": 10}, "second_addition": {"result": 15}})

    with patch.object(pipeline, "run_async", mock_run_async):
        response = client.post("/test_run_include/run", json={"value": 3})
        assert response.status_code == 200

        mock_run_async.assert_called_once()
        call_kwargs = mock_run_async.call_args.kwargs
        assert call_kwargs["include_outputs_from"] == {"double", "second_addition"}

    undeploy_pipeline(client, "test_run_include")


def test_yaml_pipeline_supports_list_declared_inputs(client, deploy_yaml_pipeline, undeploy_pipeline):
    yaml_source = LIST_INPUT_PIPELINE_PATH.read_text()

    response = deploy_yaml_pipeline(client, "test_list_inputs", yaml_source)
    assert response.status_code == 200

    metadata = registry.get_metadata("test_list_inputs")
    assert metadata["include_outputs_from"] == {"answer_builder"}

    request_model = metadata["request_model"]
    assert "query" in request_model.model_fields

    undeploy_pipeline(client, "test_list_inputs")
