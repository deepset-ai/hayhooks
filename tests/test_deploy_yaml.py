from pathlib import Path

import pytest
from haystack import AsyncPipeline

from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.deploy_utils import add_yaml_pipeline_to_registry


@pytest.fixture(autouse=True)
def cleanup_test_pipelines():
    yield
    for pipeline_name in registry.get_names():
        registry.remove(pipeline_name)


def test_deploy_pipeline_with_inputs_outputs():
    pipeline_file = Path(__file__).parent / "test_files/yaml/inputs_outputs_pipeline.yml"
    pipeline_data = {
        "name": pipeline_file.stem,
        "source_code": pipeline_file.read_text(),
    }

    add_yaml_pipeline_to_registry(
        pipeline_name=pipeline_data["name"],
        source_code=pipeline_data["source_code"],
    )

    assert registry.get(pipeline_data["name"]) is not None

    metadata = registry.get_metadata(pipeline_data["name"])
    assert metadata is not None
    assert metadata["request_model"] is not None
    assert metadata["response_model"] is not None

    assert metadata["request_model"].model_json_schema() == {
        "properties": {
            "urls": {
                "title": "Urls",
                "type": "array",
                "items": {"type": "string"},
            },
            "query": {
                "title": "Query",
            },
        },
        "required": ["urls", "query"],
        "title": "Inputs_outputs_pipelineRunRequest",
        "type": "object",
    }

    assert metadata["response_model"].model_json_schema() == {
        "properties": {
            "result": {
                "additionalProperties": True,
                "description": "Pipeline result",
                "type": "object",
                "title": "Result",
            },
        },
        "required": ["result"],
        "type": "object",
        "title": "Inputs_outputs_pipelineRunResponse",
    }


def test_yaml_pipeline_is_async_pipeline():
    pipeline_file = Path(__file__).parent / "test_files/yaml/inputs_outputs_pipeline.yml"
    pipeline_name = pipeline_file.stem
    source_code = pipeline_file.read_text()

    add_yaml_pipeline_to_registry(pipeline_name=pipeline_name, source_code=source_code)

    pipeline_instance = registry.get(pipeline_name)
    assert isinstance(pipeline_instance, AsyncPipeline)
