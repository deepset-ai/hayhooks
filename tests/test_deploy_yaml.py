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


def test_deploy_yaml_pipeline_with_utf8_characters():
    pipeline_file = Path(__file__).parent / "test_files/yaml/utf8_pipeline.yml"
    pipeline_data = {
        "name": pipeline_file.stem,
        "source_code": pipeline_file.read_text(encoding="utf-8"),
    }

    # Verify UTF-8 characters are present in the source code
    assert "你好世界" in pipeline_data["source_code"]
    assert "🌍" in pipeline_data["source_code"]
    assert "こんにちは" in pipeline_data["source_code"]
    assert "мир" in pipeline_data["source_code"]

    add_yaml_pipeline_to_registry(
        pipeline_name=pipeline_data["name"],
        source_code=pipeline_data["source_code"],
    )

    assert registry.get(pipeline_data["name"]) is not None

    metadata = registry.get_metadata(pipeline_data["name"])
    assert metadata is not None


def test_deploy_yaml_pipeline_with_streaming_components():
    """Test that streaming_components field is properly parsed from YAML and stored in metadata."""
    pipeline_file = Path(__file__).parent / "test_files/yaml/multi_llm_streaming_pipeline.yml"
    pipeline_data = {
        "name": pipeline_file.stem,
        "source_code": pipeline_file.read_text(),
    }

    # Verify streaming_components is in the YAML
    assert "streaming_components:" in pipeline_data["source_code"]

    add_yaml_pipeline_to_registry(
        pipeline_name=pipeline_data["name"],
        source_code=pipeline_data["source_code"],
    )

    # Verify pipeline was added to registry
    assert registry.get(pipeline_data["name"]) is not None

    # Verify metadata contains streaming_components configuration
    metadata = registry.get_metadata(pipeline_data["name"])
    assert metadata is not None
    assert "streaming_components" in metadata
    assert metadata["streaming_components"] is not None
    assert metadata["streaming_components"] == ["llm_1", "llm_2"]

    # Verify it's an AsyncPipeline
    pipeline_instance = registry.get(pipeline_data["name"])
    assert isinstance(pipeline_instance, AsyncPipeline)


def test_deploy_yaml_pipeline_without_streaming_components():
    """Test that pipelines without streaming_components field have None in metadata."""
    pipeline_file = Path(__file__).parent / "test_files/yaml/inputs_outputs_pipeline.yml"
    pipeline_name = pipeline_file.stem
    source_code = pipeline_file.read_text()

    # Verify streaming_components is NOT in this YAML
    assert "streaming_components:" not in source_code

    add_yaml_pipeline_to_registry(pipeline_name=pipeline_name, source_code=source_code)

    # Verify metadata contains streaming_components as None (default behavior)
    metadata = registry.get_metadata(pipeline_name)
    assert metadata is not None
    assert "streaming_components" in metadata
    assert metadata["streaming_components"] is None


def test_deploy_yaml_pipeline_with_streaming_components_all_keyword():
    """Test that streaming_components: all is properly parsed and stored."""
    pipeline_file = Path(__file__).parent / "test_files/yaml/multi_llm_streaming_all_pipeline.yml"
    pipeline_data = {
        "name": pipeline_file.stem,
        "source_code": pipeline_file.read_text(),
    }

    # Verify streaming_components: all is in the YAML
    assert "streaming_components: all" in pipeline_data["source_code"]

    add_yaml_pipeline_to_registry(
        pipeline_name=pipeline_data["name"],
        source_code=pipeline_data["source_code"],
    )

    # Verify pipeline was added to registry
    assert registry.get(pipeline_data["name"]) is not None

    # Verify metadata contains streaming_components as "all"
    metadata = registry.get_metadata(pipeline_data["name"])
    assert metadata is not None
    assert "streaming_components" in metadata
    assert metadata["streaming_components"] == "all"

    # Verify it's an AsyncPipeline
    pipeline_instance = registry.get(pipeline_data["name"])
    assert isinstance(pipeline_instance, AsyncPipeline)
