from pathlib import Path

import pytest
from haystack import AsyncPipeline

from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.deploy_utils import deploy_pipeline_yaml
from hayhooks.server.utils.yaml_pipeline_wrapper import YAMLPipelineWrapper


@pytest.fixture(autouse=True)
def cleanup_test_pipelines():
    yield
    for pipeline_name in registry.get_names():
        registry.remove(pipeline_name)


def test_deploy_pipeline_with_inputs_outputs():
    # Use sample_calc_pipeline which doesn't require optional dependencies
    pipeline_file = Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml"
    pipeline_data = {
        "name": pipeline_file.stem,
        "source_code": pipeline_file.read_text(),
    }

    deploy_pipeline_yaml(
        pipeline_name=pipeline_data["name"],
        source_code=pipeline_data["source_code"],
        options={"save_file": False},
    )

    wrapper = registry.get(pipeline_data["name"])
    assert wrapper is not None
    assert isinstance(wrapper, YAMLPipelineWrapper)

    metadata = registry.get_metadata(pipeline_data["name"])
    assert metadata is not None

    # Verify input_resolutions metadata exists
    assert "input_resolutions" in metadata
    assert "value" in metadata["input_resolutions"]


def test_yaml_pipeline_wrapper_has_async_pipeline():
    # Use sample_calc_pipeline which doesn't require optional dependencies
    pipeline_file = Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml"
    pipeline_name = pipeline_file.stem
    source_code = pipeline_file.read_text()

    deploy_pipeline_yaml(pipeline_name=pipeline_name, source_code=source_code, options={"save_file": False})

    wrapper = registry.get(pipeline_name)
    assert isinstance(wrapper, YAMLPipelineWrapper)
    # The internal pipeline should be an AsyncPipeline
    assert isinstance(wrapper.pipeline, AsyncPipeline)


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

    deploy_pipeline_yaml(
        pipeline_name=pipeline_data["name"],
        source_code=pipeline_data["source_code"],
        options={"save_file": False},
    )

    wrapper = registry.get(pipeline_data["name"])
    assert wrapper is not None
    assert isinstance(wrapper, YAMLPipelineWrapper)

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

    deploy_pipeline_yaml(
        pipeline_name=pipeline_data["name"],
        source_code=pipeline_data["source_code"],
        options={"save_file": False},
    )

    # Verify pipeline was added to registry
    wrapper = registry.get(pipeline_data["name"])
    assert wrapper is not None
    assert isinstance(wrapper, YAMLPipelineWrapper)

    # Verify metadata contains streaming_components configuration
    metadata = registry.get_metadata(pipeline_data["name"])
    assert metadata is not None
    assert "streaming_components" in metadata
    assert metadata["streaming_components"] is not None
    assert metadata["streaming_components"] == ["llm_1", "llm_2"]

    # Verify the wrapper has an AsyncPipeline internally
    assert isinstance(wrapper.pipeline, AsyncPipeline)


def test_deploy_yaml_pipeline_without_streaming_components():
    """Test that pipelines without streaming_components field have None in metadata."""
    # Use sample_calc_pipeline which doesn't require optional dependencies
    pipeline_file = Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml"
    pipeline_name = pipeline_file.stem
    source_code = pipeline_file.read_text()

    # Verify streaming_components is NOT in this YAML
    assert "streaming_components:" not in source_code

    deploy_pipeline_yaml(pipeline_name=pipeline_name, source_code=source_code, options={"save_file": False})

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

    deploy_pipeline_yaml(
        pipeline_name=pipeline_data["name"],
        source_code=pipeline_data["source_code"],
        options={"save_file": False},
    )

    # Verify pipeline was added to registry
    wrapper = registry.get(pipeline_data["name"])
    assert wrapper is not None
    assert isinstance(wrapper, YAMLPipelineWrapper)

    # Verify metadata contains streaming_components as "all"
    metadata = registry.get_metadata(pipeline_data["name"])
    assert metadata is not None
    assert "streaming_components" in metadata
    assert metadata["streaming_components"] == "all"

    # Verify the wrapper has an AsyncPipeline internally
    assert isinstance(wrapper.pipeline, AsyncPipeline)
