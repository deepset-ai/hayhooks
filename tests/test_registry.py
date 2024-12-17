import pytest
from pathlib import Path
from haystack import Pipeline
from haystack.core.errors import PipelineError

from hayhooks.server.pipelines.registry import _PipelineRegistry


@pytest.fixture
def pipeline_registry():
    return _PipelineRegistry()


@pytest.fixture
def sample_pipeline_yaml():
    return (Path(__file__).parent / "test_files" / "working_pipelines" / "basic_rag_pipeline.yml").read_text()


def test_add_pipeline(pipeline_registry, sample_pipeline_yaml):
    result = pipeline_registry.add("test_pipeline", sample_pipeline_yaml)

    expected_pipeline = Pipeline.loads(sample_pipeline_yaml)
    assert result == expected_pipeline
    assert pipeline_registry.get("test_pipeline") == expected_pipeline


def test_add_duplicate_pipeline(pipeline_registry, sample_pipeline_yaml):
    result = pipeline_registry.add("test_pipeline", sample_pipeline_yaml)

    expected_pipeline = Pipeline.loads(sample_pipeline_yaml)
    assert result == expected_pipeline

    with pytest.raises(ValueError, match="A pipeline with name test_pipeline is already in the registry"):
        pipeline_registry.add("test_pipeline", sample_pipeline_yaml)


def test_add_invalid_pipeline(pipeline_registry, mocker):
    mocker.patch('haystack.Pipeline.loads', side_effect=PipelineError("Invalid pipeline"))

    with pytest.raises(ValueError, match="Unable to parse Haystack Pipeline test_pipeline"):
        pipeline_registry.add("test_pipeline", "invalid yaml")


def test_remove_pipeline(pipeline_registry, sample_pipeline_yaml):
    pipeline_registry.add("test_pipeline", sample_pipeline_yaml)
    pipeline_registry.remove("test_pipeline")

    assert pipeline_registry.get("test_pipeline") is None


def test_remove_nonexistent_pipeline(pipeline_registry):
    pipeline_registry.remove("nonexistent_pipeline")


def test_get_nonexistent_pipeline(pipeline_registry):
    assert pipeline_registry.get("nonexistent_pipeline") is None


def test_get_names(pipeline_registry, sample_pipeline_yaml, mocker):
    mocker.patch('haystack.Pipeline.loads', return_value=mocker.Mock(spec=Pipeline))

    pipeline_registry.add("pipeline1", sample_pipeline_yaml)
    pipeline_registry.add("pipeline2", sample_pipeline_yaml)

    names = pipeline_registry.get_names()
    assert sorted(names) == ["pipeline1", "pipeline2"]
