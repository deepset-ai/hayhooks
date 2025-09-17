from pathlib import Path

import pytest
from haystack import Document, Pipeline

from hayhooks.server.exceptions import PipelineNotFoundError
from hayhooks.server.pipelines.registry import _PipelineRegistry
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


@pytest.fixture
def pipeline_registry():
    return _PipelineRegistry()


@pytest.fixture
def sample_pipeline_yaml():
    return (Path(__file__).parent / "test_files/yaml" / "inputs_outputs_pipeline.yml").read_text()


@pytest.fixture
def test_pipeline_wrapper_class():
    class TestPipelineWrapper(BasePipelineWrapper):
        def setup(self) -> None:
            pass

        def run_api(self, urls: list[str], question: str) -> dict:
            return {}

        def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> dict:
            return {}

    return TestPipelineWrapper


def test_add_yaml_pipeline(pipeline_registry, sample_pipeline_yaml):
    result = pipeline_registry.add("test_pipeline", sample_pipeline_yaml)

    pipeline = Pipeline.loads(sample_pipeline_yaml)
    assert result == pipeline
    assert pipeline_registry.get("test_pipeline") == pipeline


def test_get_non_existent_pipeline(pipeline_registry):
    pipeline = pipeline_registry.get("test_pipeline")
    assert pipeline is None


def test_remove_pipeline(pipeline_registry, sample_pipeline_yaml):
    pipeline_registry.add("test_pipeline", sample_pipeline_yaml)
    pipeline_registry.remove("test_pipeline")
    assert pipeline_registry.get("test_pipeline") is None


def test_remove_nonexistent_pipeline(pipeline_registry):
    pipeline_registry.remove("nonexistent_pipeline")


def test_get_nonexistent_pipeline(pipeline_registry):
    result = pipeline_registry.get("nonexistent_pipeline")
    assert result is None


def test_get_names(pipeline_registry, sample_pipeline_yaml, mocker):
    mocker.patch("haystack.Pipeline.loads", return_value=mocker.Mock(spec=Pipeline))

    pipeline_registry.add("pipeline1", sample_pipeline_yaml)
    pipeline_registry.add("pipeline2", sample_pipeline_yaml)

    names = pipeline_registry.get_names()
    assert sorted(names) == ["pipeline1", "pipeline2"]


def test_add_duplicate_pipeline_instance(pipeline_registry):
    pipeline = Pipeline()
    pipeline_registry.add("test_pipeline", pipeline)

    with pytest.raises(ValueError, match="A pipeline with name test_pipeline is already in the registry"):
        pipeline_registry.add("test_pipeline", pipeline)


def test_add_pipeline_wrapper(pipeline_registry, test_pipeline_wrapper_class):
    wrapper = test_pipeline_wrapper_class()
    result = pipeline_registry.add("test_wrapper", wrapper)

    assert result == wrapper
    assert pipeline_registry.get("test_wrapper") == wrapper


def test_add_invalid_pipeline_or_wrapper(pipeline_registry):
    not_a_pipeline_or_wrapper = Document()
    with pytest.raises(ValueError, match="Unable to parse Haystack Pipeline test_wrapper"):
        pipeline_registry.add("test_wrapper", not_a_pipeline_or_wrapper)


def test_add_duplicate_pipeline_wrapper(pipeline_registry, test_pipeline_wrapper_class):
    wrapper = test_pipeline_wrapper_class()
    pipeline_registry.add("test_wrapper", wrapper)

    with pytest.raises(ValueError, match="A pipeline with name test_wrapper is already in the registry"):
        pipeline_registry.add("test_wrapper", wrapper)


def test_get_pipeline_wrapper(pipeline_registry, test_pipeline_wrapper_class):
    class TestPipelineWrapperWithPipeline(test_pipeline_wrapper_class):
        def setup(self) -> None:
            self.pipeline = Pipeline()

    wrapper = TestPipelineWrapperWithPipeline()
    wrapper.setup()
    pipeline_registry.add("test_wrapper", wrapper)

    assert pipeline_registry.get("test_wrapper") == wrapper


def test_clear_registry(pipeline_registry, sample_pipeline_yaml, test_pipeline_wrapper_class):
    pipeline_registry.add("pipeline1", sample_pipeline_yaml)
    wrapper = test_pipeline_wrapper_class()
    pipeline_registry.add("wrapper1", wrapper)

    pipeline_registry.clear()
    assert len(pipeline_registry.get_names()) == 0
    assert pipeline_registry.get("pipeline1") is None
    assert pipeline_registry.get("wrapper1") is None


def test_remove_pipeline_preserves_others(pipeline_registry, sample_pipeline_yaml, test_pipeline_wrapper_class):
    pipeline_registry.add("pipeline1", sample_pipeline_yaml)
    wrapper = test_pipeline_wrapper_class()
    pipeline_registry.add("wrapper1", wrapper)

    pipeline_registry.remove("pipeline1")
    assert "pipeline1" not in pipeline_registry.get_names()
    assert pipeline_registry.get("wrapper1") == wrapper


def test_add_pipeline_with_metadata(pipeline_registry, sample_pipeline_yaml):
    metadata = {"description": "Test RAG pipeline"}
    result = pipeline_registry.add("test_pipeline", sample_pipeline_yaml, metadata=metadata)

    pipeline = Pipeline.loads(sample_pipeline_yaml)
    assert result == pipeline
    assert pipeline_registry.get("test_pipeline") == pipeline
    assert pipeline_registry.get_metadata("test_pipeline") == metadata


def test_add_pipeline_wrapper_with_metadata(pipeline_registry, test_pipeline_wrapper_class):
    metadata = {"description": "Test pipeline wrapper"}
    wrapper = test_pipeline_wrapper_class()
    result = pipeline_registry.add("test_wrapper", wrapper, metadata=metadata)

    assert result == wrapper
    assert pipeline_registry.get("test_wrapper") == wrapper
    assert pipeline_registry.get_metadata("test_wrapper") == metadata


def test_get_pipeline_with_metadata(pipeline_registry, sample_pipeline_yaml):
    metadata = {"description": "Test pipeline"}
    pipeline_registry.add("test_pipeline", sample_pipeline_yaml, metadata=metadata)

    pipeline = pipeline_registry.get("test_pipeline")
    meta = pipeline_registry.get_metadata("test_pipeline")
    assert isinstance(pipeline, Pipeline)
    assert meta == metadata


def test_get_metadata_for_nonexistent_pipeline(pipeline_registry):
    assert pipeline_registry.get_metadata("nonexistent_pipeline") is None


def test_remove_pipeline_removes_metadata(pipeline_registry, sample_pipeline_yaml):
    metadata = {"description": "Test pipeline"}
    pipeline_registry.add("test_pipeline", sample_pipeline_yaml, metadata=metadata)

    pipeline_registry.remove("test_pipeline")
    assert pipeline_registry.get_metadata("test_pipeline") is None


def test_clear_registry_removes_metadata(pipeline_registry, sample_pipeline_yaml):
    metadata = {"description": "Test pipeline"}
    pipeline_registry.add("test_pipeline", sample_pipeline_yaml, metadata=metadata)

    pipeline_registry.clear()
    assert pipeline_registry.get_metadata("test_pipeline") is None


def test_update_metadata(pipeline_registry, sample_pipeline_yaml):
    initial_metadata = {"description": "Initial description"}
    pipeline_registry.add("test_pipeline", sample_pipeline_yaml, metadata=initial_metadata)

    new_metadata = {"version": "1.0.0"}
    pipeline_registry.update_metadata("test_pipeline", new_metadata)

    updated_metadata = pipeline_registry.get_metadata("test_pipeline")
    assert updated_metadata == {"description": "Initial description", "version": "1.0.0"}


def test_update_metadata_nonexistent_pipeline(pipeline_registry):
    with pytest.raises(PipelineNotFoundError, match="Pipeline nonexistent_pipeline not found in registry"):
        pipeline_registry.update_metadata("nonexistent_pipeline", {"version": "1.0.0"})
