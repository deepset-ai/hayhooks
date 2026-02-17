"""Unit tests for YAMLPipelineWrapper module."""

import inspect
from pathlib import Path
from typing import Any

import pytest
from haystack import AsyncPipeline
from haystack.core.errors import PipelineError

from hayhooks.server.exceptions import InvalidYamlIOError
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.yaml_pipeline_wrapper import (
    YAMLPipelineWrapper,
    _create_dynamic_run_api_async,
    _map_flat_inputs_to_components,
    _set_method_signature,
)
from hayhooks.server.utils.yaml_utils import InputResolution, OutputResolution


@pytest.fixture
def sample_calc_yaml() -> str:
    path = Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml"
    return path.read_text()


@pytest.fixture
def sample_calc_path() -> Path:
    return Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml"


@pytest.fixture
def list_input_yaml() -> str:
    path = Path(__file__).parent / "test_files/yaml/list_input.yml"
    return path.read_text()


@pytest.fixture
def streaming_yaml() -> str:
    path = Path(__file__).parent / "test_files/yaml/multi_llm_streaming_pipeline.yml"
    return path.read_text()


def test_set_method_signature_sets_params_and_return_type():
    def dummy_func(self, **kwargs):
        pass

    params = [
        inspect.Parameter("name", inspect.Parameter.KEYWORD_ONLY, annotation=str),
        inspect.Parameter("age", inspect.Parameter.KEYWORD_ONLY, annotation=int, default=0),
    ]

    _set_method_signature(dummy_func, params, return_annotation=dict)

    sig = inspect.signature(dummy_func)
    assert "self" in sig.parameters
    assert "name" in sig.parameters
    assert sig.parameters["name"].annotation is str
    assert sig.parameters["age"].annotation is int
    assert sig.parameters["age"].default == 0
    assert sig.return_annotation is dict


def test_set_method_signature_self_removed_when_bound():
    class MyClass:
        def method(self, **kwargs):
            pass

    params = [inspect.Parameter("value", inspect.Parameter.KEYWORD_ONLY, annotation=int)]
    _set_method_signature(MyClass.method, params, return_annotation=dict)

    instance = MyClass()
    bound_sig = inspect.signature(instance.method)
    assert "self" not in bound_sig.parameters
    assert "value" in bound_sig.parameters


def test_map_flat_inputs_single_target():
    input_resolutions = {
        "query": InputResolution(
            path="prompt.query",
            component="prompt",
            name="query",
            type=str,
            required=True,
            targets=["prompt.query"],
        ),
    }

    result = _map_flat_inputs_to_components({"query": "hello"}, input_resolutions)

    assert result == {"prompt": {"query": "hello"}}


def test_map_flat_inputs_multi_target():
    input_resolutions = {
        "query": InputResolution(
            path="prompt.query",
            component="prompt",
            name="query",
            type=str,
            required=True,
            targets=["prompt.query", "builder.query"],
        ),
    }

    result = _map_flat_inputs_to_components({"query": "test"}, input_resolutions)

    assert result == {"prompt": {"query": "test"}, "builder": {"query": "test"}}


def test_map_flat_inputs_empty_returns_original():
    assert _map_flat_inputs_to_components({}, {}) == {}
    assert _map_flat_inputs_to_components({"x": 1}, {}) == {"x": 1}


def test_map_flat_inputs_unresolved_falls_back():
    input_resolutions = {
        "query": InputResolution(
            path="prompt.query", component="prompt", name="query", type=str, required=True, targets=["prompt.query"]
        ),
    }
    flat_inputs = {"query": "test", "unknown": "value"}

    result = _map_flat_inputs_to_components(flat_inputs, input_resolutions)

    assert result == flat_inputs


def test_create_dynamic_run_api_async_signature():
    input_resolutions = {
        "query": InputResolution(
            path="prompt.query", component="prompt", name="query", type=str, required=True, targets=["prompt.query"]
        ),
        "limit": InputResolution(
            path="retriever.top_k",
            component="retriever",
            name="top_k",
            type=int,
            required=False,
            targets=["retriever.top_k"],
        ),
    }

    func = _create_dynamic_run_api_async(input_resolutions, include_outputs_from=None)

    sig = inspect.signature(func)
    assert sig.parameters["query"].annotation is str
    assert sig.parameters["limit"].annotation is int
    assert sig.parameters["query"].default is inspect.Parameter.empty
    assert sig.parameters["limit"].default is None
    assert sig.return_annotation is dict
    assert inspect.iscoroutinefunction(func)


def test_create_dynamic_run_api_async_uses_any_when_type_none():
    input_resolutions = {
        "data": InputResolution(
            path="comp.data", component="comp", name="data", type=None, required=True, targets=["comp.data"]
        ),
    }

    func = _create_dynamic_run_api_async(input_resolutions, include_outputs_from=None)

    sig = inspect.signature(func)
    assert sig.parameters["data"].annotation is Any


def test_from_yaml_creates_wrapper(sample_calc_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml)

    assert isinstance(wrapper, YAMLPipelineWrapper)
    assert isinstance(wrapper, BasePipelineWrapper)
    assert wrapper._is_run_api_async_implemented is True


def test_from_yaml_parses_inputs_outputs(sample_calc_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml)

    assert "value" in wrapper.input_resolutions
    assert wrapper.input_resolutions["value"].type is int
    assert "result" in wrapper.output_resolutions
    assert wrapper.include_outputs_from == {"double"}


def test_from_yaml_parses_streaming_components(streaming_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(streaming_yaml)

    assert wrapper.streaming_components == ["llm_1", "llm_2"]


def test_from_yaml_no_streaming_is_none(sample_calc_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml)

    assert wrapper.streaming_components is None


def test_from_yaml_sets_description(sample_calc_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml, description="My pipeline")

    assert wrapper.description == "My pipeline"


def test_from_yaml_raises_on_missing_inputs_outputs():
    yaml_without_io = """
components:
  adder:
    type: haystack.testing.sample_components.add_value.AddFixedValue
    init_parameters:
      add: 1
connections: []
metadata: {}
"""
    with pytest.raises(InvalidYamlIOError):
        YAMLPipelineWrapper.from_yaml(yaml_without_io)


def test_from_yaml_multi_target_inputs(list_input_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(list_input_yaml)

    targets = wrapper.input_resolutions["query"].targets
    assert len(targets) == 2
    assert "chat_summary_prompt_builder.query" in targets
    assert "answer_builder.query" in targets


def test_from_yaml_raises_on_invalid_component():
    invalid_yaml = """
components:
  invalid:
    type: nonexistent.module.Component
    init_parameters: {}
connections: []
metadata: {}
inputs:
  value: invalid.value
outputs:
  result: invalid.result
"""
    with pytest.raises(PipelineError, match="not imported"):
        YAMLPipelineWrapper.from_yaml(invalid_yaml)


def test_from_file_creates_wrapper(sample_calc_path):
    wrapper = YAMLPipelineWrapper.from_file(sample_calc_path)

    assert isinstance(wrapper, YAMLPipelineWrapper)
    assert "value" in wrapper.input_resolutions


def test_from_file_accepts_string_path(sample_calc_path):
    wrapper = YAMLPipelineWrapper.from_file(str(sample_calc_path))

    assert isinstance(wrapper, YAMLPipelineWrapper)


def test_from_file_raises_on_missing_file():
    with pytest.raises(FileNotFoundError, match="YAML pipeline file not found"):
        YAMLPipelineWrapper.from_file("/nonexistent/path/pipeline.yml")


def test_setup_loads_async_pipeline(sample_calc_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml)
    wrapper.setup()

    assert isinstance(wrapper.pipeline, AsyncPipeline)


def test_setup_is_idempotent(sample_calc_yaml):
    """Test that calling setup twice does not reload the pipeline."""
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml)
    wrapper.setup()

    # Store reference to first pipeline instance
    first_pipeline = wrapper.pipeline
    assert isinstance(first_pipeline, AsyncPipeline)

    # Call setup again
    wrapper.setup()

    # Should be the exact same instance (not reloaded)
    assert wrapper.pipeline is first_pipeline


def test_run_api_async_has_correct_signature(sample_calc_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml)

    sig = inspect.signature(wrapper.run_api_async)

    assert "self" not in sig.parameters
    assert "value" in sig.parameters
    assert sig.parameters["value"].annotation is int


@pytest.mark.asyncio
async def test_run_api_async_executes_pipeline(sample_calc_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml)
    wrapper.setup()

    # sample_calc_pipeline: adds 2, then doubles -> 5+2=7 -> 7*2=14
    result = await wrapper.run_api_async(value=5)

    assert result == {"double": {"value": 14}}


@pytest.mark.asyncio
async def test_run_api_async_with_different_input(sample_calc_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml)
    wrapper.setup()

    # 10+2=12 -> 12*2=24
    result = await wrapper.run_api_async(value=10)

    assert result == {"double": {"value": 24}}


def test_properties_return_correct_types(sample_calc_yaml):
    wrapper = YAMLPipelineWrapper.from_yaml(sample_calc_yaml, description="Test")

    assert isinstance(wrapper.input_resolutions, dict)
    assert isinstance(wrapper.input_resolutions["value"], InputResolution)
    assert isinstance(wrapper.output_resolutions, dict)
    assert isinstance(wrapper.output_resolutions["result"], OutputResolution)
    assert isinstance(wrapper.include_outputs_from, set)
    assert wrapper.description == "Test"
