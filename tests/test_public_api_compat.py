"""Public import paths and messages that downstream runtimes depend on."""

import inspect

import pytest

import hayhooks
from hayhooks.server.pipelines.utils import is_streaming_component
from hayhooks.server.utils.yaml_utils import get_streaming_components_from_yaml


def test_downstream_runtime_imports_remain_public() -> None:
    assert inspect.isfunction(hayhooks.streaming_generator)
    assert inspect.isfunction(hayhooks.async_streaming_generator)
    assert inspect.isfunction(hayhooks.coerce_pipeline_inputs)
    assert inspect.isclass(hayhooks.Pipeline)
    assert callable(is_streaming_component)
    assert callable(get_streaming_components_from_yaml)


def test_missing_streaming_component_error_text_is_stable() -> None:
    with pytest.raises(ValueError) as error:
        list(hayhooks.streaming_generator(hayhooks.Pipeline()))

    assert str(error.value) == "No streaming-capable components found in the pipeline"
