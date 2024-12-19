from collections.abc import Callable as CallableABC
from typing import Any, Callable, Optional, Union

import haystack
import pytest

from hayhooks.server.pipelines.models import get_request_model
from hayhooks.server.utils.create_valid_type import is_callable_type


@pytest.mark.parametrize(
    "t, expected",
    [
        (Callable, True),
        (CallableABC, True),
        (Callable[[int], str], True),
        (Callable[..., Any], True),
        (int, False),
        (str, False),
        (Any, False),
        (Union[int, str], False),
        (Optional[Callable[[haystack.dataclasses.streaming_chunk.StreamingChunk], type(None)]], True),
    ],
)
def test_is_callable_type(t, expected):
    assert is_callable_type(t) == expected


def test_skip_callables_when_creating_pipeline_models():
    pipeline_name = "test_pipeline"
    pipeline_inputs = {
        "generator": {
            "system_prompt": {"type": Optional[str], "is_mandatory": False, "default_value": None},
            "streaming_callback": {
                "type": Optional[Callable[[haystack.dataclasses.streaming_chunk.StreamingChunk], type(None)]],
                "is_mandatory": False,
                "default_value": None,
            },
            "generation_kwargs": {
                "type": Optional[dict[str, Any]],
                "is_mandatory": False,
                "default_value": None,
            },
        }
    }

    request_model = get_request_model(pipeline_name, pipeline_inputs)

    # This line used to throw an error because the Callable type was not handled correctly
    # by the handle_unsupported_types function
    assert request_model.model_json_schema() is not None
    assert request_model.__name__ == "Test_pipelineRunRequest"
    assert "streaming_callback" not in request_model.model_json_schema()["$defs"]["ComponentParams"]["properties"]
