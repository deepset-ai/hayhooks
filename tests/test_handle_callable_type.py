import typing
import haystack
from types import NoneType
from hayhooks.server.pipelines.models import get_request_model


def test_handle_callable_type_when_creating_pipeline_models():
    pipeline_name = "test_pipeline"
    pipeline_inputs = {
        'generator': {
            'system_prompt': {'type': typing.Optional[str], 'is_mandatory': False, 'default_value': None},
            'streaming_callback': {
                'type': typing.Optional[
                    typing.Callable[[haystack.dataclasses.streaming_chunk.StreamingChunk], NoneType]
                ],
                'is_mandatory': False,
                'default_value': None,
            },
            'generation_kwargs': {
                'type': typing.Optional[typing.Dict[str, typing.Any]],
                'is_mandatory': False,
                'default_value': None,
            },
        }
    }

    request_model = get_request_model(pipeline_name, pipeline_inputs)

    # This line used to throw an error because the Callable type was not handled correctly
    # by the handle_unsupported_types function
    assert request_model.model_json_schema() is not None
    assert request_model.__name__ == "Test_pipelineRunRequest"
