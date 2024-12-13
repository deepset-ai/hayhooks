from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, create_model
from hayhooks.server.utils.create_valid_type import handle_unsupported_types
from haystack import Document


class PipelineDefinition(BaseModel):
    name: str
    source_code: str


DEFAULT_TYPES_MAPPING = {
    DataFrame: dict,
    Document: dict,
}


def get_request_model(pipeline_name: str, pipeline_inputs):
    """
    Inputs have this form:
    {
        'first_addition': { <-- Component Name
            'value': {'type': <class 'int'>, 'is_mandatory': True}, <-- Input
            'add': {'type': typing.Optional[int], 'is_mandatory': False, 'default_value': None}, <-- Input
        },
        'second_addition': {'add': {'type': typing.Optional[int], 'is_mandatory': False}},
    }
    """
    request_model = {}
    config = ConfigDict(arbitrary_types_allowed=True)

    for component_name, inputs in pipeline_inputs.items():
        component_model = {}
        for name, typedef in inputs.items():
            try:
                input_type = handle_unsupported_types(type_=typedef["type"], types_mapping=DEFAULT_TYPES_MAPPING)
            except TypeError as e:
                print(f"ERROR at {component_name!r}, {name}: {typedef}")
                raise e

            if input_type is not None:
                component_model[name] = (
                    input_type,
                    typedef.get("default_value", ...),
                )
        request_model[component_name] = (create_model("ComponentParams", **component_model, __config__=config), ...)

    return create_model(f"{pipeline_name.capitalize()}RunRequest", **request_model, __config__=config)


def get_response_model(pipeline_name: str, pipeline_outputs):
    """
    Outputs have this form:
    {
        'second_addition': { <-- Component Name
            'result': {'type': "<class 'int'>"}  <-- Output
        },
    }
    """
    response_model = {}
    config = ConfigDict(arbitrary_types_allowed=True)
    for component_name, outputs in pipeline_outputs.items():
        component_model = {}
        for name, typedef in outputs.items():
            output_type = typedef["type"]
            component_model[name] = (
                handle_unsupported_types(type_=output_type, types_mapping=DEFAULT_TYPES_MAPPING),
                ...,
            )
        response_model[component_name] = (create_model("ComponentParams", **component_model, __config__=config), ...)

    return create_model(f"{pipeline_name.capitalize()}RunResponse", **response_model, __config__=config)


def convert_value_to_dict(value):
    """Convert a single value to a dictionary if possible"""
    if hasattr(value, "to_dict"):
        if "init_parameters" in value.to_dict():
            return value.to_dict()["init_parameters"]
        return value.to_dict()
    elif hasattr(value, "model_dump"):
        return value.model_dump()
    elif isinstance(value, dict):
        return {k: convert_value_to_dict(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_value_to_dict(item) for item in value]
    else:
        return value


def convert_component_output(component_output):
    """
    Converts component outputs to dictionaries that can be validated against response model.
    Handles nested structures recursively.

    Args:
        component_output: Dict with component outputs

    Returns:
        Dict with all nested objects converted to dictionaries
    """
    if isinstance(component_output, dict):
        return {name: convert_value_to_dict(data) for name, data in component_output.items()}

    return convert_value_to_dict(component_output)
