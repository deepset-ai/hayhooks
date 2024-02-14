from typing import get_args, get_origin, List

from pydantic import BaseModel, create_model
from haystack.dataclasses import Document


class HaystackDocument(BaseModel):
    id: str
    content: str


class PipelineDefinition(BaseModel):
    name: str
    source_code: str


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
    for component_name, inputs in pipeline_inputs.items():

        component_model = {}
        for name, typedef in inputs.items():
            component_model[name] = (typedef["type"], typedef.get("default_value", ...))
        request_model[component_name] = (create_model('ComponentParams', **component_model, __config__=config), ...)

    return create_model(f'{pipeline_name.capitalize()}RunRequest', **request_model, __config__=config)


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
    for component_name, outputs in pipe.outputs().items():
        component_model = {}
        for name, typedef in outputs.items():
            output_type = typedef["type"]
            if get_origin(output_type) == list and get_args(output_type)[0] == Document:
                component_model[name] = (List[HaystackDocument], ...)
            else:
                component_model[name] = (typedef["type"], ...)
        response_model[component_name] = (create_model('ComponentParams', **component_model, __config__=config), ...)

    return create_model(f'{pipeline_name.capitalize()}RunResponse', **response_model, __config__=config)


def convert_component_output(component_output):
    """
    Component output has this form:

    "documents":[
        {"id":"818170...", "content":"RapidAPI for Mac is a full-featured HTTP client."}
    ]

    We inspect the output and convert haystack.Document into the HaystackDocument pydantic model as needed
    """
    result = {}
    for output_name, data in component_output.items():
        # Empty containers, None values, empty strings and the likes: do nothing
        if not data:
            result[output_name] = data

        # Output contains a list of Document
        if type(data) is list and type(data[0]) is Document:
            result[output_name] = [HaystackDocument(id=d.id, content=d.content) for d in data]
        # Output is a single Document
        elif type(data) is Document:
            result[output_name] = HaystackDocument(id=data.id, content=data.content or "")
        # Any other type: do nothing
        else:
            result[output_name] = data

    return result
