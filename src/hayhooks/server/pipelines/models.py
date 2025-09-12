import inspect
from typing import Any, Callable

from docstring_parser.common import Docstring
from pydantic import BaseModel, Field, create_model

from hayhooks.server.exceptions import PipelineWrapperError
from hayhooks.server.utils.yaml_utils import InputResolution, OutputResolution


def get_request_model_from_resolved_io(
    pipeline_name: str, declared_inputs: dict[str, InputResolution]
) -> type[BaseModel]:
    """
    Create a flat Pydantic request model from declared inputs resolved by yaml_utils.

    Args:
        pipeline_name: Name of the pipeline used for model naming.
        declared_inputs: Mapping of declared input name to InputResolution.

    Returns:
        A Pydantic model with top-level fields matching declared input names.
    """
    fields: dict[str, Any] = {}

    for input_name, resolution in declared_inputs.items():
        input_type = resolution.type
        default_value = ... if resolution.required else None
        fields[input_name] = (input_type, default_value)

    return create_model(f"{pipeline_name.capitalize()}RunRequest", **fields)


def get_response_model_from_resolved_io(
    pipeline_name: str, declared_outputs: dict[str, OutputResolution]
) -> type[BaseModel]:
    """
    Create a flat Pydantic response model from declared outputs resolved by yaml_utils.

    Args:
        pipeline_name: Name of the pipeline used for model naming.
        declared_outputs: Mapping of declared output name to OutputResolution.

    Returns:
        A Pydantic model with top-level fields matching declared output names.
    """
    fields: dict[str, Any] = {}

    for output_name, resolution in declared_outputs.items():
        output_type = resolution.type
        fields[output_name] = (output_type, ...)

    return create_model(
        f"{pipeline_name.capitalize()}RunResponse", result=(dict, Field(..., description="Pipeline result"))
    )


def create_request_model_from_callable(func: Callable, model_name: str, docstring: Docstring) -> type[BaseModel]:
    """
    Create a dynamic Pydantic model based on callable's signature.

    Args:
        func: The callable (function/method) to analyze
        model_name: Name for the generated model

    Returns:
        Pydantic model class for request
    """

    params = inspect.signature(func).parameters
    param_docs = {p.arg_name: p.description for p in docstring.params}

    fields: dict[str, Any] = {}
    for name, param in params.items():
        default_value = ... if param.default == param.empty else param.default
        description = param_docs.get(name) or f"Parameter '{name}'"
        field_info = Field(default=default_value, description=description)
        fields[name] = (param.annotation, field_info)

    return create_model(f"{model_name}Request", **fields)


def create_response_model_from_callable(func: Callable, model_name: str, docstring: Docstring) -> type[BaseModel]:
    """
    Create a dynamic Pydantic model based on callable's return type.

    Args:
        func: The callable (function/method) to analyze
        model_name: Name for the generated model

    Returns:
        Pydantic model class for response
    """

    return_type = inspect.signature(func).return_annotation

    if return_type is inspect.Signature.empty:
        msg = f"Pipeline wrapper is missing a return type for '{func.__name__}' method"
        raise PipelineWrapperError(msg)

    return_description = docstring.returns.description if docstring.returns else None

    return create_model(f"{model_name}Response", result=(return_type, Field(..., description=return_description)))
