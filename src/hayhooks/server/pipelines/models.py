import inspect
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any, get_origin

from docstring_parser.common import Docstring
from fastapi.responses import Response
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


def _is_streaming_type(return_type: type) -> bool:
    """Check if the return type is a Generator or AsyncGenerator (streaming response)."""
    origin = get_origin(return_type)
    return origin in {Generator, AsyncGenerator} or return_type in {Generator, AsyncGenerator}


def _is_response_type(return_type: type) -> bool:
    """Check if the return type is a Response or a subclass (e.g. FileResponse, StreamingResponse)."""
    return inspect.isclass(return_type) and issubclass(return_type, Response)


def create_response_model_from_callable(
    func: Callable, model_name: str, docstring: Docstring
) -> type[BaseModel] | None:
    """
    Create a dynamic Pydantic model based on callable's return type.

    Returns None when the return type indicates the endpoint will bypass JSON serialization
    (i.e. streaming generators or Response subclasses). In these cases, the response is handled
    directly at runtime and no Pydantic model is needed. Returning None follows FastAPI's convention
    of passing ``response_model=None`` for non-JSON endpoints.

    Args:
        func: The callable (function/method) to analyze
        model_name: Name for the generated model
        docstring: Parsed docstring for field descriptions

    Returns:
        Pydantic model class for response, or None for streaming/file responses.
    """

    return_type = inspect.signature(func).return_annotation

    if return_type is inspect.Signature.empty:
        msg = f"Pipeline wrapper is missing a return type for '{func.__name__}' method"  # type:ignore[attr-defined]
        raise PipelineWrapperError(msg)

    # When a pipeline wrapper returns a Generator/AsyncGenerator (streaming) or a Response subclass
    # (e.g. FileResponse), there is no JSON response to validate. At runtime these are intercepted
    # by _streaming_response_from_result() and returned directly. Returning None here signals to
    # add_pipeline_api_route() that it should pass response_model=None to FastAPI, which is the
    # idiomatic way to declare non-JSON endpoints.
    if _is_streaming_type(return_type) or _is_response_type(return_type):
        return None

    return_description = docstring.returns.description if docstring.returns else None

    return create_model(f"{model_name}Response", result=(return_type, Field(..., description=return_description)))
