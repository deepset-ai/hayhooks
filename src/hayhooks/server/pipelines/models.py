import inspect
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any, get_origin

from docstring_parser.common import Docstring
from fastapi.responses import Response, StreamingResponse
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
    directly at runtime and no Pydantic model is needed. Returning None causes the route to be
    registered with ``response_model=None``, telling FastAPI to skip response validation.

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

    # Streaming generators and Response subclasses (e.g. FileResponse) produce non-JSON output
    # that cannot be described by a Pydantic model. At runtime, these are intercepted by
    # _streaming_response_from_result() in deploy_utils.py and returned directly to the client,
    # so a Pydantic response model is never used for validation or serialization.
    #
    # Returning None here causes add_pipeline_api_route() to register the FastAPI route with
    # response_model=None, which tells FastAPI to skip response validation and not generate a
    # JSON schema for this endpoint. The companion function get_response_class_from_callable()
    # provides the concrete response_class (e.g. FileResponse, StreamingResponse) so that
    # OpenAPI docs also show the correct Content-Type.
    if _is_streaming_type(return_type) or _is_response_type(return_type):
        return None

    return_description = docstring.returns.description if docstring.returns else None

    return create_model(f"{model_name}Response", result=(return_type, Field(..., description=return_description)))


def get_response_class_from_callable(func: Callable) -> type[Response] | None:
    """
    Determine the appropriate ``response_class`` for a FastAPI route.

    FastAPI uses ``response_class`` to set the Content-Type header in OpenAPI docs and to decide
    how to wrap a return value that is not already a ``Response`` instance. By returning the
    concrete Response subclass here, we produce more accurate OpenAPI documentation for endpoints
    that return files or streams.

    Returns:
        * The annotated Response subclass (e.g. ``FileResponse``) when the return type is a
          Response subclass.
        * ``StreamingResponse`` when the return type is a ``Generator`` or ``AsyncGenerator``,
          since generators are wrapped in ``StreamingResponse`` at runtime.
        * ``None`` for normal JSON endpoints (the caller should omit the ``response_class``
          kwarg so FastAPI uses its default ``JSONResponse``).
    """
    return_type = inspect.signature(func).return_annotation

    if return_type is inspect.Signature.empty:
        return None

    if _is_response_type(return_type):
        return return_type

    if _is_streaming_type(return_type):
        return StreamingResponse

    return None
