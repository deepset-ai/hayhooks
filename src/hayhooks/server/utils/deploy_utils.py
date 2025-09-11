import importlib.util
import inspect
import shutil
import sys
import tempfile
import traceback
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional, Union, cast

import docstring_parser
from docstring_parser.common import Docstring
from fastapi import FastAPI, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from haystack import AsyncPipeline, Pipeline
from pydantic import BaseModel, Field, create_model

from hayhooks.server.exceptions import (
    PipelineAlreadyExistsError,
    PipelineFilesError,
    PipelineModuleLoadError,
    PipelineWrapperError,
)
from hayhooks.server.logger import log
from hayhooks.server.pipelines import registry
from hayhooks.server.pipelines.models import (
    PipelineDefinition,
    convert_component_output,
    get_request_model,
    get_request_model_from_resolved_io,
    get_response_model,
    get_response_model_from_resolved_io,
)
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.yaml_utils import get_inputs_outputs_from_yaml
from hayhooks.settings import settings


def deploy_pipeline_def(app: FastAPI, pipeline_def: PipelineDefinition) -> dict[str, str]:
    """
    Deploy a pipeline definition to the FastAPI application.

    NOTE: This is a legacy method which is used in YAML-only based deployments.
          It's not maintained anymore and will be removed in a future version.

    Args:
        app: FastAPI application instance
        pipeline_def: PipelineDefinition instance
    """
    try:
        pipe = registry.add(pipeline_def.name, pipeline_def.source_code)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=f"{e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e

    if isinstance(pipe, BasePipelineWrapper):
        msg = "Pipelines of type BasePipelineWrapper are not supported"
        raise ValueError(msg)

    PipelineRunRequest = get_request_model(pipeline_def.name, pipe.inputs())
    PipelineRunResponse = get_response_model(pipeline_def.name, pipe.outputs())

    # There's no way in FastAPI to define the type of the request body other than annotating
    # the endpoint handler. We have to ignore the type here to make FastAPI happy while
    # silencing static type checkers (that would have good reasons to trigger!).
    async def pipeline_run(pipeline_run_req: PipelineRunRequest) -> JSONResponse:  # type:ignore[valid-type]
        result = await run_in_threadpool(pipe.run, data=pipeline_run_req.dict())  # type:ignore[attr-defined]
        final_output = {}
        for component_name, output in result.items():
            final_output[component_name] = convert_component_output(output)

        return JSONResponse(PipelineRunResponse(**final_output).model_dump(), status_code=200)

    app.add_api_route(
        path=f"/{pipeline_def.name}",
        endpoint=pipeline_run,
        methods=["POST"],
        name=pipeline_def.name,
        response_model=PipelineRunResponse,
        tags=["pipelines"],
    )
    app.openapi_schema = None
    app.setup()

    return {"name": pipeline_def.name}


def save_pipeline_files(pipeline_name: str, files: dict[str, str], pipelines_dir: str) -> dict[str, str]:
    """
    Save pipeline files to disk and return their paths.

    Args:
        pipeline_name: Name of the pipeline
        files: Dictionary mapping filenames to their contents
        pipelines_dir: Path to the pipelines directory
    Returns:
        Dictionary mapping filenames to their saved paths

    Raises:
        PipelineFilesError: If there are any issues saving the files
    """
    try:
        # Create pipeline directory under the configured pipelines directory
        pipeline_dir = Path(pipelines_dir) / pipeline_name
        log.debug(f"Creating pipeline dir: {pipeline_dir}")

        pipeline_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        for filename, content in files.items():
            file_path = pipeline_dir / filename

            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save file content
            file_path.write_text(content)
            saved_files[filename] = str(file_path)

        return saved_files

    except Exception as e:
        msg = f"Failed to save pipeline files: {e!s}"
        raise PipelineFilesError(msg) from e


def save_yaml_pipeline_file(pipeline_name: str, source_code: str, pipelines_dir: str) -> str:
    """
    Save a single YAML pipeline file in the pipelines directory as {name}.yml.

    Args:
        pipeline_name: Name of the pipeline
        source_code: YAML content
        pipelines_dir: Path to the pipelines directory

    Returns:
        The saved file path as string

    Raises:
        PipelineFilesError: If there are any issues saving the file
    """
    try:
        pipelines_dir_path = Path(pipelines_dir)
        pipelines_dir_path.mkdir(parents=True, exist_ok=True)
        file_path = pipelines_dir_path / f"{pipeline_name}.yml"
        log.debug(f"Saving YAML pipeline file: {file_path}")
        file_path.write_text(source_code)
        return str(file_path)
    except Exception as e:
        msg = f"Failed to save YAML pipeline file: {e!s}"
        raise PipelineFilesError(msg) from e


def remove_pipeline_files(pipeline_name: str, pipelines_dir: str) -> None:
    """
    Remove pipeline files from disk.

    Args:
        pipeline_name: Name of the pipeline
        pipelines_dir: Path to the pipelines directory
    """
    pipeline_dir = Path(pipelines_dir) / pipeline_name
    if pipeline_dir.exists():
        shutil.rmtree(pipeline_dir, ignore_errors=True)


def load_pipeline_module(pipeline_name: str, dir_path: Union[Path, str]) -> ModuleType:
    """
    Load a pipeline module from a directory path.

    Args:
        pipeline_name: Name of the pipeline
        dir_path: Path to the directory containing the pipeline files

    Returns:
        The loaded module

    Raises:
        ValueError: If the module cannot be loaded
    """
    log.trace(f"Loading pipeline module from {dir_path}")
    log.trace(f"Is folder present: {Path(dir_path).exists()}")

    try:
        dir_path = Path(dir_path)
        wrapper_path = dir_path / "pipeline_wrapper.py"

        if not wrapper_path.exists():
            msg = f"Required file '{wrapper_path}' not found"
            raise PipelineWrapperError(msg)

        # Clear the module from sys.modules if it exists to ensure a fresh load
        module_name = pipeline_name
        if module_name in sys.modules:
            log.debug(f"Removing existing module {module_name} from sys.modules")
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(pipeline_name, wrapper_path)
        if spec is None or spec.loader is None:
            msg = f"Failed to load pipeline module '{pipeline_name}' - module loader not available"
            raise PipelineModuleLoadError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        log.debug(f"Loaded module {module}")

        if not hasattr(module, "PipelineWrapper"):
            msg = f"Failed to load '{pipeline_name}' pipeline module spec"
            raise PipelineWrapperError(msg)

        return module

    except Exception as e:
        log.error(f"Error loading pipeline module: {e!s}")
        error_msg = f"Failed to load pipeline module '{pipeline_name}' - {e!s}"
        if settings.show_tracebacks:
            error_msg += f"\n{traceback.format_exc()}"
        raise PipelineModuleLoadError(error_msg) from e


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


def handle_pipeline_exceptions() -> Callable:
    """Decorator to handle pipeline execution exceptions."""

    def decorator(func):
        @wraps(func)  # Preserve the original function's metadata
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException as e:
                raise e from e
            except Exception as e:
                error_msg = f"Pipeline execution failed: {e!s}"
                if settings.show_tracebacks:
                    log.error(f"Pipeline execution error: {e!s} - {traceback.format_exc()}")
                    error_msg += f"\n{traceback.format_exc()}"
                else:
                    log.error(f"Pipeline execution error: {e!s}")
                raise HTTPException(status_code=500, detail=error_msg) from e

        return wrapper

    return decorator


def create_run_endpoint_handler(
    pipeline_wrapper: BasePipelineWrapper,
    request_model: type[BaseModel],
    response_model: type[BaseModel],
    requires_files: bool,
) -> Callable:
    """
    Factory method to create the appropriate run endpoint handler based on whether file uploads are supported.

    Note:
        There's no way in FastAPI to define the type of the request body other than annotating
        the endpoint handler. We have to **ignore types several times in this method** to make FastAPI happy while
        silencing static type checkers (that would have good reasons to trigger!).

    Args:
        pipeline_wrapper: The pipeline wrapper instance
        request_model: The request model
        response_model: The response model
        requires_files: Whether the pipeline requires file uploads
    """

    @handle_pipeline_exceptions()
    async def run_endpoint_with_files(
        run_req: request_model = Form(..., media_type="multipart/form-data"),  # type:ignore[valid-type] # noqa: B008
    ) -> response_model:  # type:ignore[valid-type]
        if pipeline_wrapper._is_run_api_async_implemented:
            result = await pipeline_wrapper.run_api_async(**run_req.model_dump())  # type:ignore[attr-defined]
        else:
            result = await run_in_threadpool(pipeline_wrapper.run_api, **run_req.model_dump())  # type:ignore[attr-defined]
        return response_model(result=result)

    @handle_pipeline_exceptions()
    async def run_endpoint_without_files(run_req: request_model) -> response_model:  # type:ignore[valid-type]
        if pipeline_wrapper._is_run_api_async_implemented:
            result = await pipeline_wrapper.run_api_async(**run_req.model_dump())  # type:ignore[attr-defined]
        else:
            result = await run_in_threadpool(pipeline_wrapper.run_api, **run_req.model_dump())  # type:ignore[attr-defined]
        return response_model(result=result)

    return run_endpoint_with_files if requires_files else run_endpoint_without_files


def add_pipeline_wrapper_api_route(app: FastAPI, pipeline_name: str, pipeline_wrapper: BasePipelineWrapper) -> None:
    clog = log.bind(pipeline_name=pipeline_name)

    # Determine which run_api method to use (prefer async if available)
    if pipeline_wrapper._is_run_api_async_implemented:
        run_method_to_inspect = pipeline_wrapper.run_api_async
        clog.debug("Using `run_api_async` as API route handler.")
    elif pipeline_wrapper._is_run_api_implemented:
        run_method_to_inspect = pipeline_wrapper.run_api
        clog.debug("Using `run_api` as API route handler.")
    else:
        # If neither run_api nor run_api_async is implemented,
        # this pipeline will not have a generic /<pipeline_name>/run endpoint.
        # This is a valid configuration (e.g., for chat-only pipelines).
        clog.warning(
            f"Pipeline '{pipeline_name}' does not implement `run_api` or `run_api_async`. "
            f"Skipping /{pipeline_name}/run API route creation."
        )
        return

    docstring_content = inspect.getdoc(run_method_to_inspect) or ""
    docstring = docstring_parser.parse(docstring_content)
    RunRequest = create_request_model_from_callable(run_method_to_inspect, f"{pipeline_name}Run", docstring)
    RunResponse = create_response_model_from_callable(run_method_to_inspect, f"{pipeline_name}Run", docstring)

    run_api_params = inspect.signature(run_method_to_inspect).parameters
    requires_files = "files" in run_api_params
    clog.debug(f"Pipeline requires files: {requires_files}")

    run_endpoint = create_run_endpoint_handler(
        pipeline_wrapper=pipeline_wrapper,
        request_model=RunRequest,
        response_model=RunResponse,
        requires_files=requires_files,
    )

    # Clear existing pipeline run route if it exists
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == f"/{pipeline_name}/run":
            app.routes.remove(route)

    app.add_api_route(
        path=f"/{pipeline_name}/run",
        endpoint=run_endpoint,
        methods=["POST"],
        name=f"{pipeline_name}_run",
        response_model=RunResponse,
        tags=["pipelines"],
        description=docstring.short_description or None,
    )

    registry.update_metadata(
        pipeline_name,
        {
            "request_model": RunRequest,
            "response_model": RunResponse,
            "requires_files": requires_files,
        },
    )

    clog.debug("Setting up FastAPI app")
    app.openapi_schema = None
    app.setup()


def add_pipeline_yaml_api_route(app: FastAPI, pipeline_name: str, source_code: str) -> None:
    """
    Create or replace the YAML pipeline run endpoint at /{pipeline_name}/run.

    Builds the flat request/response models from declared YAML inputs/outputs and wires a handler that
    maps the flat body into the nested structure required by Haystack Pipeline.run.
    """
    pipeline_instance = registry.get(pipeline_name)
    if pipeline_instance is None:
        msg = f"Pipeline '{pipeline_name}' not found after registration"
        raise HTTPException(status_code=500, detail=msg)

    # Ensure the registered object is a Haystack Pipeline, not a wrapper
    if not isinstance(pipeline_instance, (Pipeline, AsyncPipeline)):
        msg = f"Pipeline '{pipeline_name}' is not a Haystack Pipeline instance"
        raise HTTPException(status_code=500, detail=msg)

    pipeline: Union[Pipeline, AsyncPipeline] = pipeline_instance

    # Compute IO resolution to map flat request fields to pipeline.run nested inputs
    resolved_io = get_inputs_outputs_from_yaml(source_code)
    declared_inputs = resolved_io["inputs"]
    declared_outputs = resolved_io["outputs"]

    metadata = registry.get_metadata(pipeline_name) or {}
    PipelineRunRequest = metadata.get("request_model")
    PipelineRunResponse = metadata.get("response_model")

    if PipelineRunRequest is None or PipelineRunResponse is None:
        msg = f"Missing request/response models for YAML pipeline '{pipeline_name}'"
        raise HTTPException(status_code=500, detail=msg)

    @handle_pipeline_exceptions()
    async def pipeline_run(run_req: PipelineRunRequest) -> PipelineRunResponse:  # type: ignore
        # Map flat declared inputs to the nested structure expected by Haystack Pipeline.run
        payload: dict[str, dict[str, Any]] = {}
        req_dict = run_req.model_dump()  # type: ignore[attr-defined]
        for input_name, in_resolution in declared_inputs.items():
            value = req_dict.get(input_name)
            if value is None:
                continue
            component_inputs = payload.setdefault(in_resolution.component, {})
            component_inputs[in_resolution.name] = value

        # Execute the pipeline
        result = await run_in_threadpool(pipeline.run, data=payload)

        # Map pipeline outputs back to declared outputs (flat)
        final_output: dict[str, Any] = {}
        for output_name, out_resolution in declared_outputs.items():
            component_result = (result or {}).get(out_resolution.component, {})
            raw_value = component_result.get(out_resolution.name)
            final_output[output_name] = convert_component_output(raw_value)

        return PipelineRunResponse(**final_output)

    # Clear existing YAML run route if it exists (old or new path)
    for route in list(app.routes):
        if isinstance(route, APIRoute) and route.path in (f"/{pipeline_name}", f"/{pipeline_name}/run"):
            app.routes.remove(route)

    # Register the run endpoint at /{pipeline_name}/run
    app.add_api_route(
        path=f"/{pipeline_name}/run",
        endpoint=pipeline_run,
        methods=["POST"],
        name=f"{pipeline_name}_run",
        response_model=PipelineRunResponse,
        tags=["pipelines"],
    )

    # Invalidate OpenAPI cache
    app.openapi_schema = None
    app.setup()


def deploy_pipeline_files(
    pipeline_name: str,
    files: dict[str, str],
    app: Optional[FastAPI] = None,
    save_files: bool = True,
    overwrite: bool = False,
) -> dict[str, str]:
    """
    Deploy a pipeline.

    This will add the pipeline to the registry and optionally set up the API route if `app` is provided.

    Args:
        pipeline_name: Name of the pipeline to deploy
        files: Dictionary mapping filenames to their contents
        app: Optional FastAPI application instance. If provided, the API route will be added.
        save_files: Whether to save the pipeline files to disk
        overwrite: Whether to overwrite an existing pipeline
    """
    pipeline_wrapper = add_pipeline_wrapper_to_registry(pipeline_name, files, save_files, overwrite)

    if app:
        add_pipeline_wrapper_api_route(app, pipeline_name, pipeline_wrapper)

    return {"name": pipeline_name}


def deploy_pipeline_yaml(
    app: FastAPI,
    pipeline_name: str,
    source_code: str,
    overwrite: bool = False,
    options: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """
    Deploy a YAML pipeline to the FastAPI application with IO declared in the YAML.

    This will add the pipeline to the registry, create flat request/response models based on
    declared inputs/outputs, and set up the API route at /{pipeline_name}/run.

    Args:
        app: FastAPI application instance
        pipeline_name: Name of the pipeline
        source_code: YAML pipeline source code
        overwrite: Whether to overwrite an existing pipeline
        options: Optional dict with additional deployment options. Supported keys:
            - description: Optional[str]
            - skip_mcp: Optional[bool]
    """

    # Optionally save YAML to disk as pipelines/{name}.yml (default True)
    save_file: bool = True if options is None else bool(options.get("save_file", True))
    if save_file:
        save_yaml_pipeline_file(pipeline_name, source_code, settings.pipelines_dir)

    # Add pipeline to the registry and build metadata (request/response models)
    add_yaml_pipeline_to_registry(
        pipeline_name=pipeline_name,
        source_code=source_code,
        overwrite=overwrite,
        description=(options or {}).get("description"),
        skip_mcp=(options or {}).get("skip_mcp"),
    )

    # Add the YAML pipeline API route at /{pipeline_name}/run
    add_pipeline_yaml_api_route(app, pipeline_name, source_code)

    return {"name": pipeline_name}


def add_yaml_pipeline_to_registry(
    pipeline_name: str,
    source_code: str,
    overwrite: bool = False,
    description: Optional[str] = None,
    skip_mcp: Optional[bool] = False,
) -> None:
    """
    Add a YAML pipeline to the registry.

    Args:
        pipeline_name: Name of the pipeline to deploy
        source_code: Source code of the pipeline
    """

    log.debug(f"Checking if YAML pipeline '{pipeline_name}' already exists: {registry.get(pipeline_name)}")
    if registry.get(pipeline_name):
        if overwrite:
            log.debug(f"Clearing existing YAML pipeline '{pipeline_name}'")
            registry.remove(pipeline_name)
        else:
            msg = f"YAML pipeline '{pipeline_name}' already exists"
            raise PipelineAlreadyExistsError(msg)

    clog = log.bind(pipeline_name=pipeline_name, type="yaml")

    clog.debug("Creating request/response models from declared YAML inputs/outputs")

    # Build request/response models from declared YAML inputs/outputs using resolved IO types
    try:
        resolved_io = get_inputs_outputs_from_yaml(source_code)

        pipeline_inputs = resolved_io["inputs"]
        pipeline_outputs = resolved_io["outputs"]

        # Prefer resolved IO-based flat models for API schema
        request_model = get_request_model_from_resolved_io(pipeline_name, pipeline_inputs)
        response_model = get_response_model_from_resolved_io(pipeline_name, pipeline_outputs)
    except Exception as e:
        clog.error(f"Failed creating request/response models for YAML pipeline: {e!s}")
        raise

    metadata = {
        "description": description or pipeline_name,
        "request_model": request_model,
        "response_model": response_model,
        "skip_mcp": bool(skip_mcp),
    }

    clog.debug(f"Adding YAML pipeline to registry with metadata: {metadata}")

    # Store the instantiated pipeline together with its metadata
    try:
        from haystack import Pipeline

        pipeline = Pipeline.loads(source_code)
    except Exception as e:
        msg = f"Unable to parse Haystack Pipeline {pipeline_name}: {e!s}"
        raise ValueError(msg) from e

    registry.add(pipeline_name, pipeline, metadata=metadata)


def add_pipeline_wrapper_to_registry(
    pipeline_name: str, files: dict[str, str], save_files: bool = True, overwrite: bool = False
) -> BasePipelineWrapper:
    """
    Add a pipeline to the registry.

    Args:

    Args:
        pipeline_name: Name of the pipeline to deploy
        files: Dictionary mapping filenames to their contents

    Returns:
        dict: Dictionary containing the deployed pipeline name
    """

    log.debug(f"Checking if pipeline '{pipeline_name}' already exists: {registry.get(pipeline_name)}")
    if registry.get(pipeline_name):
        if overwrite:
            log.debug(f"Clearing existing pipeline '{pipeline_name}'")
            registry.remove(pipeline_name)

            log.debug(f"Removing pipeline files for '{pipeline_name}'")
            remove_pipeline_files(pipeline_name, settings.pipelines_dir)
        else:
            msg = f"Pipeline '{pipeline_name}' already exists"
            raise PipelineAlreadyExistsError(msg)

    tmp_dir = None

    if save_files:
        log.debug(f"Saving pipeline files for '{pipeline_name}' in '{settings.pipelines_dir}'")
        save_pipeline_files(pipeline_name, files=files, pipelines_dir=settings.pipelines_dir)
        pipeline_dir = Path(settings.pipelines_dir) / pipeline_name
    else:
        # We still need to save the pipeline files to disk to be able to load the module
        # We do in a temporary directory to avoid polluting the pipelines directory
        tmp_dir = tempfile.mkdtemp()
        save_pipeline_files(pipeline_name, files=files, pipelines_dir=tmp_dir)
        pipeline_dir = Path(tmp_dir) / pipeline_name

    clog = log.bind(pipeline_name=pipeline_name, pipeline_dir=str(pipeline_dir), files=list(files.keys()))

    clog.debug("Loading pipeline module")
    module = load_pipeline_module(pipeline_name, dir_path=pipeline_dir)

    clog.debug("Creating PipelineWrapper instance")
    pipeline_wrapper = create_pipeline_wrapper_instance(module)

    clog.debug("Running setup()")
    pipeline_wrapper.setup()

    # Determine which run_api method to use for creating request model (prefer async if available)
    if pipeline_wrapper._is_run_api_async_implemented:
        run_method_to_inspect = pipeline_wrapper.run_api_async
        clog.debug("Using `run_api_async` for metadata creation.")
    elif pipeline_wrapper._is_run_api_implemented:
        run_method_to_inspect = pipeline_wrapper.run_api
        clog.debug("Using `run_api` for metadata creation.")
    else:
        # If neither run_api nor run_api_async is implemented, skip creating request model
        run_method_to_inspect = None
        clog.debug("No run_api method implemented, skipping request model creation.")

    if run_method_to_inspect:
        docstring = docstring_parser.parse(inspect.getdoc(run_method_to_inspect) or "")
        request_model = create_request_model_from_callable(run_method_to_inspect, f"{pipeline_name}Run", docstring)
    else:
        docstring = docstring_parser.Docstring()
        request_model = None

    metadata = {
        "description": docstring.short_description or "",
        "request_model": request_model,
        "skip_mcp": pipeline_wrapper.skip_mcp,
    }

    clog.debug(f"Adding pipeline to registry with metadata: {metadata}")
    registry.add(
        pipeline_name,
        pipeline_wrapper,
        metadata=metadata,
    )

    clog.success("Pipeline successfully added to registry")

    if tmp_dir is not None:
        log.debug(f"Removing temporary pipeline files for '{pipeline_name}'")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return pipeline_wrapper


def create_pipeline_wrapper_instance(pipeline_module: ModuleType) -> BasePipelineWrapper:
    try:
        pipeline_wrapper = pipeline_module.PipelineWrapper()
    except Exception as e:
        error_msg = "Failed to create pipeline wrapper instance: " + str(e)
        if settings.show_tracebacks:
            error_msg += f"\n{traceback.format_exc()}"
        raise PipelineWrapperError(error_msg) from e

    try:
        pipeline_wrapper.setup()
    except Exception as e:
        error_msg = "Failed to call setup() on pipeline wrapper instance: " + str(e)
        if settings.show_tracebacks:
            error_msg += f"\n{traceback.format_exc()}"
        raise PipelineWrapperError(error_msg) from e

    # Determine if the run_api, run_chat_completion, and their async versions are implemented
    _set_method_implementation_flag(pipeline_wrapper, "_is_run_api_implemented", "run_api")
    _set_method_implementation_flag(pipeline_wrapper, "_is_run_api_async_implemented", "run_api_async")
    _set_method_implementation_flag(pipeline_wrapper, "_is_run_chat_completion_implemented", "run_chat_completion")
    _set_method_implementation_flag(
        pipeline_wrapper, "_is_run_chat_completion_async_implemented", "run_chat_completion_async"
    )

    log.debug(f"pipeline_wrapper._is_run_api_implemented: {pipeline_wrapper._is_run_api_implemented}")
    log.debug(f"pipeline_wrapper._is_run_api_async_implemented: {pipeline_wrapper._is_run_api_async_implemented}")
    log.debug(
        f"pipeline_wrapper._is_run_chat_completion_implemented: {pipeline_wrapper._is_run_chat_completion_implemented}"
    )
    log.debug(
        "pipeline_wrapper._is_run_chat_completion_async_implemented: "
        f"{pipeline_wrapper._is_run_chat_completion_async_implemented}"
    )

    if not (
        pipeline_wrapper._is_run_api_implemented
        or pipeline_wrapper._is_run_api_async_implemented
        or pipeline_wrapper._is_run_chat_completion_implemented
        or pipeline_wrapper._is_run_chat_completion_async_implemented
    ):
        msg = (
            "At least one of run_api, run_api_async, run_chat_completion, or run_chat_completion_async "
            "must be implemented"
        )
        raise PipelineWrapperError(msg)

    return pipeline_wrapper


def _set_method_implementation_flag(pipeline_wrapper: BasePipelineWrapper, attr_name: str, method_name: str) -> None:
    """Helper to check if a method is implemented on the wrapper compared to the base."""
    wrapper_method = getattr(pipeline_wrapper, method_name, None)
    base_method = getattr(BasePipelineWrapper, method_name, None)
    if wrapper_method and base_method:
        # Ensure we are comparing the function itself, not the bound method if one is already bound.
        # For unbound methods (like on the class itself), __func__ is the function.
        # For bound methods (on an instance), __func__ gives the original function.
        setattr(
            pipeline_wrapper,
            attr_name,
            getattr(wrapper_method, "__func__", wrapper_method) is not getattr(base_method, "__func__", base_method),
        )
    else:
        # Fallback or error handling if methods are not found
        setattr(pipeline_wrapper, attr_name, False)


def read_pipeline_files_from_dir(dir_path: Path) -> dict[str, str]:
    """
    Read pipeline files from a directory and return a dictionary mapping filenames to their contents.

    Skips directories, hidden files, and common Python artifacts.

    Args:
        dir_path: Path to the directory containing the pipeline files

    Returns:
        Dictionary mapping filenames to their contents
    """

    files = {}
    for file_path in dir_path.rglob("*"):
        if file_path.is_dir() or file_path.name.startswith("."):
            continue

        if any(file_path.match(pattern) for pattern in settings.files_to_ignore_patterns):
            continue

        try:
            files[str(file_path.relative_to(dir_path))] = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log.warning(f"Skipping file {file_path}: {e!s}")
            continue

    return files


def undeploy_pipeline(pipeline_name: str, app: Optional[FastAPI] = None) -> None:
    """
    Undeploy a pipeline.

    Removes a pipeline from the registry, removes its API routes and deletes its files from disk.

    Args:
        pipeline_name: Name of the pipeline to undeploy.
        app: Optional FastAPI application instance. If provided, API routes will be removed.
    """
    # Check if pipeline exists in registry
    if pipeline_name not in registry.get_names():
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")

    # Remove pipeline from registry
    registry.remove(pipeline_name)

    if app:
        # Remove API routes for the pipeline
        # YAML based pipelines have a run endpoint at /<pipeline_name>
        # Wrapper based pipelines have a run endpoint at /<pipeline_name>/run
        routes_to_remove = [
            route
            for route in app.routes
            if isinstance(route, APIRoute) and (route.path in (f"/{pipeline_name}/run", f"/{pipeline_name}"))
        ]
        for route in routes_to_remove:
            app.routes.remove(route)

        # Invalidate OpenAPI cache
        app.openapi_schema = None
        app.setup()

    # Remove pipeline files if they exist
    remove_pipeline_files(pipeline_name, settings.pipelines_dir)
