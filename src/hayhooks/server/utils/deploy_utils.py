import inspect
import json
import shutil
import tempfile
import traceback
from collections.abc import AsyncGenerator as AsyncGeneratorABC
from collections.abc import Callable
from collections.abc import Generator as GeneratorABC
from functools import wraps
from pathlib import Path
from typing import Any

import docstring_parser
from fastapi import FastAPI, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import APIRoute
from haystack.dataclasses import StreamingChunk
from pydantic import BaseModel

from hayhooks.open_webui import OpenWebUIEvent
from hayhooks.server.exceptions import PipelineAlreadyExistsError, PipelineFilesError
from hayhooks.server.logger import log
from hayhooks.server.pipelines import registry
from hayhooks.server.pipelines.models import create_request_model_from_callable, create_response_model_from_callable
from hayhooks.server.pipelines.sse import SSEStream
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.module_loader import (
    create_pipeline_wrapper_instance,
    load_pipeline_module,
    unload_pipeline_modules,
)
from hayhooks.server.utils.yaml_pipeline_wrapper import YAMLPipelineWrapper
from hayhooks.settings import settings


def _is_single_yaml_file(files: dict[str, str]) -> bool:
    """Check if files dict represents a single YAML pipeline file."""
    if len(files) != 1:
        return False
    filename = next(iter(files.keys()))
    return filename.endswith((".yml", ".yaml"))


def save_pipeline_files(pipeline_name: str, files: dict[str, str], pipelines_dir: str) -> dict[str, str]:
    """
    Save pipeline files to disk and return their paths.

    For single YAML files, saves directly as pipelines_dir/{pipeline_name}.yml.
    For multiple files (wrapper pipelines), saves to pipelines_dir/{pipeline_name}/.

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
        pipelines_path = Path(pipelines_dir)
        pipelines_path.mkdir(parents=True, exist_ok=True)

        # Single YAML file
        # Save directly in pipelines_dir as {name}.yml
        if _is_single_yaml_file(files):
            content = next(iter(files.values()))
            file_path = pipelines_path / f"{pipeline_name}.yml"
            log.debug("Saving YAML pipeline file: '{}'", file_path)
            file_path.write_text(content)
            return {f"{pipeline_name}.yml": str(file_path)}

        # Multiple files
        # Save in subdirectory pipelines_dir/{name}/
        pipeline_dir = pipelines_path / pipeline_name
        log.debug("Creating pipeline dir: '{}'", pipeline_dir)
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        for filename, content in files.items():
            file_path = pipeline_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            saved_files[filename] = str(file_path)

        return saved_files

    except Exception as e:
        msg = f"Failed to save pipeline files for '{pipeline_name}': {e!s}"
        raise PipelineFilesError(msg) from e


def remove_pipeline_files(pipeline_name: str, pipelines_dir: str) -> None:
    """
    Remove pipeline files from disk.

    Removes both:
    - Pipeline directories (for wrapper-based pipelines)
    - YAML files (for YAML-based pipelines saved as {name}.yml)

    Args:
        pipeline_name: Name of the pipeline
        pipelines_dir: Path to the pipelines directory
    """
    pipelines_path = Path(pipelines_dir)

    # Remove pipeline directory (wrapper-based pipelines)
    shutil.rmtree(pipelines_path / pipeline_name, ignore_errors=True)

    # Remove YAML files (YAML-based pipelines)
    for ext in (".yml", ".yaml"):
        (pipelines_path / f"{pipeline_name}{ext}").unlink(missing_ok=True)


def handle_pipeline_exceptions() -> Callable:
    """
    Decorator factory that wraps pipeline run methods and processes unexpected exceptions.

    Returns:
        A decorator that can be applied to async pipeline run methods.
    """

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
                    log.opt(exception=True).error("Pipeline execution error: {} - {}", e, traceback.format_exc())
                    error_msg += f"\n{traceback.format_exc()}"
                else:
                    log.opt(exception=True).error("Pipeline execution error: {}", e)
                raise HTTPException(status_code=500, detail=error_msg) from e

        return wrapper

    return decorator


def _format_run_stream_chunk(stream_item: Any) -> str | bytes | None:
    if isinstance(stream_item, StreamingChunk):
        return stream_item.content or ""
    if isinstance(stream_item, OpenWebUIEvent):
        log.warning("OpenWebUIEvent emitted during /run streaming; skipping. Use OpenAI chat endpoints for UI events.")
        return None
    if isinstance(stream_item, (str, bytes)):
        return stream_item
    if stream_item is None:
        return ""
    try:
        return json.dumps(stream_item)
    except TypeError:
        return str(stream_item)


def _format_sse_chunk(formatted: str | bytes) -> str:
    text = formatted.decode("utf-8", errors="replace") if isinstance(formatted, bytes) else str(formatted)

    if text == "":
        return "data:\n\n"

    lines = text.splitlines()
    if not lines:
        return "data:\n\n"

    data_lines = "".join(f"data: {line}\n" for line in lines)
    return f"{data_lines}\n"


def _streaming_response_from_async_gen(async_gen: Any, media_type: str = "text/plain") -> Response:
    is_sse = media_type == "text/event-stream"

    async def async_stream():
        try:
            async for item in async_gen:
                formatted = _format_run_stream_chunk(item)
                if formatted is None:
                    continue
                if is_sse:
                    formatted = _format_sse_chunk(formatted)
                yield formatted
        finally:
            aclose = getattr(async_gen, "aclose", None)
            if callable(aclose):
                await aclose()

    return StreamingResponse(async_stream(), media_type=media_type)


def _streaming_response_from_gen(gen: Any, media_type: str = "text/plain") -> Response:
    is_sse = media_type == "text/event-stream"

    def sync_stream():
        try:
            for item in gen:
                formatted = _format_run_stream_chunk(item)
                if formatted is None:
                    continue
                if is_sse:
                    formatted = _format_sse_chunk(formatted)
                yield formatted
        finally:
            close = getattr(gen, "close", None)
            if callable(close):
                close()

    return StreamingResponse(sync_stream(), media_type=media_type)


def _streaming_response_from_result(result: Any) -> Response | None:
    # If the result is a SSEStream, return a StreamingResponse with the appropriate media type
    if isinstance(result, SSEStream):
        # Get the stream from the SSEStream
        stream = result.stream

        # If the stream is an async generator, return a StreamingResponse with the appropriate media type
        if isinstance(stream, AsyncGeneratorABC):
            return _streaming_response_from_async_gen(stream, media_type="text/event-stream")

        # If the stream is a generator, return a StreamingResponse with the appropriate media type
        if isinstance(stream, GeneratorABC):
            return _streaming_response_from_gen(stream, media_type="text/event-stream")

        # If the stream is not a generator or async generator, raise a TypeError
        msg = f"SSEStream.stream must be a generator or async generator (got type {type(stream)!r})"
        raise TypeError(msg)

    # If the result is a Response, return the result
    if isinstance(result, Response):
        return result

    # Following generic cases are for non-SSE streaming responses (plain text)
    if isinstance(result, AsyncGeneratorABC):
        return _streaming_response_from_async_gen(result)
    if isinstance(result, GeneratorABC):
        return _streaming_response_from_gen(result)
    return None


async def _execute_pipeline_run(
    pipeline_wrapper: BasePipelineWrapper,
    payload: dict[str, Any],
) -> Any:
    if pipeline_wrapper._is_run_api_async_implemented:
        return await pipeline_wrapper.run_api_async(**payload)
    return await run_in_threadpool(pipeline_wrapper.run_api, **payload)


def create_run_endpoint_handler(
    pipeline_wrapper: BasePipelineWrapper,
    request_model: type[BaseModel],
    response_model: type[BaseModel] | None,
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
        response_model: The response model, or None for streaming/file response endpoints
        requires_files: Whether the pipeline requires file uploads

    Returns:
        A FastAPI endpoint function that executes the pipeline and returns the response model.
    """

    async def _handle_request(run_req: BaseModel) -> Response | BaseModel:
        payload = run_req.model_dump()

        result = await _execute_pipeline_run(pipeline_wrapper, payload)
        streaming_response = _streaming_response_from_result(result)
        if streaming_response is not None:
            return streaming_response

        # response_model is None for streaming/file response endpoints, where
        # _streaming_response_from_result() always handles the result above.
        # For normal JSON endpoints, wrap the result in the Pydantic response model.
        if response_model is None:
            return result

        return response_model(result=result)

    @handle_pipeline_exceptions()
    async def run_endpoint_with_files(
        run_req: request_model = Form(..., media_type="multipart/form-data"),  # type:ignore[valid-type] # noqa: B008
    ) -> response_model:  # type:ignore[valid-type]
        return await _handle_request(run_req)

    @handle_pipeline_exceptions()
    async def run_endpoint_without_files(run_req: request_model) -> response_model:  # type:ignore[valid-type]
        return await _handle_request(run_req)

    return run_endpoint_with_files if requires_files else run_endpoint_without_files


def add_pipeline_api_route(app: FastAPI, pipeline_name: str, pipeline_wrapper: BasePipelineWrapper) -> None:
    """
    Create or replace the wrapper-based pipeline run endpoint at /{pipeline_name}/run.

    Args:
        app: FastAPI application instance.
        pipeline_name: Name of the pipeline.
        pipeline_wrapper: Initialized pipeline wrapper instance to use as handler target.

    Side Effects:
        - Removes any existing route at /{pipeline_name}/run
        - Rebuilds and invalidates the OpenAPI schema
        - Updates registry metadata with request/response models and file requirement flag
    """
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
    clog.debug("Pipeline requires files: {}", requires_files)

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


def _register_and_deploy_pipeline(
    pipeline_name: str,
    pipeline_wrapper: BasePipelineWrapper,
    app: FastAPI | None = None,
    overwrite: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Common logic for registering and deploying any pipeline wrapper.

    This is the shared core that handles:
    - Checking if pipeline exists and handling overwrite
    - Calling setup() on the wrapper
    - Generating request/response models
    - Building metadata and adding to registry
    - Adding API route if app is provided

    Args:
        pipeline_name: Name of the pipeline.
        pipeline_wrapper: Already-created wrapper instance (setup() will be called here).
        app: Optional FastAPI app for route creation.
        overwrite: Whether to overwrite existing pipeline.
        extra_metadata: Additional metadata fields (e.g., streaming_components for YAML).

    Returns:
        A dictionary containing the deployed pipeline name, e.g. {"name": pipeline_name}.

    Raises:
        PipelineAlreadyExistsError: If the pipeline exists and overwrite is False.
    """
    clog = log.bind(pipeline_name=pipeline_name)

    # Check if pipeline already exists
    if registry.get(pipeline_name):
        if overwrite:
            clog.debug("Clearing existing pipeline '{}'", pipeline_name)
            registry.remove(pipeline_name)
            remove_pipeline_files(pipeline_name, settings.pipelines_dir)
        else:
            msg = f"Pipeline '{pipeline_name}' already exists"
            raise PipelineAlreadyExistsError(msg)

    # Call setup to initialize the pipeline (if not already done)
    if getattr(pipeline_wrapper, "pipeline", None) is None:
        clog.debug("Running setup() on pipeline wrapper")
        pipeline_wrapper.setup()

    # Determine which run method to use for model generation (prefer async)
    if pipeline_wrapper._is_run_api_async_implemented:
        run_method = pipeline_wrapper.run_api_async
        clog.debug("Using `run_api_async` for model generation")
    elif pipeline_wrapper._is_run_api_implemented:
        run_method = pipeline_wrapper.run_api
        clog.debug("Using `run_api` for model generation")
    else:
        run_method = None
        clog.debug("No run_api method implemented, skipping model generation")

    # Generate request/response models from the run method signature
    request_model = None
    description = ""
    if run_method:
        docstring = docstring_parser.parse(inspect.getdoc(run_method) or "")
        description = docstring.short_description or ""
        request_model = create_request_model_from_callable(run_method, f"{pipeline_name}Run", docstring)

    # Build metadata
    metadata: dict[str, Any] = {
        "description": description,
        "request_model": request_model,
        "skip_mcp": pipeline_wrapper.skip_mcp,
    }

    # Merge extra metadata (e.g., YAML-specific fields)
    if extra_metadata:
        metadata.update(extra_metadata)

    # Add wrapper to registry
    clog.debug("Adding pipeline to registry with metadata: {}", metadata)
    registry.add(pipeline_name, pipeline_wrapper, metadata=metadata)
    clog.success("Pipeline '{}' successfully added to registry", pipeline_name)

    # Create API route if app is provided
    if app:
        add_pipeline_api_route(app, pipeline_name, pipeline_wrapper)

    return {"name": pipeline_name}


def deploy_pipeline_files(
    pipeline_name: str,
    files: dict[str, str],
    app: FastAPI | None = None,
    save_files: bool = True,
    overwrite: bool = False,
) -> dict[str, str]:
    """
    Deploy a pipeline from Python files (pipeline_wrapper.py and other files).

    This will save the files, load the module, create a wrapper instance,
    add it to the registry, and optionally set up the API route.

    Args:
        pipeline_name: Name of the pipeline to deploy
        files: Dictionary mapping filenames to their contents (must include pipeline_wrapper.py)
        app: Optional FastAPI application instance. If provided, the API route will be added.
        save_files: Whether to save the pipeline files to disk permanently
        overwrite: Whether to overwrite an existing pipeline

    Returns:
        A dictionary containing the deployed pipeline name, e.g. {"name": pipeline_name}.

    Raises:
        PipelineAlreadyExistsError: If the pipeline exists and overwrite is False.
        PipelineFilesError: If saving files fails.
        PipelineModuleLoadError: If loading the pipeline module fails.
        PipelineWrapperError: If wrapper creation or setup fails.
    """
    tmp_dir = None

    # Save files to disk (required for module loading)
    if save_files:
        save_pipeline_files(pipeline_name, files=files, pipelines_dir=settings.pipelines_dir)
        pipeline_dir = Path(settings.pipelines_dir) / pipeline_name
    else:
        # Use temporary directory if not saving permanently
        tmp_dir = tempfile.mkdtemp()
        save_pipeline_files(pipeline_name, files=files, pipelines_dir=tmp_dir)
        pipeline_dir = Path(tmp_dir) / pipeline_name

    try:
        # Load module and create wrapper instance
        module = load_pipeline_module(pipeline_name, dir_path=pipeline_dir)
        pipeline_wrapper = create_pipeline_wrapper_instance(module)

        # Use shared helper for registration
        return _register_and_deploy_pipeline(
            pipeline_name=pipeline_name,
            pipeline_wrapper=pipeline_wrapper,
            app=app,
            overwrite=overwrite,
        )
    finally:
        # Clean up temp directory if used
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def deploy_pipeline_yaml(
    pipeline_name: str,
    source_code: str,
    app: FastAPI | None = None,
    overwrite: bool = False,
    options: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Deploy a YAML pipeline to the FastAPI application with IO declared in the YAML.

    This will create a YAMLPipelineWrapper, add it to the registry, and set up the
    API route at /{pipeline_name}/run.

    Args:
        pipeline_name: Name of the pipeline
        source_code: YAML pipeline source code
        app: Optional FastAPI application instance. If provided, the API route will be added.
        overwrite: Whether to overwrite an existing pipeline
        options: Optional dict with additional deployment options. Supported keys:
            - save_file: bool | None - whether to persist the YAML to disk (default: True)
            - description: str | None
            - skip_mcp: bool | None

    Returns:
        A dictionary containing the deployed pipeline name, e.g. {"name": pipeline_name}.

    Raises:
        PipelineAlreadyExistsError: If the pipeline exists and overwrite is False.
        ValueError: If the YAML cannot be parsed into an AsyncPipeline.
        InvalidYamlIOError: If the YAML is missing inputs/outputs declarations.
    """
    # Optionally save YAML to disk as pipelines/{name}.yml (default True)
    save_file: bool = True if options is None else bool(options.get("save_file", True))
    if save_file:
        save_pipeline_files(pipeline_name, {f"{pipeline_name}.yml": source_code}, settings.pipelines_dir)

    # Create YAMLPipelineWrapper from source code
    description = (options or {}).get("description")
    pipeline_wrapper = YAMLPipelineWrapper.from_yaml(source_code, description=description)

    # Set skip_mcp from options
    skip_mcp = (options or {}).get("skip_mcp", False)
    pipeline_wrapper.skip_mcp = bool(skip_mcp)

    # YAML-specific metadata
    extra_metadata = {
        "description": description or pipeline_name,
        "streaming_components": pipeline_wrapper.streaming_components,
        "include_outputs_from": pipeline_wrapper.include_outputs_from,
        "input_resolutions": pipeline_wrapper.input_resolutions,
    }

    return _register_and_deploy_pipeline(
        pipeline_name=pipeline_name,
        pipeline_wrapper=pipeline_wrapper,
        app=app,
        overwrite=overwrite,
        extra_metadata=extra_metadata,
    )


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
            log.warning("Skipping file '{}': {}", file_path, e)
            continue

    return files


def undeploy_pipeline(pipeline_name: str, app: FastAPI | None = None) -> None:
    """
    Undeploy a pipeline.

    Removes a pipeline from the registry, removes its API routes, cleans up sys.modules,
    and deletes its files from disk.

    Args:
        pipeline_name: Name of the pipeline to undeploy.
        app: Optional FastAPI application instance. If provided, API routes will be removed.

    Raises:
        HTTPException: If the pipeline is not found in the registry (404).
    """
    # Check if pipeline exists in registry
    if pipeline_name not in registry.get_names():
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")

    # Remove pipeline from registry
    registry.remove(pipeline_name)

    # Clean up sys.modules for wrapper-based pipelines
    unload_pipeline_modules(pipeline_name)

    if app:
        # Remove API routes for the pipeline
        # All pipelines have a run endpoint at /<pipeline_name>/run
        routes_to_remove = [
            route for route in app.routes if isinstance(route, APIRoute) and route.path == f"/{pipeline_name}/run"
        ]
        for route in routes_to_remove:
            app.routes.remove(route)

        # Invalidate OpenAPI cache
        app.openapi_schema = None
        app.setup()

    # Remove pipeline files if they exist
    remove_pipeline_files(pipeline_name, settings.pipelines_dir)
