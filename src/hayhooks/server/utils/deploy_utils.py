import asyncio
import inspect
import json
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import AsyncGenerator, Callable, Generator
from functools import wraps
from pathlib import Path
from typing import Any, cast

import docstring_parser
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.datastructures import Headers

from hayhooks.server.exceptions import (
    PipelineAlreadyExistsError,
    PipelineFilesError,
    PipelineModeError,
    PipelineRollbackError,
)
from hayhooks.server.logger import log, log_elapsed
from hayhooks.server.pipelines.loader import YAML_SUFFIXES
from hayhooks.server.pipelines.models import (
    create_pipeline_metadata,
    create_request_model_from_callable,
    create_response_model_from_callable,
    get_response_class_from_callable,
    get_run_api_method,
)
from hayhooks.server.pipelines.registry import ImmutablePipelineRegistry, registry
from hayhooks.server.pipelines.sse import SSEStream
from hayhooks.server.tracing import (
    SPAN_PIPELINE_DEPLOY,
    SPAN_PIPELINE_DEPLOY_COMMIT,
    SPAN_PIPELINE_DEPLOY_PREPARE,
    SPAN_PIPELINE_RUN,
    SPAN_PIPELINE_UNDEPLOY,
    build_streaming_trace_tags,
    build_trace_tags,
    trace_async_stream,
    trace_operation,
    trace_sync_stream,
)
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.models import PreparedPipeline
from hayhooks.server.utils.module_loader import (
    create_pipeline_wrapper_instance,
    load_pipeline_module,
    pipeline_modules,
    reject_durable_wrapper,
    unload_pipeline_modules,
)
from hayhooks.server.utils.request_headers import accepts_request_headers
from hayhooks.server.utils.streaming_response_utils import _streaming_response_from_result
from hayhooks.server.utils.yaml_pipeline_wrapper import YAMLPipelineWrapper
from hayhooks.settings import DeployConcurrencyPolicy, settings

# threading.Lock (not asyncio.Lock) because it's only acquired inside worker
# threads spawned by asyncio.to_thread, so never on the event loop itself.
_deploy_lock = threading.Lock()


def _with_deploy_lock(func: Callable) -> Callable:
    """Wrap *func* so it acquires ``_deploy_lock`` before executing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _deploy_lock:
            return func(*args, **kwargs)

    return wrapper


def require_live_deployment(app: FastAPI | None = None) -> None:
    """
    Reject a pipeline mutation in durable mode, where the pipeline set is fixed at startup.

    Raises:
        PipelineModeError: If durable mode is enabled or *app* serves an immutable registry.
    """
    if settings.durable_mode or (
        app is not None and isinstance(getattr(app.state, "pipeline_registry", None), ImmutablePipelineRegistry)
    ):
        msg = (
            "Pipelines cannot be deployed or undeployed in durable mode (HAYHOOKS_DURABLE_MODE); "
            "change the pipelines directory and restart the server instead"
        )
        raise PipelineModeError(msg)


async def _offload(func: Callable, **kwargs: Any) -> Any:
    """Run *func* in a thread, applying the deploy lock if policy is SERIALIZED."""
    if settings.deploy_concurrency == DeployConcurrencyPolicy.SERIALIZED:
        func = _with_deploy_lock(func)
    return await asyncio.to_thread(func, **kwargs)


async def deploy_pipeline_yaml_async(
    pipeline_name: str,
    source_code: str,
    app: FastAPI | None = None,
    overwrite: bool = False,
    options: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Async wrapper that offloads ``deploy_pipeline_yaml`` off the event loop.

    Respects the ``deploy_concurrency`` setting: when *serialized* (default), a
    global lock ensures only one deploy/undeploy runs at a time; when *parallel*,
    the call runs in a thread without serialization.
    """
    return await _offload(
        deploy_pipeline_yaml,
        pipeline_name=pipeline_name,
        source_code=source_code,
        app=app,
        overwrite=overwrite,
        options=options,
    )


async def deploy_pipeline_files_async(
    pipeline_name: str,
    files: dict[str, str],
    app: FastAPI | None = None,
    save_files: bool = True,
    overwrite: bool = False,
) -> dict[str, str]:
    """Async wrapper that offloads ``deploy_pipeline_files`` off the event loop."""
    return await _offload(
        deploy_pipeline_files,
        pipeline_name=pipeline_name,
        files=files,
        app=app,
        save_files=save_files,
        overwrite=overwrite,
    )


async def undeploy_pipeline_async(pipeline_name: str, app: FastAPI | None = None) -> None:
    """Async wrapper that offloads ``undeploy_pipeline`` off the event loop."""
    return await _offload(undeploy_pipeline, pipeline_name=pipeline_name, app=app)


def _is_single_yaml_file(files: dict[str, str]) -> bool:
    """Check if files dict represents a single YAML pipeline file."""
    if len(files) != 1:
        return False
    filename = next(iter(files.keys()))
    return filename.endswith(YAML_SUFFIXES)


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
        PipelineModeError: In durable mode
    """
    require_live_deployment()
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

    Raises:
        PipelineModeError: In durable mode
    """
    require_live_deployment()
    pipeline_dir, *yaml_files = _pipeline_source_paths(Path(pipelines_dir), pipeline_name)
    shutil.rmtree(pipeline_dir, ignore_errors=True)
    for path in yaml_files:
        path.unlink(missing_ok=True)


def _pipeline_source_paths(pipelines_dir: Path, pipeline_name: str) -> tuple[Path, ...]:
    """Return a pipeline's persisted source paths: its wrapper directory first, then its YAML files."""
    return (pipelines_dir / pipeline_name, *(pipelines_dir / f"{pipeline_name}{ext}" for ext in YAML_SUFFIXES))


def _backup_pipeline_files(pipeline_name: str) -> Path:
    """Move a pipeline's persisted source aside so a failed deployment can restore it."""
    pipelines_dir = Path(settings.pipelines_dir)
    pipelines_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=f".{pipeline_name}-", dir=pipelines_dir))
    try:
        for path in _pipeline_source_paths(pipelines_dir, pipeline_name):
            if path.exists():
                path.replace(backup_dir / path.name)
    except BaseException:
        _restore_pipeline_files(pipeline_name, str(pipelines_dir), backup_dir, remove_candidate=False)
        _cleanup_pipeline_backup(pipeline_name, backup_dir, rolled_back=True)
        raise
    return backup_dir


def _restore_pipeline_files(
    pipeline_name: str, pipelines_dir: str, backup_dir: Path, *, remove_candidate: bool = True
) -> None:
    """Best-effort rollback that retains any backup files it cannot restore."""
    clog = log.bind(pipeline_name=pipeline_name, backup_dir=str(backup_dir))
    if remove_candidate:
        try:
            remove_pipeline_files(pipeline_name, pipelines_dir)
        except BaseException as error:
            clog.bind(exception_type=type(error).__name__).error("Failed to remove candidate pipeline files")
    try:
        paths = tuple(backup_dir.iterdir())
    except BaseException as error:
        clog.bind(exception_type=type(error).__name__).error("Failed to read pipeline backup")
        return
    for path in paths:
        try:
            path.replace(Path(pipelines_dir) / path.name)
        except BaseException as error:
            clog.bind(exception_type=type(error).__name__, path=str(path)).error("Failed to restore pipeline backup")


def _cleanup_pipeline_backup(pipeline_name: str, backup_dir: Path | None, rolled_back: bool) -> None:
    if backup_dir is None:
        return
    if not rolled_back:
        shutil.rmtree(backup_dir, ignore_errors=True)
        return
    try:
        backup_dir.rmdir()
    except OSError:
        log.bind(pipeline_name=pipeline_name, backup_dir=str(backup_dir)).error("Retaining incomplete pipeline backup")


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


async def _execute_pipeline_run(
    pipeline_wrapper: BasePipelineWrapper,
    payload: dict[str, Any],
    *,
    headers: Headers | None = None,
) -> Any:
    method = (
        pipeline_wrapper.run_api_async if pipeline_wrapper._is_run_api_async_implemented else pipeline_wrapper.run_api
    )
    if accepts_request_headers(method):
        if "headers" in payload:
            msg = "Request headers cannot be supplied as pipeline arguments"
            raise ValueError(msg)
        payload = {**payload, "headers": headers}
    if pipeline_wrapper._is_run_api_async_implemented:
        return await method(**payload)
    return await run_in_threadpool(method, **payload)


_SENSITIVE_KEY_PATTERNS = {
    "api_key",
    "token",
    "authorization",
    "password",
    "secret",
    "key",
    "credential",
    "passwd",
    "access_key",
    "secret_key",
    "api_token",
    "auth_token",
}


def _payload_key_is_sensitive(key: str) -> bool:
    lower = key.lower()
    return any(pattern in lower for pattern in _SENSITIVE_KEY_PATTERNS)


def _payload_value_to_safe_text(value: Any) -> str:  # noqa: PLR0911
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return f"str({len(value)})"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list | tuple | set):
        return f"list({len(value)})"
    if isinstance(value, dict):
        return f"dict({len(value)})"
    type_name = type(value).__name__
    try:
        size = len(value)
        return f"{type_name}({size})"
    except TypeError:
        return type_name


def _payload_value_to_trace_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str | int | float):
        return str(value)
    if isinstance(value, set):
        value = sorted(value, key=str)
    elif isinstance(value, tuple):
        value = list(value)
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _build_payload_value_tags(payload: dict[str, Any]) -> list[str]:
    include_values = settings.dashboard_trace_include_payload_values
    tags: list[str] = []
    for key, value in sorted(payload.items()):
        if include_values:
            if _payload_key_is_sensitive(key):
                tags.append(f"{key}=[redacted]")
            else:
                tags.append(f"{key}={_payload_value_to_trace_text(value)}")
        else:
            tags.append(f"{key}={_payload_value_to_safe_text(value)}")
    return tags


def _build_run_trace_tags(pipeline_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return build_trace_tags(
        {
            "hayhooks.transport": "rest",
            "hayhooks.pipeline.name": pipeline_name,
            "hayhooks.route": f"/{pipeline_name}/run",
            "hayhooks.payload.values": _build_payload_value_tags(payload),
            "hayhooks.payload.has_files": "files" in payload,
        }
    )


async def _execute_pipeline_run_with_tracing(
    pipeline_wrapper: BasePipelineWrapper,
    payload: dict[str, Any],
    *,
    headers: Headers,
    trace_tags: dict[str, Any],
    is_streaming_response: bool,
) -> Any:
    if is_streaming_response:
        try:
            return await _execute_pipeline_run(pipeline_wrapper, payload, headers=headers)
        except BaseException:
            with trace_operation(SPAN_PIPELINE_RUN, tags=trace_tags):
                raise

    with trace_operation(SPAN_PIPELINE_RUN, tags=trace_tags):
        return await _execute_pipeline_run(pipeline_wrapper, payload, headers=headers)


def _trace_streaming_run_result(result: Any, trace_tags: dict[str, Any]) -> Any:
    if isinstance(result, SSEStream):
        if isinstance(result.stream, AsyncGenerator):
            return SSEStream(
                trace_async_stream(
                    result.stream,
                    SPAN_PIPELINE_RUN,
                    tags=build_streaming_trace_tags(trace_tags, stream_type="sse"),
                )
            )
        if isinstance(result.stream, Generator):
            return SSEStream(
                trace_sync_stream(
                    result.stream,
                    SPAN_PIPELINE_RUN,
                    tags=build_streaming_trace_tags(trace_tags, stream_type="sse"),
                )
            )
        return result

    if isinstance(result, AsyncGenerator):
        return trace_async_stream(
            result, SPAN_PIPELINE_RUN, tags=build_streaming_trace_tags(trace_tags, stream_type="plain")
        )
    if isinstance(result, Generator):
        return trace_sync_stream(
            result, SPAN_PIPELINE_RUN, tags=build_streaming_trace_tags(trace_tags, stream_type="plain")
        )
    return result


def create_run_endpoint_handler(
    pipeline_wrapper: BasePipelineWrapper,
    pipeline_name: str,
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
        pipeline_name: Name of the pipeline (used for logging)
        request_model: The request model
        response_model: The response model, or None for streaming/file response endpoints
        requires_files: Whether the pipeline requires file uploads

    Returns:
        A FastAPI endpoint function that executes the pipeline and returns the response model.
    """
    run_method = get_run_api_method(pipeline_wrapper)
    is_streaming_response = run_method is not None and get_response_class_from_callable(run_method) is StreamingResponse

    async def _handle_request(run_req: BaseModel, request: Request) -> Response | BaseModel:
        payload = run_req.model_dump()
        trace_tags = _build_run_trace_tags(pipeline_name, payload)

        log.bind(params=payload).opt(colors=True).info("Running pipeline '<bold>{}</bold>'", pipeline_name)
        t0 = time.monotonic()
        result = await _execute_pipeline_run_with_tracing(
            pipeline_wrapper,
            payload,
            headers=request.headers,
            trace_tags=trace_tags,
            is_streaming_response=is_streaming_response,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        traced_result = _trace_streaming_run_result(result, trace_tags)

        streaming_response = _streaming_response_from_result(traced_result)
        if streaming_response is not None:
            log.opt(colors=True).info(
                "Pipeline '<bold>{}</bold>' streaming response started (<bold>{:.0f}ms</bold>)",
                pipeline_name,
                elapsed_ms,
            )
            return streaming_response

        log.opt(colors=True).info(
            "Pipeline '<bold>{}</bold>' completed in <bold>{:.0f}ms</bold>",
            pipeline_name,
            elapsed_ms,
        )

        # response_model is None for streaming/file response endpoints, where
        # _streaming_response_from_result() always handles the result above.
        # For normal JSON endpoints, wrap the result in the Pydantic response model.
        if response_model is None:
            return cast(Response | BaseModel, traced_result)

        # response_model is built dynamically via create_model(..., result=(...)); cast to Any so ty does not
        # treat the real `result` field as an extra, discarded argument.
        response_instance = cast("Any", response_model)(result=traced_result)
        return cast(Response | BaseModel, response_instance)

    @handle_pipeline_exceptions()
    async def run_endpoint_with_files(
        request: Request,
        run_req: request_model = Form(..., media_type="multipart/form-data"),  # ty: ignore[invalid-type-form] # noqa: B008
    ) -> response_model:  # ty: ignore[invalid-type-form]
        return await _handle_request(run_req, request)

    @handle_pipeline_exceptions()
    async def run_endpoint_without_files(
        request: Request,
        run_req: request_model,  # ty: ignore[invalid-type-form]
    ) -> response_model:  # ty: ignore[invalid-type-form]
        return await _handle_request(run_req, request)

    return run_endpoint_with_files if requires_files else run_endpoint_without_files


def _build_run_route(
    pipeline_name: str, pipeline_wrapper: BasePipelineWrapper, *, request_model: type[BaseModel] | None = None
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """
    Build the ``add_api_route`` arguments of the ordinary /{pipeline_name}/run endpoint.

    Pass the ``request_model`` that ``create_pipeline_metadata`` built to reuse it instead of building another.

    Returns:
        The route arguments and the request/response metadata they derive, or ``None`` when the
        wrapper implements neither ``run_api`` nor ``run_api_async``.
    """
    clog = log.bind(pipeline_name=pipeline_name)
    run_method_to_inspect = get_run_api_method(pipeline_wrapper)
    if run_method_to_inspect is None:
        return None

    docstring_content = inspect.getdoc(run_method_to_inspect) or ""
    docstring = docstring_parser.parse(docstring_content)
    RunRequest = request_model or create_request_model_from_callable(
        run_method_to_inspect, f"{pipeline_name}Run", docstring
    )
    RunResponse = create_response_model_from_callable(run_method_to_inspect, f"{pipeline_name}Run", docstring)
    RunResponseClass = get_response_class_from_callable(run_method_to_inspect)

    run_api_params = inspect.signature(run_method_to_inspect).parameters
    requires_files = "files" in run_api_params
    clog.debug("Pipeline requires files: {}", requires_files)

    run_endpoint = create_run_endpoint_handler(
        pipeline_wrapper=pipeline_wrapper,
        pipeline_name=pipeline_name,
        request_model=RunRequest,
        response_model=RunResponse,
        requires_files=requires_files,
    )

    # Build the route kwargs. response_class is only set for non-JSON endpoints
    # (e.g. FileResponse for file downloads, StreamingResponse for generators) so that
    # OpenAPI docs show the correct Content-Type. For normal JSON endpoints we omit it
    # and let FastAPI use its default JSONResponse.
    route_kwargs: dict[str, Any] = {
        "path": f"/{pipeline_name}/run",
        "endpoint": run_endpoint,
        "methods": ["POST"],
        "name": f"{pipeline_name}_run",
        "response_model": RunResponse,
        "tags": ["pipelines"],
        "description": docstring.short_description or None,
    }
    if RunResponseClass is not None:
        route_kwargs["response_class"] = RunResponseClass
    return route_kwargs, {"request_model": RunRequest, "response_model": RunResponse, "requires_files": requires_files}


def add_pipeline_api_route(
    app: FastAPI,
    pipeline_name: str,
    pipeline_wrapper: BasePipelineWrapper,
    *,
    _defer_openapi_rebuild: bool = False,
) -> None:
    """
    Create or replace the wrapper-based pipeline run endpoint at /{pipeline_name}/run.

    Args:
        app: FastAPI application instance.
        pipeline_name: Name of the pipeline.
        pipeline_wrapper: Initialized pipeline wrapper instance to use as handler target.
        _defer_openapi_rebuild: When True, skip ``app.setup()`` / schema invalidation so
            the caller can batch multiple route additions and rebuild once at the end.

    Side Effects:
        - Removes any existing route at /{pipeline_name}/run
        - Rebuilds and invalidates the OpenAPI schema (unless ``_defer_openapi_rebuild``)
        - Updates registry metadata with request/response models and file requirement flag

    Raises:
        PipelineModeError: In durable mode, before any route or metadata changes.
    """
    require_live_deployment(app)
    metadata = registry.get_metadata(pipeline_name) or {}
    route = _build_run_route(pipeline_name, pipeline_wrapper, request_model=metadata.get("request_model"))
    if route is None:
        # If neither run_api nor run_api_async is implemented,
        # this pipeline will not have a generic /<pipeline_name>/run endpoint.
        # This is a valid configuration (e.g., for chat-only pipelines).
        log.bind(pipeline_name=pipeline_name).warning(
            f"Pipeline '{pipeline_name}' does not implement `run_api` or `run_api_async`. "
            f"Skipping /{pipeline_name}/run API route creation."
        )
        return
    route_kwargs, route_metadata = route

    _remove_pipeline_routes(app, pipeline_name)
    app.add_api_route(**route_kwargs)
    registry.update_metadata(pipeline_name, route_metadata)

    if not _defer_openapi_rebuild:
        log.bind(pipeline_name=pipeline_name).debug("Setting up FastAPI app")
        rebuild_openapi(app)


def rebuild_openapi(app: FastAPI) -> None:
    """Invalidate and rebuild the OpenAPI schema for *app*."""
    app.openapi_schema = None
    app.setup()


def _remove_pipeline_routes(app: FastAPI, pipeline_name: str) -> None:
    for route in tuple(app.routes):
        if isinstance(route, APIRoute) and route.path == f"/{pipeline_name}/run":
            app.routes.remove(route)


def _register_prepared_pipeline(
    pipeline_name: str,
    pipeline_wrapper: BasePipelineWrapper,
    app: FastAPI | None = None,
    extra_metadata: dict[str, Any] | None = None,
    *,
    _defer_openapi_rebuild: bool = False,
) -> dict[str, str]:
    """
    Register a prepared pipeline wrapper and optionally add its API route.

    Commit-level overwrite handling must happen before this function is called.

    Args:
        pipeline_name: Name of the pipeline.
        pipeline_wrapper: Already-prepared wrapper instance.
        app: Optional FastAPI app for route creation.
        extra_metadata: Additional metadata fields (e.g., streaming_components for YAML).
        _defer_openapi_rebuild: Forward to ``add_pipeline_api_route`` to skip per-pipeline
            OpenAPI rebuild during batch operations.

    Returns:
        A dictionary containing the deployed pipeline name, e.g. {"name": pipeline_name}.

    Raises:
        PipelineAlreadyExistsError: If the pipeline already exists at commit time.
    """
    clog = log.bind(pipeline_name=pipeline_name)

    # Commit resolves overwrite semantics before registration.
    if registry.get(pipeline_name):
        msg = f"Pipeline '{pipeline_name}' already exists"
        raise PipelineAlreadyExistsError(msg)

    metadata = create_pipeline_metadata(pipeline_name, pipeline_wrapper)

    # Merge extra metadata (e.g., YAML-specific fields)
    if extra_metadata:
        metadata.update(extra_metadata)

    clog.debug("Adding pipeline to registry with metadata: {}", metadata)
    registry.add(pipeline_name, pipeline_wrapper, metadata=metadata)
    clog.success("Pipeline '{}' successfully added to registry", pipeline_name)

    if app:
        try:
            add_pipeline_api_route(app, pipeline_name, pipeline_wrapper, _defer_openapi_rebuild=_defer_openapi_rebuild)
        except BaseException:
            try:
                _remove_pipeline_routes(app, pipeline_name)
            except BaseException as error:
                clog.bind(exception_type=type(error).__name__).error("Failed to remove candidate pipeline routes")
            registry.remove(pipeline_name)
            raise

    return {"name": pipeline_name}


@log_elapsed()
def prepare_pipeline_files(
    pipeline_name: str,
    files: dict[str, str],
    save_files: bool = True,
) -> PreparedPipeline:
    """
    Prepare a files-based pipeline without mutating registry or routes.

    Does file I/O, module loading, wrapper creation, and ``setup()`` — all the
    expensive work that is safe to run in a thread.  The returned
    ``PreparedPipeline`` can be committed later via ``_register_prepared_pipeline``.

    Raises:
        PipelineModeError: In durable mode, or if the wrapper is durable.
    """
    require_live_deployment()
    with trace_operation(
        SPAN_PIPELINE_DEPLOY_PREPARE,
        tags=build_trace_tags(
            {
                "hayhooks.transport": "runtime",
                "hayhooks.pipeline.name": pipeline_name,
                "hayhooks.pipeline.source_type": "files",
                "hayhooks.deploy.save_files": save_files,
                "hayhooks.deploy.file_count": len(files),
            }
        ),
    ):
        tmp_dir = None

        if save_files:
            save_pipeline_files(pipeline_name, files=files, pipelines_dir=settings.pipelines_dir)
            pipeline_dir = Path(settings.pipelines_dir) / pipeline_name
        else:
            tmp_dir = tempfile.mkdtemp()
            save_pipeline_files(pipeline_name, files=files, pipelines_dir=tmp_dir)
            pipeline_dir = Path(tmp_dir) / pipeline_name

        try:
            module = load_pipeline_module(pipeline_name, dir_path=pipeline_dir)
            pipeline_wrapper = create_pipeline_wrapper_instance(module, allow_durable=False)
            return PreparedPipeline(name=pipeline_name, wrapper=pipeline_wrapper)
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)


@log_elapsed()
def prepare_pipeline_yaml(
    pipeline_name: str,
    source_code: str,
    options: dict[str, Any] | None = None,
) -> PreparedPipeline:
    """
    Prepare a YAML pipeline without mutating registry or routes.

    Does file I/O, YAML parsing, wrapper creation, and ``setup()`` — all the
    expensive work that is safe to run in a thread.

    Raises:
        PipelineModeError: In durable mode.
    """
    require_live_deployment()
    save_file: bool = True if options is None else bool(options.get("save_file", True))
    description = (options or {}).get("description")
    skip_mcp = bool((options or {}).get("skip_mcp", False))

    with trace_operation(
        SPAN_PIPELINE_DEPLOY_PREPARE,
        tags=build_trace_tags(
            {
                "hayhooks.transport": "runtime",
                "hayhooks.pipeline.name": pipeline_name,
                "hayhooks.pipeline.source_type": "yaml",
                "hayhooks.deploy.save_file": save_file,
                "hayhooks.deploy.skip_mcp": skip_mcp,
            }
        ),
    ):
        if save_file:
            save_pipeline_files(pipeline_name, {f"{pipeline_name}.yml": source_code}, settings.pipelines_dir)

        pipeline_wrapper = YAMLPipelineWrapper.from_yaml(source_code, description=description)
        pipeline_wrapper.skip_mcp = skip_mcp
        pipeline_wrapper.setup()

        extra_metadata = {
            "description": description or pipeline_name,
            "streaming_components": pipeline_wrapper.streaming_components,
            "include_outputs_from": pipeline_wrapper.include_outputs_from,
            "input_resolutions": pipeline_wrapper.input_resolutions,
        }

        return PreparedPipeline(name=pipeline_name, wrapper=pipeline_wrapper, extra_metadata=extra_metadata)


def _restore_replaced_pipeline(
    pipeline_name: str,
    pipeline_wrapper: BasePipelineWrapper,
    metadata: dict[str, Any] | None,
    app: FastAPI | None,
    error: BaseException,
    *,
    _defer_openapi_rebuild: bool,
) -> None:
    """Republish the pipeline a failed commit removed, reporting when that fails too."""
    try:
        registry.add(pipeline_name, pipeline_wrapper, metadata=metadata)
        if app:
            add_pipeline_api_route(app, pipeline_name, pipeline_wrapper, _defer_openapi_rebuild=_defer_openapi_rebuild)
    except Exception as restore_error:
        msg = f"{error!s}; restoring the previous pipeline '{pipeline_name}' also failed: {restore_error!s}"
        raise PipelineRollbackError(msg) from error


def commit_prepared_pipeline(
    prepared: PreparedPipeline,
    app: FastAPI | None = None,
    overwrite: bool = False,
    *,
    _defer_openapi_rebuild: bool = False,
    cleanup_files_on_overwrite: bool = True,
    source_files: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Commit a prepared pipeline to the registry and (optionally) add its route.

    This mutates shared state and must NOT be called concurrently.
    The prepared wrapper has already run setup(), so commit must not run it again.

    Args:
        prepared: Pipeline prepared by ``prepare_pipeline_files`` or ``prepare_pipeline_yaml``.
        app: Optional FastAPI app for route creation.
        overwrite: Whether to replace an existing deployed pipeline with the same name.
        _defer_openapi_rebuild: Forwarded to route registration.
        cleanup_files_on_overwrite: If ``True``, remove persisted files when replacing an existing pipeline.
        source_files: Candidate source to persist atomically with the host-side publication.

    Raises:
        PipelineModeError: In durable mode, or if the prepared wrapper is durable.
        PipelineRollbackError: If the commit failed and the replaced pipeline could not be restored.
    """
    require_live_deployment(app)
    with trace_operation(
        SPAN_PIPELINE_DEPLOY_COMMIT,
        tags=build_trace_tags(
            {
                "hayhooks.transport": "runtime",
                "hayhooks.pipeline.name": prepared.name,
                "hayhooks.deploy.overwrite": overwrite,
                "hayhooks.deploy.with_fastapi_route": app is not None,
                "hayhooks.deploy.defer_openapi_rebuild": _defer_openapi_rebuild,
            }
        ),
    ):
        reject_durable_wrapper(prepared.wrapper)
        old_wrapper = registry.get(prepared.name)
        old_metadata = registry.get_metadata(prepared.name)
        if old_wrapper is not None and not overwrite:
            msg = f"Pipeline '{prepared.name}' already exists"
            raise PipelineAlreadyExistsError(msg)

        old_removed = False
        backup_dir: Path | None = None
        rolled_back = False
        try:
            if source_files is not None or (old_wrapper is not None and cleanup_files_on_overwrite):
                backup_dir = _backup_pipeline_files(prepared.name)

            if source_files is not None:
                save_pipeline_files(prepared.name, source_files, settings.pipelines_dir)

            if old_wrapper is not None:
                log.bind(pipeline_name=prepared.name).debug("Clearing existing pipeline '{}'", prepared.name)
                registry.remove(prepared.name)
                old_removed = True
                if app:
                    _remove_pipeline_routes(app, prepared.name)

            return _register_prepared_pipeline(
                pipeline_name=prepared.name,
                pipeline_wrapper=prepared.wrapper,
                app=app,
                extra_metadata=prepared.extra_metadata,
                _defer_openapi_rebuild=_defer_openapi_rebuild,
            )
        except BaseException as error:
            rolled_back = True
            if backup_dir is not None:
                _restore_pipeline_files(prepared.name, settings.pipelines_dir, backup_dir)
            if old_removed and old_wrapper is not None:
                _restore_replaced_pipeline(
                    prepared.name, old_wrapper, old_metadata, app, error, _defer_openapi_rebuild=_defer_openapi_rebuild
                )
            raise
        finally:
            _cleanup_pipeline_backup(prepared.name, backup_dir, rolled_back)


def deploy_pipeline_files(
    pipeline_name: str,
    files: dict[str, str],
    app: FastAPI | None = None,
    save_files: bool = True,
    overwrite: bool = False,
    *,
    _defer_openapi_rebuild: bool = False,
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
        _defer_openapi_rebuild: Skip per-pipeline OpenAPI rebuild (for batch startup).

    Returns:
        A dictionary containing the deployed pipeline name, e.g. {"name": pipeline_name}.

    Raises:
        PipelineAlreadyExistsError: If the pipeline exists and overwrite is False.
        PipelineFilesError: If saving files fails.
        PipelineModuleLoadError: If loading the pipeline module fails.
        PipelineWrapperError: If wrapper creation or setup fails.
        PipelineModeError: In durable mode, or if the wrapper is durable.
        PipelineRollbackError: If the deployment failed and the replaced pipeline could not be restored.
    """
    require_live_deployment(app)
    with trace_operation(
        SPAN_PIPELINE_DEPLOY,
        tags=build_trace_tags(
            {
                "hayhooks.transport": "runtime",
                "hayhooks.pipeline.name": pipeline_name,
                "hayhooks.pipeline.source_type": "files",
                "hayhooks.deploy.save_files": save_files,
                "hayhooks.deploy.overwrite": overwrite,
                "hayhooks.deploy.with_fastapi_route": app is not None,
            }
        ),
    ):
        if registry.get(pipeline_name) is not None and not overwrite:
            msg = f"Pipeline '{pipeline_name}' already exists"
            raise PipelineAlreadyExistsError(msg)

        old_modules = pipeline_modules(pipeline_name)
        backup_dir: Path | None = None
        rolled_back = False
        try:
            if save_files:
                backup_dir = _backup_pipeline_files(pipeline_name)
            prepared = prepare_pipeline_files(pipeline_name, files=files, save_files=save_files)
            return commit_prepared_pipeline(
                prepared,
                app=app,
                overwrite=overwrite,
                _defer_openapi_rebuild=_defer_openapi_rebuild,
                cleanup_files_on_overwrite=overwrite and not save_files,
            )
        except BaseException:
            if backup_dir is not None:
                rolled_back = True
                _restore_pipeline_files(pipeline_name, settings.pipelines_dir, backup_dir)
            unload_pipeline_modules(pipeline_name)
            sys.modules.update(old_modules)
            raise
        finally:
            _cleanup_pipeline_backup(pipeline_name, backup_dir, rolled_back)


def deploy_pipeline_yaml(
    pipeline_name: str,
    source_code: str,
    app: FastAPI | None = None,
    overwrite: bool = False,
    options: dict[str, Any] | None = None,
    *,
    _defer_openapi_rebuild: bool = False,
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
        _defer_openapi_rebuild: Skip per-pipeline OpenAPI rebuild (for batch startup).

    Returns:
        A dictionary containing the deployed pipeline name, e.g. {"name": pipeline_name}.

    Raises:
        PipelineAlreadyExistsError: If the pipeline exists and overwrite is False.
        ValueError: If the YAML cannot be parsed into a Pipeline.
        InvalidYamlIOError: If the YAML is missing inputs/outputs declarations.
        PipelineModeError: In durable mode.
    """
    require_live_deployment(app)
    with trace_operation(
        SPAN_PIPELINE_DEPLOY,
        tags=build_trace_tags(
            {
                "hayhooks.transport": "runtime",
                "hayhooks.pipeline.name": pipeline_name,
                "hayhooks.pipeline.source_type": "yaml",
                "hayhooks.deploy.overwrite": overwrite,
                "hayhooks.deploy.with_fastapi_route": app is not None,
                "hayhooks.deploy.skip_mcp": (options or {}).get("skip_mcp"),
            }
        ),
    ):
        save_file = True if options is None else bool(options.get("save_file", True))
        prepared = prepare_pipeline_yaml(
            pipeline_name,
            source_code=source_code,
            options={**(options or {}), "save_file": False},
        )
        return commit_prepared_pipeline(
            prepared,
            app=app,
            overwrite=overwrite,
            _defer_openapi_rebuild=_defer_openapi_rebuild,
            cleanup_files_on_overwrite=overwrite and not save_file,
            source_files={f"{pipeline_name}.yml": source_code} if save_file else None,
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


def deploy_pipelines() -> None:
    """
    Deploy pipelines from the configured directory.

    Raises:
        PipelineModeError: In durable mode, or if a wrapper is durable.
    """
    require_live_deployment()
    # Imported here to avoid a circular import (hayhooks.server.app imports this module)
    from hayhooks.server.app import init_pipeline_dir

    pipelines_dir = init_pipeline_dir(settings.pipelines_dir)

    log.info("Pipelines dir set to: '{}'", pipelines_dir)
    pipelines_path = Path(pipelines_dir)

    pipeline_dirs = [d for d in pipelines_path.iterdir() if d.is_dir()]
    log.debug("Found {} pipeline directories", len(pipeline_dirs))

    for pipeline_dir in pipeline_dirs:
        log.debug("Deploying pipeline from '{}'", pipeline_dir)

        try:
            deploy_pipeline_files(
                pipeline_name=pipeline_dir.name,
                files=read_pipeline_files_from_dir(pipeline_dir),
                save_files=False,  # Files already exist on disk
            )
        except PipelineModeError:
            raise
        except Exception as e:
            log.warning("Skipping pipeline directory '{}': {}", pipeline_dir, e)


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
        PipelineModeError: In durable mode.
    """
    require_live_deployment(app)
    with trace_operation(
        SPAN_PIPELINE_UNDEPLOY,
        tags=build_trace_tags(
            {
                "hayhooks.transport": "runtime",
                "hayhooks.pipeline.name": pipeline_name,
                "hayhooks.deploy.with_fastapi_route": app is not None,
            }
        ),
    ):
        # Check if pipeline exists in registry
        if pipeline_name not in registry.get_names():
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")

        # Remove pipeline from registry
        registry.remove(pipeline_name)

        # Clean up sys.modules for wrapper-based pipelines
        unload_pipeline_modules(pipeline_name)

        if app:
            _remove_pipeline_routes(app, pipeline_name)
            rebuild_openapi(app)

        # Remove pipeline files if they exist
        remove_pipeline_files(pipeline_name, settings.pipelines_dir)
