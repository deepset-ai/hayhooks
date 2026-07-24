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
from fastapi import FastAPI, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel

from hayhooks.durable.runtime import DurableDeployment, durable_runtime
from hayhooks.server.durable.routes import DURABLE_ROUTE_SUFFIXES as _DURABLE_ROUTE_SUFFIXES
from hayhooks.server.durable.routes import add_durable_api_routes as _add_durable_api_routes
from hayhooks.server.exceptions import PipelineAlreadyExistsError, PipelineFilesError
from hayhooks.server.logger import log, log_elapsed
from hayhooks.server.pipelines.lifecycle import close_pipeline_wrapper_lifecycle as _close_wrapper_lifecycle
from hayhooks.server.pipelines.lifecycle import start_pipeline_wrapper_lifecycle as _start_wrapper_lifecycle
from hayhooks.server.pipelines.models import (
    create_request_model_from_callable,
    create_response_model_from_callable,
    get_response_class_from_callable,
)
from hayhooks.server.pipelines.registry import registry
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
    unload_pipeline_modules,
)
from hayhooks.server.utils.streaming_response_utils import _streaming_response_from_result
from hayhooks.server.utils.yaml_pipeline_wrapper import YAMLPipelineWrapper
from hayhooks.settings import DeployConcurrencyPolicy, settings

# threading.Lock (not asyncio.Lock) because it's only acquired inside worker
# threads spawned by asyncio.to_thread, so never on the event loop itself.
_deploy_lock = threading.Lock()
_deployment_transaction_lock = asyncio.Lock()


def _with_deploy_lock(func: Callable) -> Callable:
    """Wrap *func* so it acquires ``_deploy_lock`` before executing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _deploy_lock:
            return func(*args, **kwargs)

    return wrapper


async def _offload(func: Callable, **kwargs: Any) -> Any:
    """Run *func* in a thread, applying the deploy lock if policy is SERIALIZED."""
    if settings.deploy_concurrency == DeployConcurrencyPolicy.SERIALIZED:
        func = _with_deploy_lock(func)
    return await asyncio.to_thread(func, **kwargs)


class _DeploymentSnapshot:
    """Rollback state captured before preparation mutates files or loaded modules."""

    def __init__(self, pipeline_name: str, app: FastAPI | None) -> None:
        self.pipeline_name = pipeline_name
        self.app = app
        self.wrapper = registry.get(pipeline_name)
        metadata = registry.get_metadata(pipeline_name)
        self.metadata = dict(metadata) if metadata is not None else None
        self.deployment = durable_runtime.current_deployment(pipeline_name)
        self.routes = list(app.routes) if app is not None else None
        self.openapi_schema = app.openapi_schema if app is not None else None
        self.modules = {
            name: module
            for name, module in sys.modules.items()
            if name == pipeline_name or name.startswith(f"{pipeline_name}.")
        }
        pipelines_dir = Path(settings.pipelines_dir)
        source_dir = pipelines_dir / pipeline_name
        sources = [source_dir] if source_dir.is_dir() else []
        sources.extend(
            source
            for extension in (".yml", ".yaml")
            if (source := pipelines_dir / f"{pipeline_name}{extension}").is_file()
        )
        self.backup_dir = Path(tempfile.mkdtemp(prefix="hayhooks-deploy-rollback-")) if sources else None
        if source_dir.is_dir() and self.backup_dir is not None:
            shutil.copytree(source_dir, self.backup_dir / "pipeline")
        for extension in (".yml", ".yaml"):
            source = pipelines_dir / f"{pipeline_name}{extension}"
            if source.is_file() and self.backup_dir is not None:
                shutil.copy2(source, self.backup_dir / f"pipeline{extension}")

    @classmethod
    def capture(cls, pipeline_name: str, app: FastAPI | None) -> "_DeploymentSnapshot":
        return cls(pipeline_name, app)

    def restore_publication(self) -> None:
        registry.remove(self.pipeline_name)
        if self.wrapper is not None:
            registry.add(self.pipeline_name, self.wrapper, metadata=dict(self.metadata or {}))
        durable_runtime.install_deployment(self.pipeline_name, self.deployment)
        if self.app is not None and self.routes is not None:
            self.app.routes[:] = self.routes
            self.app.openapi_schema = self.openapi_schema

    def restore_files_and_modules(self) -> None:
        remove_pipeline_files(self.pipeline_name, settings.pipelines_dir)
        pipelines_dir = Path(settings.pipelines_dir)
        pipelines_dir.mkdir(parents=True, exist_ok=True)
        if self.backup_dir is not None:
            backup_pipeline = self.backup_dir / "pipeline"
            if backup_pipeline.is_dir():
                shutil.copytree(backup_pipeline, pipelines_dir / self.pipeline_name)
            for extension in (".yml", ".yaml"):
                backup = self.backup_dir / f"pipeline{extension}"
                if backup.is_file():
                    shutil.copy2(backup, pipelines_dir / f"{self.pipeline_name}{extension}")

        unload_pipeline_modules(self.pipeline_name)
        sys.modules.update(self.modules)

    def cleanup(self) -> None:
        if self.backup_dir is not None:
            shutil.rmtree(self.backup_dir, ignore_errors=True)


async def _publish_prepared_pipeline(  # noqa: C901, PLR0912, PLR0915
    prepared: PreparedPipeline,
    snapshot: _DeploymentSnapshot,
    *,
    app: FastAPI | None,
    overwrite: bool,
    cleanup_files_on_overwrite: bool,
) -> dict[str, str]:
    """Start and initialize a candidate, then swap all observable deployment state."""
    if snapshot.wrapper is not None and not overwrite:
        msg = f"Pipeline '{prepared.name}' already exists"
        raise PipelineAlreadyExistsError(msg)

    candidate = durable_runtime.create_deployment(prepared.name, prepared.wrapper)
    candidate_lifecycle_started = True
    old_quiesced = False
    try:
        await _start_wrapper_lifecycle(prepared.name, prepared.wrapper)
        if candidate is not None and durable_runtime.started:
            await candidate.prepare()

        if snapshot.deployment is not None:
            snapshot.deployment.deactivate()
            old_quiesced = True
            await snapshot.deployment.close()

        old_wrapper = snapshot.wrapper
        if old_wrapper is not None and old_wrapper is not prepared.wrapper:
            retirement = candidate or snapshot.deployment
            target_revision = candidate.revision if candidate is not None else "__deployment_removed__"
            if retirement is not None:
                retired = await retirement.store.retire_incompatible(target_revision)
                if retired:
                    log.info(
                        "{} | retired {} durable execution(s) from a replaced definition",
                        prepared.name,
                        retired,
                    )

        result = commit_prepared_pipeline(
            prepared,
            app=app,
            overwrite=overwrite,
            cleanup_files_on_overwrite=cleanup_files_on_overwrite,
            _durable_deployment=candidate,
        )
        durable_runtime.install_deployment(prepared.name, candidate)
        if candidate is not None and durable_runtime.started:
            candidate.activate()
    except BaseException:
        snapshot.restore_publication()
        if candidate is not None:
            candidate.deactivate()
            await candidate.close()
        if old_quiesced and snapshot.deployment is not None and durable_runtime.started:
            await snapshot.deployment.start()
        if candidate_lifecycle_started:
            try:
                await _close_wrapper_lifecycle(prepared.name, prepared.wrapper)
            except Exception as error:
                log.opt(exception=True).warning("Error rolling back pipeline '{}' lifecycle: {}", prepared.name, error)
        raise

    old_wrapper = snapshot.wrapper
    if old_wrapper is not None and old_wrapper is not prepared.wrapper:
        old_deployment = snapshot.deployment
        retirement = candidate or old_deployment
        target_revision = candidate.revision if candidate is not None else "__deployment_removed__"

        async def retire_requeued_work_until_success() -> None:
            delay = 1.0
            while True:
                try:
                    if retirement is not None:
                        retired = await retirement.store.retire_incompatible(target_revision)
                        if retired:
                            log.info(
                                "{} | retired {} durable execution(s) from a replaced definition",
                                prepared.name,
                                retired,
                            )
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    log.opt(exception=error).warning(
                        "{} | durable post-drain revision retirement failed; retrying: {}",
                        prepared.name,
                        error,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

        async def close_old_wrapper() -> None:
            try:
                await _close_wrapper_lifecycle(prepared.name, old_wrapper)
            except Exception as error:
                log.opt(exception=True).warning("Error closing replaced pipeline '{}': {}", prepared.name, error)

        if old_deployment is not None and old_deployment.manager.draining:

            async def finish_replacement() -> None:
                await old_deployment.manager.wait_drained()
                await close_old_wrapper()
                await retire_requeued_work_until_success()

            durable_runtime.track_background_task(
                finish_replacement(),
                name=f"durable-replacement-cleanup:{prepared.name}",
            )
        else:
            await close_old_wrapper()
    return result


async def deploy_pipeline_yaml_async(
    pipeline_name: str,
    source_code: str,
    app: FastAPI | None = None,
    overwrite: bool = False,
    options: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Async wrapper that offloads ``deploy_pipeline_yaml`` off the event loop.

    Preparation respects ``deploy_concurrency``. Publication and lifecycle are
    always serialized so registry, route, file, and runtime state change together.
    """
    async with _deployment_transaction_lock:
        save_file = True if options is None else bool(options.get("save_file", True))
        snapshot = _DeploymentSnapshot.capture(pipeline_name, app)
        try:
            if overwrite and save_file:
                remove_pipeline_files(pipeline_name, settings.pipelines_dir)
            prepared = await _offload(
                prepare_pipeline_yaml,
                pipeline_name=pipeline_name,
                source_code=source_code,
                options=options,
            )
            return await _publish_prepared_pipeline(
                prepared,
                snapshot,
                app=app,
                overwrite=overwrite,
                cleanup_files_on_overwrite=overwrite and not save_file,
            )
        except BaseException:
            snapshot.restore_files_and_modules()
            raise
        finally:
            snapshot.cleanup()


async def deploy_pipeline_files_async(
    pipeline_name: str,
    files: dict[str, str],
    app: FastAPI | None = None,
    save_files: bool = True,
    overwrite: bool = False,
) -> dict[str, str]:
    """Async wrapper that offloads ``deploy_pipeline_files`` off the event loop."""
    async with _deployment_transaction_lock:
        snapshot = _DeploymentSnapshot.capture(pipeline_name, app)
        try:
            if overwrite and save_files:
                remove_pipeline_files(pipeline_name, settings.pipelines_dir)
            prepared = await _offload(
                prepare_pipeline_files,
                pipeline_name=pipeline_name,
                files=files,
                save_files=save_files,
            )
            return await _publish_prepared_pipeline(
                prepared,
                snapshot,
                app=app,
                overwrite=overwrite,
                cleanup_files_on_overwrite=overwrite and not save_files,
            )
        except BaseException:
            snapshot.restore_files_and_modules()
            raise
        finally:
            snapshot.cleanup()


async def undeploy_pipeline_async(  # noqa: C901
    pipeline_name: str,
    app: FastAPI | None = None,
) -> None:
    """Atomically unpublish a pipeline before stopping its owned resources."""
    async with _deployment_transaction_lock:
        wrapper = registry.get(pipeline_name)
        if wrapper is None:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
        deployment = durable_runtime.current_deployment(pipeline_name)
        if deployment is not None:
            deployment.deactivate()

        undeploy_pipeline(pipeline_name=pipeline_name, app=app)
        durable_runtime.install_deployment(pipeline_name, None)

        if deployment is not None:
            await deployment.close()

        async def close_wrapper() -> None:
            try:
                await _close_wrapper_lifecycle(pipeline_name, wrapper)
            except Exception as error:
                log.opt(exception=True).warning("Error closing undeployed pipeline '{}': {}", pipeline_name, error)

        if deployment is None:
            await close_wrapper()
            return

        async def retirement_pass() -> None:
            retired = await deployment.store.retire_incompatible("__deployment_removed__")
            if retired:
                log.info("{} | retired {} execution(s) during undeploy", pipeline_name, retired)

        async def retire_until_success() -> None:
            delay = 1.0
            while True:
                try:
                    await retirement_pass()
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    log.opt(exception=error).warning(
                        "{} | undeploy retirement failed; retrying: {}",
                        pipeline_name,
                        error,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

        try:
            await retirement_pass()
        except Exception as error:
            log.opt(exception=error).warning(
                "{} | initial undeploy retirement failed; retrying in background: {}",
                pipeline_name,
                error,
            )
            durable_runtime.track_background_task(
                retire_until_success(),
                name=f"durable-undeploy-retirement:{pipeline_name}",
            )

        if deployment.manager.draining:

            async def finish_undeploy() -> None:
                await deployment.manager.wait_drained()
                await close_wrapper()
                await retire_until_success()

            durable_runtime.track_background_task(
                finish_undeploy(),
                name=f"durable-undeploy-cleanup:{pipeline_name}",
            )
        else:
            await close_wrapper()


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


async def _execute_pipeline_run(
    pipeline_wrapper: BasePipelineWrapper,
    payload: dict[str, Any],
) -> Any:
    if pipeline_wrapper._is_run_api_async_implemented:
        return await pipeline_wrapper.run_api_async(**payload)
    return await run_in_threadpool(pipeline_wrapper.run_api, **payload)


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
    trace_tags: dict[str, Any],
    is_streaming_response: bool,
) -> Any:
    if is_streaming_response:
        try:
            return await _execute_pipeline_run(pipeline_wrapper, payload)
        except BaseException:
            with trace_operation(SPAN_PIPELINE_RUN, tags=trace_tags):
                raise

    with trace_operation(SPAN_PIPELINE_RUN, tags=trace_tags):
        return await _execute_pipeline_run(pipeline_wrapper, payload)


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
    run_method = (
        pipeline_wrapper.run_api_async if pipeline_wrapper._is_run_api_async_implemented else pipeline_wrapper.run_api
    )
    is_streaming_response = get_response_class_from_callable(run_method) is StreamingResponse

    async def _handle_request(run_req: BaseModel) -> Response | BaseModel:
        payload = run_req.model_dump()
        trace_tags = _build_run_trace_tags(pipeline_name, payload)

        log.bind(params=payload).opt(colors=True).info("Running pipeline '<bold>{}</bold>'", pipeline_name)
        t0 = time.monotonic()
        result = await _execute_pipeline_run_with_tracing(
            pipeline_wrapper,
            payload,
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

        return cast(Response | BaseModel, cast(Any, response_model)(result=traced_result))

    @handle_pipeline_exceptions()
    async def run_endpoint_with_files(
        run_req: request_model = Form(..., media_type="multipart/form-data"),  # ty: ignore[invalid-type-form] # noqa: B008
    ) -> response_model:  # ty: ignore[invalid-type-form]
        return await _handle_request(run_req)

    @handle_pipeline_exceptions()
    async def run_endpoint_without_files(run_req: request_model) -> response_model:  # ty: ignore[invalid-type-form]
        return await _handle_request(run_req)

    return run_endpoint_with_files if requires_files else run_endpoint_without_files


def add_pipeline_api_route(
    app: FastAPI,
    pipeline_name: str,
    pipeline_wrapper: BasePipelineWrapper,
    *,
    _defer_openapi_rebuild: bool = False,
    _durable_deployment: DurableDeployment | None = None,
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
        _add_durable_api_routes(
            app,
            pipeline_name,
            pipeline_wrapper,
            deployment=_durable_deployment,
            _defer_openapi_rebuild=_defer_openapi_rebuild,
        )
        return

    docstring_content = inspect.getdoc(run_method_to_inspect) or ""
    docstring = docstring_parser.parse(docstring_content)
    RunRequest = create_request_model_from_callable(run_method_to_inspect, f"{pipeline_name}Run", docstring)
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

    # Clear existing pipeline run route if it exists
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == f"/{pipeline_name}/run":
            app.routes.remove(route)

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

    app.add_api_route(**route_kwargs)

    _add_durable_api_routes(
        app,
        pipeline_name,
        pipeline_wrapper,
        deployment=_durable_deployment,
        _defer_openapi_rebuild=True,
    )

    registry.update_metadata(
        pipeline_name,
        {
            "request_model": RunRequest,
            "response_model": RunResponse,
            "requires_files": requires_files,
        },
    )

    if not _defer_openapi_rebuild:
        clog.debug("Setting up FastAPI app")
        app.openapi_schema = None
        app.setup()


def rebuild_openapi(app: FastAPI) -> None:
    """Invalidate and rebuild the OpenAPI schema for *app*."""
    app.openapi_schema = None
    app.setup()


def _register_prepared_pipeline(
    pipeline_name: str,
    pipeline_wrapper: BasePipelineWrapper,
    app: FastAPI | None = None,
    extra_metadata: dict[str, Any] | None = None,
    *,
    _defer_openapi_rebuild: bool = False,
    _durable_deployment: DurableDeployment | None = None,
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
        "skip_a2a": pipeline_wrapper.skip_a2a,
        "a2a_card": pipeline_wrapper.a2a_card,
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
        add_pipeline_api_route(
            app,
            pipeline_name,
            pipeline_wrapper,
            _defer_openapi_rebuild=_defer_openapi_rebuild,
            _durable_deployment=_durable_deployment,
        )

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
    """
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
            pipeline_wrapper = create_pipeline_wrapper_instance(module)
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
    """
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


def commit_prepared_pipeline(
    prepared: PreparedPipeline,
    app: FastAPI | None = None,
    overwrite: bool = False,
    *,
    _defer_openapi_rebuild: bool = False,
    cleanup_files_on_overwrite: bool = True,
    _durable_deployment: DurableDeployment | None = None,
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
    """
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
        if registry.get(prepared.name) is not None:
            if not overwrite:
                msg = f"Pipeline '{prepared.name}' already exists"
                raise PipelineAlreadyExistsError(msg)

            log.bind(pipeline_name=prepared.name).debug("Clearing existing pipeline '{}'", prepared.name)
            registry.remove(prepared.name)
            if cleanup_files_on_overwrite:
                remove_pipeline_files(prepared.name, settings.pipelines_dir)

        return _register_prepared_pipeline(
            pipeline_name=prepared.name,
            pipeline_wrapper=prepared.wrapper,
            app=app,
            extra_metadata=prepared.extra_metadata,
            _defer_openapi_rebuild=_defer_openapi_rebuild,
            _durable_deployment=_durable_deployment,
        )


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
    """
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
        cleanup_files_on_overwrite = overwrite and not save_files
        if overwrite and save_files:
            remove_pipeline_files(pipeline_name, settings.pipelines_dir)

        prepared = prepare_pipeline_files(pipeline_name, files=files, save_files=save_files)
        return commit_prepared_pipeline(
            prepared,
            app=app,
            overwrite=overwrite,
            _defer_openapi_rebuild=_defer_openapi_rebuild,
            cleanup_files_on_overwrite=cleanup_files_on_overwrite,
        )


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
    """
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
        cleanup_files_on_overwrite = overwrite and not save_file
        if overwrite and save_file:
            remove_pipeline_files(pipeline_name, settings.pipelines_dir)

        prepared = prepare_pipeline_yaml(pipeline_name, source_code=source_code, options=options)
        return commit_prepared_pipeline(
            prepared,
            app=app,
            overwrite=overwrite,
            _defer_openapi_rebuild=_defer_openapi_rebuild,
            cleanup_files_on_overwrite=cleanup_files_on_overwrite,
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
    """Deploy pipelines from the configured directory"""
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
    """
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
            route_paths = {
                f"/{pipeline_name}/run",
                *(f"/{pipeline_name}{suffix}" for suffix in _DURABLE_ROUTE_SUFFIXES),
            }
            app.routes[:] = [
                route for route in app.routes if not (isinstance(route, APIRoute) and route.path in route_paths)
            ]

            # Invalidate OpenAPI cache
            app.openapi_schema = None
            app.setup()

        # Remove pipeline files if they exist
        remove_pipeline_files(pipeline_name, settings.pipelines_dir)
