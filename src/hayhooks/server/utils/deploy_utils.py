import inspect
import importlib.util
import shutil
import tempfile
import traceback
import sys
from functools import wraps
from types import ModuleType
from typing import Callable, Union
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pathlib import Path

from fastapi.routing import APIRoute
from hayhooks.server.exceptions import (
    PipelineAlreadyExistsError,
    PipelineFilesError,
    PipelineModuleLoadError,
    PipelineWrapperError,
)
from hayhooks.server.pipelines import registry
from hayhooks.server.pipelines.models import (
    PipelineDefinition,
    convert_component_output,
    get_request_model,
    get_response_model,
)
from hayhooks.server.logger import log
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings
from pydantic import create_model


def deploy_pipeline_def(app, pipeline_def: PipelineDefinition):
    try:
        pipe = registry.add(pipeline_def.name, pipeline_def.source_code)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=f"{e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e

    PipelineRunRequest = get_request_model(pipeline_def.name, pipe.inputs())
    PipelineRunResponse = get_response_model(pipeline_def.name, pipe.outputs())

    # There's no way in FastAPI to define the type of the request body other than annotating
    # the endpoint handler. We have to ignore the type here to make FastAPI happy while
    # silencing static type checkers (that would have good reasons to trigger!).
    async def pipeline_run(pipeline_run_req: PipelineRunRequest) -> JSONResponse:  # type: ignore
        result = await run_in_threadpool(pipe.run, data=pipeline_run_req.dict())
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
    """Save pipeline files to disk and return their paths.

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
        raise PipelineFilesError(f"Failed to save pipeline files: {str(e)}") from e


def remove_pipeline_files(pipeline_name: str, pipelines_dir: str):
    """Remove pipeline files from disk.

    Args:
        pipeline_name: Name of the pipeline
        pipelines_dir: Path to the pipelines directory
    """
    pipeline_dir = Path(pipelines_dir) / pipeline_name
    if pipeline_dir.exists():
        shutil.rmtree(pipeline_dir, ignore_errors=True)


def load_pipeline_module(pipeline_name: str, dir_path: Union[Path, str]) -> ModuleType:
    """Load a pipeline module from a directory path.

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
            raise PipelineWrapperError(f"Required file '{wrapper_path}' not found")

        # Clear the module from sys.modules if it exists to ensure a fresh load
        module_name = pipeline_name
        if module_name in sys.modules:
            log.debug(f"Removing existing module {module_name} from sys.modules")
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(pipeline_name, wrapper_path)
        if spec is None or spec.loader is None:
            raise PipelineModuleLoadError(
                f"Failed to load pipeline module '{pipeline_name}' - module loader not available"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        log.debug(f"Loaded module {module}")

        if not hasattr(module, "PipelineWrapper"):
            raise PipelineWrapperError(f"Failed to load '{pipeline_name}' pipeline module spec")

        return module

    except Exception as e:
        log.error(f"Error loading pipeline module: {str(e)}")
        error_msg = f"Failed to load pipeline module '{pipeline_name}' - {str(e)}"
        if settings.show_tracebacks:
            error_msg += f"\n{traceback.format_exc()}"
        raise PipelineModuleLoadError(error_msg) from e


def create_request_model_from_callable(func: Callable, model_name: str):
    """Create a dynamic Pydantic model based on callable's signature.

    Args:
        func: The callable (function/method) to analyze
        model_name: Name for the generated model

    Returns:
        Pydantic model class for request
    """

    params = inspect.signature(func).parameters
    fields = {
        name: (param.annotation, ... if param.default == param.empty else param.default)
        for name, param in params.items()
    }
    return create_model(f'{model_name}Request', **fields)


def create_response_model_from_callable(func: Callable, model_name: str):
    """Create a dynamic Pydantic model based on callable's return type.

    Args:
        func: The callable (function/method) to analyze
        model_name: Name for the generated model

    Returns:
        Pydantic model class for response
    """

    return_type = inspect.signature(func).return_annotation
    return create_model(f'{model_name}Response', result=(return_type, ...))


def handle_pipeline_exceptions():
    """Decorator to handle pipeline execution exceptions."""

    def decorator(func):
        @wraps(func)  # Preserve the original function's metadata
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException as e:
                raise e from e
            except Exception as e:
                error_msg = f"Pipeline execution failed: {str(e)}"
                if settings.show_tracebacks:
                    log.error(f"Pipeline execution error: {str(e)} - {traceback.format_exc()}")
                    error_msg += f"\n{traceback.format_exc()}"
                else:
                    log.error(f"Pipeline execution error: {str(e)}")
                raise HTTPException(status_code=500, detail=error_msg) from e

        return wrapper

    return decorator


def deploy_pipeline_files(
    app: FastAPI, pipeline_name: str, files: dict[str, str], save_files: bool = True, overwrite: bool = False
):
    """Deploy pipeline files to the FastAPI application and set up endpoints.

    Args:
        app: FastAPI application instance
        pipeline_name: Name of the pipeline to deploy
        files: Dictionary mapping filenames to their contents

    Returns:
        dict: Dictionary containing the deployed pipeline name

    Raises:
        PipelineFilesError: If there are issues saving or loading pipeline files
    """

    log.debug(f"Checking if pipeline '{pipeline_name}' already exists: {registry.get(pipeline_name)}")
    if registry.get(pipeline_name):
        if overwrite:
            log.debug(f"Clearing existing pipeline '{pipeline_name}'")
            registry.remove(pipeline_name)

            log.debug(f"Removing pipeline files for '{pipeline_name}'")
            remove_pipeline_files(pipeline_name, settings.pipelines_dir)
        else:
            raise PipelineAlreadyExistsError(f"Pipeline '{pipeline_name}' already exists")

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

    clog.debug("Adding pipeline to registry")
    registry.add(pipeline_name, pipeline_wrapper)

    if pipeline_wrapper._is_run_api_implemented:
        clog.debug("Creating dynamic Pydantic models for run_api")

        RunRequest = create_request_model_from_callable(pipeline_wrapper.run_api, f'{pipeline_name}Run')
        RunResponse = create_response_model_from_callable(pipeline_wrapper.run_api, f'{pipeline_name}Run')

        # There's no way in FastAPI to define the type of the request body other than annotating
        # the endpoint handler. We have to ignore the type here to make FastAPI happy while
        # silencing static type checkers (that would have good reasons to trigger!).
        @handle_pipeline_exceptions()
        async def run_endpoint(run_req: RunRequest) -> RunResponse:  # type: ignore
            result = await run_in_threadpool(pipeline_wrapper.run_api, **run_req.model_dump())  # type: ignore
            return RunResponse(result=result)

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
        )

    clog.debug("Setting up FastAPI app")
    app.openapi_schema = None
    app.setup()

    clog.success("Pipeline deployment complete")

    if tmp_dir is not None:
        log.debug(f"Removing temporary pipeline files for '{pipeline_name}'")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"name": pipeline_name}


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

    pipeline_wrapper._is_run_api_implemented = pipeline_wrapper.run_api.__func__ is not BasePipelineWrapper.run_api
    pipeline_wrapper._is_run_chat_completion_implemented = (
        pipeline_wrapper.run_chat_completion.__func__ is not BasePipelineWrapper.run_chat_completion
    )

    log.debug(f"pipeline_wrapper._is_run_api_implemented: {pipeline_wrapper._is_run_api_implemented}")
    log.debug(
        f"pipeline_wrapper._is_run_chat_completion_implemented: {pipeline_wrapper._is_run_chat_completion_implemented}"
    )

    if not (pipeline_wrapper._is_run_api_implemented or pipeline_wrapper._is_run_chat_completion_implemented):
        raise PipelineWrapperError("At least one of run_api or run_chat_completion must be implemented")

    return pipeline_wrapper


def read_pipeline_files_from_dir(dir_path: Path) -> dict[str, str]:
    """Read pipeline files from a directory and return a dictionary mapping filenames to their contents.
    Skips directories, hidden files, and common Python artifacts.

    Args:
        dir_path: Path to the directory containing the pipeline files

    Returns:
        Dictionary mapping filenames to their contents
    """

    files = {}
    for file_path in dir_path.rglob("*"):
        if file_path.is_dir() or file_path.name.startswith('.'):
            continue

        if any(file_path.match(pattern) for pattern in settings.files_to_ignore_patterns):
            continue

        try:
            files[str(file_path.relative_to(dir_path))] = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log.warning(f"Skipping file {file_path}: {str(e)}")
            continue

    return files
