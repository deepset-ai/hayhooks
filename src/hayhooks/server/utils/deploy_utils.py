import inspect
import importlib.util
from functools import wraps
from types import ModuleType
from typing import Callable, Union, List
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pathlib import Path
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
from hayhooks.settings import settings
from pydantic import BaseModel, create_model


class ChatRequest(BaseModel):
    user_message: str
    model_id: str
    messages: List[dict]
    body: dict


class ChatResponse(BaseModel):
    result: dict


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


def save_pipeline_files(
    pipeline_name: str, files: dict[str, str], pipelines_dir: str = settings.pipelines_dir
) -> dict[str, str]:
    """Save pipeline files to disk and return their paths.

    Args:
        pipeline_name: Name of the pipeline
        files: Dictionary mapping filenames to their contents

    Returns:
        Dictionary mapping filenames to their saved paths

    Raises:
        PipelineFilesError: If there are any issues saving the files
    """
    try:
        # Create pipeline directory under the configured pipelines directory
        pipeline_dir = Path(pipelines_dir) / pipeline_name
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


def load_pipeline_module(pipeline_name: str, folder_path: Union[Path, str]) -> ModuleType:
    """Load a pipeline module from a folder path.

    Args:
        pipeline_name: Name of the pipeline
        folder_path: Path to the folder containing the pipeline files

    Returns:
        The loaded module

    Raises:
        ValueError: If the module cannot be loaded
    """
    try:
        folder_path = Path(folder_path)
        wrapper_path = folder_path / "pipeline_wrapper.py"

        if not wrapper_path.exists():
            raise PipelineWrapperError(f"Required file '{wrapper_path}' not found")

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
        raise PipelineModuleLoadError(f"Failed to load pipeline module '{pipeline_name}' - {str(e)}") from e


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
            except Exception as e:
                log.error(f"Pipeline execution error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

        return wrapper

    return decorator


def deploy_pipeline_files(app: FastAPI, pipeline_name: str, files: dict[str, str]):
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
        raise PipelineAlreadyExistsError(f"Pipeline '{pipeline_name}' already exists")

    log.debug(f"Saving pipeline files for '{pipeline_name}'")
    save_pipeline_files(pipeline_name, files=files)

    pipeline_dir = Path(settings.pipelines_dir) / pipeline_name
    clog = log.bind(pipeline_name=pipeline_name, pipeline_dir=str(pipeline_dir), files=list(files.keys()))

    clog.debug("Loading pipeline module")
    module = load_pipeline_module(pipeline_name, folder_path=pipeline_dir)

    clog.debug("Creating PipelineWrapper instance")
    pipeline_wrapper = module.PipelineWrapper()

    clog.debug("Running setup()")
    pipeline_wrapper.setup()

    clog.debug("Adding pipeline to registry")
    registry.add(pipeline_name, pipeline_wrapper)

    clog.debug("Creating dynamic Pydantic models for run_api")
    RunRequest = create_request_model_from_callable(pipeline_wrapper.run_api, f'{pipeline_name}Run')
    RunResponse = create_response_model_from_callable(pipeline_wrapper.run_api, f'{pipeline_name}Run')

    clog.debug("Adding new API endpoints")

    @handle_pipeline_exceptions()
    async def run_endpoint(run_req: RunRequest) -> JSONResponse:  # type: ignore
        result = await run_in_threadpool(pipeline_wrapper.run_api, urls=run_req.urls, question=run_req.question)
        return JSONResponse({"result": result}, status_code=200)

    @handle_pipeline_exceptions()
    async def chat_endpoint(chat_req: ChatRequest) -> JSONResponse:
        result = await run_in_threadpool(
            pipeline_wrapper.run_chat,
            user_message=chat_req.user_message,
            model_id=chat_req.model_id,
            messages=chat_req.messages,
            body=chat_req.body,
        )
        return JSONResponse({"result": result}, status_code=200)

    # Add routes
    app.add_api_route(
        path=f"/{pipeline_name}/run",
        endpoint=run_endpoint,
        methods=["POST"],
        name=f"{pipeline_name}_run",
        response_model=RunResponse,
        tags=["pipelines"],
    )

    app.add_api_route(
        path=f"/{pipeline_name}/chat",
        endpoint=chat_endpoint,
        methods=["POST"],
        name=f"{pipeline_name}_chat",
        response_model=ChatResponse,
        tags=["pipelines"],
    )

    clog.debug("Setting up FastAPI app")
    app.openapi_schema = None
    app.setup()

    clog.success("Pipeline deployment complete")

    return {"name": pipeline_name}
