import tempfile
import importlib.util
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pathlib import Path
from hayhooks.server.exceptions import PipelineFilesError
from hayhooks.server.pipelines import registry
from hayhooks.server.pipelines.models import (
    PipelineDefinition,
    convert_component_output,
    get_request_model,
    get_response_model,
)
from hayhooks.server.logger import log
from hayhooks.settings import settings


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


def load_pipeline_module(pipeline_name: str, folder_path: Path | str):
    """Load a pipeline module from a folder path.

    Args:
        pipeline_name: Name of the pipeline
        folder_path: Path to the folder containing the pipeline files

    Returns:
        The loaded module

    Raises:
        ValueError: If the module cannot be loaded
    """
    log.debug(f"Loading pipeline module spec for {pipeline_name}")
    try:
        folder_path = Path(folder_path)
        wrapper_path = folder_path / "pipeline_wrapper.py"

        if not wrapper_path.exists():
            raise ValueError(f"Required file '{wrapper_path}' not found")

        spec = importlib.util.spec_from_file_location(pipeline_name, wrapper_path)
        if spec is None:
            raise ValueError(f"Failed to load '{pipeline_name}' pipeline module spec")

        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ValueError(f"Failed to load '{pipeline_name}' pipeline module")

        spec.loader.exec_module(module)
        log.debug(f"Loaded module {module}")

        if not hasattr(module, "PipelineWrapper"):
            raise ValueError(f"Failed to load '{pipeline_name}' pipeline module spec")

        return module

    except Exception as e:
        log.error(f"Error loading pipeline module: {str(e)}")
        raise ValueError(f"Failed to load pipeline module: {str(e)}") from e
