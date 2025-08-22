from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from hayhooks.server.exceptions import PipelineAlreadyExistsError
from hayhooks.server.utils.deploy_utils import (
    PipelineDefinition,
    PipelineFilesError,
    PipelineModuleLoadError,
    PipelineWrapperError,
    deploy_pipeline_def,
    deploy_pipeline_files,
)

router = APIRouter()


SAMPLE_PIPELINE_FILES = {
    "pipeline_wrapper.py": (
        "from typing import Dict, Any\n\ndef process(data: Dict[str, Any]) -> Dict[str, Any]:\n    "
        ":# Your processing logic here\n    return data"
    ),
    "requirements.txt": "pandas==1.3.5\nnumpy==1.21.0",
}


class PipelineFilesRequest(BaseModel):
    name: str = Field(description="Name of the pipeline to deploy")
    files: dict[str, str] = Field(
        description="Dictionary of files required for the pipeline, must include pipeline_wrapper.py",
        examples=[SAMPLE_PIPELINE_FILES],
    )
    overwrite: bool = Field(default=False, description="Whether to overwrite an existing pipeline with the same name")
    save_files: bool = Field(default=True, description="Whether to save the pipeline files to disk")

    model_config = {
        "json_schema_extra": {
            "description": "Request model for deploying a pipeline with a pipeline_wrapper.py and other files",
            "examples": [
                {
                    "name": "my_pipeline",
                    "files": {
                        "pipeline_wrapper.py": "{python code}",
                        "other_file.py": "{python code}",
                        "other_file.txt": "{text content}",
                    },
                    "overwrite": False,
                    "save_files": True,
                }
            ],
        }
    }

    @field_validator("files")
    @classmethod
    def validate_files(cls, v: dict[str, str]) -> dict[str, str]:
        if "pipeline_wrapper.py" not in v:
            msg = "Missing required file: pipeline_wrapper.py"
            raise ValueError(msg)
        return v


class DeployResponse(BaseModel):
    name: str = Field(description="Name of the deployed pipeline")
    success: bool = Field(default=True, description="Whether the deployment was successful")
    endpoint: str = Field(description="Endpoint of the deployed pipeline")

    model_config = {"json_schema_extra": {"description": "Response model for pipeline deployment operations"}}


@router.post(
    "/deploy",
    tags=["config"],
    response_model=DeployResponse,
    operation_id="legacy_yaml_deploy",
    summary="Deploy a pipeline from YAML definition (Not Maintained)",
    description=(
        "[DEPRECATED] This route is no longer maintained and will be removed in a future version. "
        "Please use /deploy_files endpoint instead. "
        "Deploys a pipeline from a PipelineDefinition object. "
        "Returns 409 if the pipeline already exists and overwrite is false."
    ),
    deprecated=True,
)
async def deploy(pipeline_def: PipelineDefinition, request: Request) -> DeployResponse:
    result = deploy_pipeline_def(request.app, pipeline_def)
    return DeployResponse(name=result["name"], success=True, endpoint=f"/{result['name']}/run")


@router.post(
    "/deploy_files",
    tags=["config"],
    operation_id="pipeline_deploy",
    response_model=DeployResponse,
    summary="Deploy a pipeline from files (`pipeline_wrapper.py` and other files)",
    description=(
        "Deploys a pipeline from a dictionary of file contents. "
        "Returns 409 if the pipeline already exists and overwrite is false."
    ),
)
async def deploy_files(pipeline_files_request: PipelineFilesRequest, request: Request) -> DeployResponse:
    try:
        result = deploy_pipeline_files(
            app=request.app,
            pipeline_name=pipeline_files_request.name,
            files=pipeline_files_request.files,
            save_files=pipeline_files_request.save_files,
            overwrite=pipeline_files_request.overwrite,
        )
        return DeployResponse(name=result["name"], success=True, endpoint=f"/{result['name']}/run")
    except PipelineFilesError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except PipelineModuleLoadError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except PipelineWrapperError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except PipelineAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error deploying pipeline: {e!s}") from e
