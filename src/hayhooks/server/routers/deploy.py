from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from hayhooks.server.exceptions import InvalidYamlIOError, PipelineAlreadyExistsError, PipelineYamlError
from hayhooks.server.utils.deploy_utils import (
    PipelineFilesError,
    PipelineModuleLoadError,
    PipelineWrapperError,
    deploy_pipeline_files,
    deploy_pipeline_yaml,
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


class YamlDeployRequest(BaseModel):
    name: str = Field(description="Name of the pipeline to deploy")
    source_code: str = Field(description="YAML pipeline definition source code")
    overwrite: bool = Field(default=False, description="Whether to overwrite an existing pipeline with the same name")
    description: Optional[str] = Field(default=None, description="Optional description for the pipeline")
    skip_mcp: Optional[bool] = Field(default=None, description="Whether to skip MCP integration for this pipeline")
    save_file: Optional[bool] = Field(default=True, description="Whether to save YAML under pipelines/{name}.yml")

    model_config = {
        "json_schema_extra": {
            "description": "Request model for deploying a YAML pipeline",
            "examples": [
                {
                    "name": "inputs_outputs_pipeline",
                    "source_code": "{yaml source}",
                    "overwrite": False,
                    "description": "My pipeline",
                    "skip_mcp": False,
                }
            ],
        }
    }


@router.post(
    "/deploy-yaml",
    tags=["config"],
    operation_id="yaml_pipeline_deploy",
    response_model=DeployResponse,
    summary="Deploy a pipeline from a YAML definition",
    description=(
        "Deploys a Haystack pipeline from a YAML string. Builds request/response schemas from declared "
        "inputs/outputs. Returns 409 if the pipeline already exists and overwrite is false."
    ),
)
async def deploy_yaml(yaml_request: YamlDeployRequest, request: Request) -> DeployResponse:
    try:
        result = deploy_pipeline_yaml(
            app=request.app,
            pipeline_name=yaml_request.name,
            source_code=yaml_request.source_code,
            overwrite=yaml_request.overwrite,
            options={
                "description": yaml_request.description,
                "skip_mcp": yaml_request.skip_mcp,
                "save_file": yaml_request.save_file,
            },
        )
        return DeployResponse(name=result["name"], success=True, endpoint=f"/{result['name']}/run")
    except InvalidYamlIOError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except PipelineYamlError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except PipelineAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error deploying YAML pipeline: {e!s}") from e


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
