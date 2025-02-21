from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Dict
from hayhooks.server.exceptions import PipelineAlreadyExistsError
from hayhooks.server.utils.deploy_utils import (
    deploy_pipeline_def,
    PipelineDefinition,
    deploy_pipeline_files,
    PipelineFilesError,
    PipelineModuleLoadError,
    PipelineWrapperError,
)

router = APIRouter()


class PipelineFilesRequest(BaseModel):
    name: str
    files: Dict[str, str]
    overwrite: bool = False
    save_files: bool = True


@router.post("/deploy", tags=["config"])
async def deploy(pipeline_def: PipelineDefinition, request: Request):
    return deploy_pipeline_def(request.app, pipeline_def)


@router.post("/deploy_files", tags=["config"])
async def deploy_files(pipeline_files_request: PipelineFilesRequest, request: Request):
    try:
        return deploy_pipeline_files(
            app=request.app,
            pipeline_name=pipeline_files_request.name,
            files=pipeline_files_request.files,
            save_files=pipeline_files_request.save_files,
            overwrite=pipeline_files_request.overwrite,
        )
    except PipelineFilesError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except PipelineModuleLoadError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except PipelineWrapperError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except PipelineAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error deploying pipeline: {str(e)}") from e
