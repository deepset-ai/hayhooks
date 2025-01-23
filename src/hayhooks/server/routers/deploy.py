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


@router.post("/deploy", tags=["config"])
async def deploy(pipeline_def: PipelineDefinition, request: Request):
    return deploy_pipeline_def(request.app, pipeline_def)


@router.post("/deploy_files", tags=["config"])
async def deploy_files(pipeline_files: PipelineFilesRequest, request: Request):
    try:
        return deploy_pipeline_files(request.app, pipeline_files.name, pipeline_files.files)
    except PipelineFilesError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except PipelineModuleLoadError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PipelineWrapperError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PipelineAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error deploying pipeline: {str(e)}")
