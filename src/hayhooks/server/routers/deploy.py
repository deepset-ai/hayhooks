from fastapi import APIRouter, Request
from hayhooks.server.utils.deploy_utils import deploy_pipeline_def, PipelineDefinition

router = APIRouter()


@router.post("/deploy", tags=["config"])
async def deploy(pipeline_def: PipelineDefinition, request: Request):
    return deploy_pipeline_def(request.app, pipeline_def)
