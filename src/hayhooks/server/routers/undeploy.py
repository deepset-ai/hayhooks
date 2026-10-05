from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field

from hayhooks.server.exceptions import PipelinePathError
from hayhooks.server.utils.deploy_utils import undeploy_pipeline_async

router = APIRouter()


class UndeployResponse(BaseModel):
    success: bool = Field(description="Whether the undeployment was successful")
    name: str = Field(description="Name of the undeployed pipeline")

    model_config = {"json_schema_extra": {"description": "Response model for pipeline undeployment operation"}}


@router.post(
    "/undeploy/{pipeline_name}",
    tags=["config"],
    operation_id="pipeline_undeploy",
    response_model=UndeployResponse,
    summary="Undeploy a pipeline",
    description="Removes a pipeline from the registry, removes its API routes and deletes its files from disk.",
    responses={200: {"description": "Pipeline successfully undeployed"}, 404: {"description": "Pipeline not found"}},
)
async def undeploy(
    request: Request,
    pipeline_name: str = Path(description="Name of the pipeline to undeploy", examples=["my_pipeline"]),
) -> UndeployResponse:
    try:
        await undeploy_pipeline_async(pipeline_name, request.app)
    except PipelinePathError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return UndeployResponse(success=True, name=pipeline_name)
