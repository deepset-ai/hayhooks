from fastapi import APIRouter, HTTPException, Request, Path
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from hayhooks.server.pipelines import registry
from hayhooks.server.utils.deploy_utils import remove_pipeline_files
from hayhooks.settings import settings

router = APIRouter()


class UndeployResponse(BaseModel):
    success: bool = Field(description="Whether the undeployment was successful")
    name: str = Field(description="Name of the undeployed pipeline")

    model_config = {"json_schema_extra": {"description": "Response model for pipeline undeployment operation"}}


@router.post(
    "/undeploy/{pipeline_name}",
    tags=["config"],
    response_model=UndeployResponse,
    summary="Undeploy a pipeline",
    description="Removes a pipeline from the registry, removes its API routes and deletes its files from disk.",
    responses={200: {"description": "Pipeline successfully undeployed"}, 404: {"description": "Pipeline not found"}},
)
async def undeploy(
    request: Request,
    pipeline_name: str = Path(description="Name of the pipeline to undeploy", examples=["my_pipeline"]),
):
    # Check if pipeline exists in registry
    if pipeline_name not in registry.get_names():
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")

    # Remove pipeline from registry
    registry.remove(pipeline_name)

    # Remove API routes for the pipeline
    # YAML based pipelines have a run endpoint at /<pipeline_name>
    # Wrapper based pipelines have a run endpoint at /<pipeline_name>/run
    for route in request.app.routes:
        if isinstance(route, APIRoute) and (route.path == f"/{pipeline_name}/run" or route.path == f"/{pipeline_name}"):
            request.app.routes.remove(route)

    # Remove pipeline files if they exist
    remove_pipeline_files(pipeline_name, settings.pipelines_dir)

    return UndeployResponse(success=True, name=pipeline_name)
