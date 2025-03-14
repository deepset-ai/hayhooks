from fastapi import APIRouter, HTTPException, Request
from fastapi.routing import APIRoute
from hayhooks.server.pipelines import registry
from hayhooks.server.utils.deploy_utils import remove_pipeline_files
from hayhooks.settings import settings

router = APIRouter()


@router.post("/undeploy/{pipeline_name}", tags=["config"])
async def undeploy(pipeline_name: str, request: Request):
    # Check if pipeline exists in registry
    if pipeline_name not in registry.get_names():
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")

    # Remove pipeline from registry
    registry.remove(pipeline_name)

    # Remove API routes for the pipeline
    # YAML based pipelines have a run endpoint at /<pipeline_name>
    # Wrapper based pipelines have a run endpoint at /<pipeline_name>/run
    for route in request.app.routes:
        if isinstance(route, APIRoute) and (route.path == f"/{pipeline_name}/run" or route.path == f"/{pipeline_name}"):
            request.app.routes.remove(route)

    # Remove pipeline files if they exist
    remove_pipeline_files(pipeline_name, settings.pipelines_dir)

    return {"success": True, "name": pipeline_name}
