from typing import cast
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from hayhooks.server.pipelines import registry

router = APIRouter()


@router.post("/undeploy/{pipeline_name}", tags=["config"])
async def undeploy(pipeline_name: str):
    if pipeline_name not in registry.get_names():
        raise HTTPException(status_code=404)

    new_routes = []
    for route in router.routes:
        route = cast(APIRoute, route)
        if route.name != pipeline_name:
            new_routes.append(route)

    router.routes = new_routes
    router.openapi_schema = None
    registry.remove(pipeline_name)
