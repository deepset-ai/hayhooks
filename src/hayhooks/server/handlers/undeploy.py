from typing import cast

from fastapi import HTTPException
from fastapi.routing import APIRoute
from hayhooks.server import app
from hayhooks.server.pipelines import registry


@app.post("/undeploy/{pipeline_name}")
async def deploy(pipeline_name: str):
    if pipeline_name not in registry.get_names():
        raise HTTPException(status_code=404)

    new_routes = []
    for route in app.router.routes:
        route = cast(APIRoute, route)
        if route.name != pipeline_name:
            new_routes.append(route)

    app.router.routes = new_routes
    app.openapi_schema = None
    app.setup()
    registry.remove(pipeline_name)
