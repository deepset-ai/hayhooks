from fastapi import Response

from pydantic import BaseModel

from hayhooks.server import app
from hayhooks.server.pipelines import registry, pipeline_run, PipelineRunResponse


class PipelineDefinition(BaseModel):
    name: str
    source_code: str


@app.post("/serve")
async def serve(pipeline: PipelineDefinition):
    registry.add(pipeline.name, pipeline.source_code)

    app.add_api_route(
        path=f"/{pipeline.name}",
        endpoint=pipeline_run,
        methods=["POST"],
        name=pipeline.name,
        response_model=PipelineRunResponse,
    )
    app.openapi_schema = None
    app.setup()

    return {"pipeline_name": pipeline.name}
