from typing import Any

from pydantic import BaseModel
from fastapi import Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from hayhooks.server.pipelines import registry


class PipelineRunRequest(BaseModel):
    data: dict[str, Any]


class PipelineRunResponse(BaseModel):
    output: dict[str, Any]


async def pipeline_run(pipeline_run_req: PipelineRunRequest, request: Request) -> JSONResponse:
    name = request.scope['route'].name
    pipe = registry.get(name)
    if pipe is None:
        raise HTTPException(status_code=500, detail=f"Pipeline {name} not found in registry!")
    output = pipe.run(pipeline_run_req.data)
    return JSONResponse(PipelineRunResponse(output=output).model_dump(), status_code=200)
