import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from hayhooks.server.pipelines import registry
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

router = APIRouter()


@router.get(
    "/draw/{pipeline_name}",
    tags=["config"],
    operation_id="pipeline_draw",
    summary="Generate a pipeline diagram",
    description=(
        "Returns a PNG image visualization of the specified pipeline. Returns 404 if the pipeline doesn't exist."
    ),
    response_class=FileResponse,
    responses={
        200: {"content": {"image/png": {}}, "description": "A PNG visualization of the pipeline graph"},
        404: {"description": "Pipeline not found"},
    },
)
async def draw(
    pipeline_name: str = PathParam(description="Name of the pipeline to visualize", examples=["my_pipeline"]),
) -> FileResponse:
    pipeline = registry.get(pipeline_name)

    if isinstance(pipeline, BasePipelineWrapper):
        pipeline = pipeline.pipeline

    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")

    _, fpath = tempfile.mkstemp(suffix=".png")
    pipeline.draw(path=Path(fpath))
    return FileResponse(fpath, media_type="image/png")
