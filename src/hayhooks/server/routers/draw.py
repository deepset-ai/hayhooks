import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from hayhooks.server.pipelines import registry

router = APIRouter()


@router.get("/draw/{pipeline_name}", tags=["config"])
async def draw(pipeline_name):
    pipeline = registry.get(pipeline_name, use_pipeline=True)
    if not pipeline:
        raise HTTPException(status_code=404)

    _, fpath = tempfile.mkstemp()
    pipeline.draw(Path(fpath))
    return FileResponse(fpath, media_type="image/png")
