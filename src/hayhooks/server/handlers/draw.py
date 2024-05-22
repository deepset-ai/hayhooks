import tempfile
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from hayhooks.server import app
from hayhooks.server.pipelines import registry


@app.get("/draw/{pipeline_name}", tags=["config"])
async def status(pipeline_name):
    pipeline = registry.get(pipeline_name)
    if not pipeline:
        raise HTTPException(status_code=404)

    _, fpath = tempfile.mkstemp()
    pipeline.draw(Path(fpath))
    return FileResponse(fpath, media_type="image/png")
