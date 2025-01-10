from fastapi import HTTPException
from hayhooks.server import app
from hayhooks.server.pipelines import registry


@app.get("/status")
async def status(pipeline_name: str | None = None):
    if pipeline_name:
        if pipeline_name not in registry.get_names():
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
        return {"status": "Up!", "pipeline": pipeline_name}

    pipelines = registry.get_names()
    return {"status": "Up!", "pipelines": pipelines}
