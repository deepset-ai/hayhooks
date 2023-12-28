from fastapi import HTTPException
from hayhooks.server import app
from hayhooks.server.pipelines import registry


@app.get("/status")
async def status():
    pipelines = registry.get_names()
    return {"status": "Up!", "pipelines": pipelines}
