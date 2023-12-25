from hayhooks.server import app


@app.get("/status")
async def status():
    return {"status": "Up!"}
