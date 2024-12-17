from fastapi import FastAPI
import os
import glob
from pathlib import Path
from hayhooks.server.utils.deploy_utils import deploy_pipeline_def, PipelineDefinition
import logging

logger = logging.getLogger("uvicorn.info")


def create_app() -> FastAPI:
    if root_path := os.environ.get("HAYHOOKS_ROOT_PATH"):
        app = FastAPI(root_path=root_path)
    else:
        app = FastAPI()

    # Deploy all pipelines in the pipelines directory
    pipelines_dir = os.environ.get("HAYHOOKS_PIPELINES_DIR")
    if pipelines_dir:
        logger.info(f"Pipelines dir set to: {pipelines_dir}")
        for pipeline_file_path in glob.glob(f"{pipelines_dir}/*.y*ml"):
            name = Path(pipeline_file_path).stem
            with open(pipeline_file_path, "r") as pipeline_file:
                source_code = pipeline_file.read()

            pipeline_defintion = PipelineDefinition(name=name, source_code=source_code)
            deployed_pipeline = deploy_pipeline_def(app, pipeline_defintion)
            logger.info(f"Deployed pipeline: {deployed_pipeline['name']}")
    return app


app = create_app()


@app.get("/")
async def root():
    return {
        "swagger_docs" : "http://localhost:1416/docs",
        "deploy_pipeline" : "http://localhost:1416/deploy",
        "draw_pipeline" : "http://localhost:1416/draw/{pipeline_name}",
        "server_status" : "http://localhost:1416/status",
        "undeploy_pipeline" : "http://localhost:1416/undeploy/{pipeline_name}",
    }
