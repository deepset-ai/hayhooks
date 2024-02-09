from hayhooks.server import app
from hayhooks.server.utils.deploy_utils import deploy_pipeline_def, PipelineDefinition


@app.post("/deploy")
async def deploy(pipeline_def: PipelineDefinition):
    return await deploy_pipeline_def(app, pipeline_def)
