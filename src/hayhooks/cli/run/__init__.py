import click
import uvicorn
import os


@click.command()
@click.option('--host', default="localhost")
@click.option('--port', default=1416)
@click.option('--pipelines-dir', default=os.environ.get("HAYHOOKS_PIPELINES_DIR"))
def run(host, port, pipelines_dir):
    if not pipelines_dir:
        pipelines_dir = "pipelines.d"
    os.environ["HAYHOOKS_PIPELINES_DIR"] = pipelines_dir
    uvicorn.run("hayhooks.server:app", host=host, port=port)
