import click
import uvicorn
import os


@click.command()
@click.option('--host', default="localhost")
@click.option('--port', default=1416)
@click.option('--pipelines-dir', default="pipelines.d")
def run(host, port, pipelines_dir):
    os.environ["PIPELINES_DIR"] = pipelines_dir
    uvicorn.run("hayhooks.server:app", host=host, port=port)
