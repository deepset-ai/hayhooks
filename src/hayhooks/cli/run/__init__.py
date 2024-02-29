import click
import uvicorn
import os
import sys


@click.command()
@click.option('--host', default="localhost")
@click.option('--port', default=1416)
@click.option('--pipelines-dir', default=os.environ.get("HAYHOOKS_PIPELINES_DIR"))
@click.option('--additional-python-path', default=os.environ.get("HAYHOOKS_ADDITIONAL_PYTHONPATH"))
def run(host, port, pipelines_dir, additional_python_path):
    if not pipelines_dir:
        pipelines_dir = "pipelines.d"
    os.environ["HAYHOOKS_PIPELINES_DIR"] = pipelines_dir

    if additional_python_path:
        sys.path.append(additional_python_path)

    uvicorn.run("hayhooks.server:app", host=host, port=port)
