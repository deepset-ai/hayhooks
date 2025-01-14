import click
import uvicorn
import sys
from hayhooks.settings import settings


@click.command()
@click.option('--host', default=settings.host)
@click.option('--port', default=settings.port)
@click.option('--pipelines-dir', default=settings.pipelines_dir)
@click.option('--additional-python-path', default=settings.additional_python_path)
def run(host, port, pipelines_dir, additional_python_path):
    settings.host = host
    settings.port = port
    settings.pipelines_dir = pipelines_dir
    settings.additional_python_path = additional_python_path

    if additional_python_path:
        sys.path.append(additional_python_path)

    uvicorn.run("hayhooks.server:app", host=host, port=port)
