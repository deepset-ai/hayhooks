import click
import uvicorn
import sys
from hayhooks.settings import settings


@click.command()
@click.option('--host', default=settings.host)
@click.option('--port', default=settings.port)
@click.option('--pipelines-dir', default=settings.pipelines_dir)
@click.option('--root-path', default=settings.root_path)
@click.option('--additional-python-path', default=settings.additional_python_path)
def run(host, port, pipelines_dir, root_path, additional_python_path):
    settings.host = host
    settings.port = port
    settings.pipelines_dir = pipelines_dir
    settings.root_path = root_path
    settings.additional_python_path = additional_python_path

    if additional_python_path:
        sys.path.append(additional_python_path)

    uvicorn.run("hayhooks.server:app", host=host, port=port)
