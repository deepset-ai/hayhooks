from pathlib import Path
from urllib.parse import urljoin

import click
import requests


@click.command()
@click.pass_obj
@click.option('-n', '--name')
@click.argument('pipeline_file', type=click.File('r'))
def deploy(server_conf, name, pipeline_file):
    server, disable_ssl = server_conf
    if name is None:
        name = Path(pipeline_file.name).stem
    resp = requests.post(
        urljoin(server, "deploy"),
        json={"name": name, "source_code": str(pipeline_file.read())}, 
        verify=not disable_ssl
    )

    if resp.status_code >= 400:
        click.echo(f"Error deploying pipeline: {resp.json().get('detail')}")
    else:
        click.echo(f"Pipeline successfully deployed with name: {resp.json().get('name')}")
