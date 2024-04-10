from pathlib import Path

import click
import requests


@click.command()
@click.pass_obj
@click.option('-n', '--name')
@click.argument('pipeline_file', type=click.File('r'))
def deploy(server, name, pipeline_file):
    if name is None:
        name = Path(pipeline_file.name).stem
    resp = requests.post(f"{server}/deploy", json={"name": name, "source_code": str(pipeline_file.read())})

    if resp.status_code >= 400:
        click.echo(f"Error deploying pipeline: {resp.json().get('detail')}")
    else:
        click.echo(f"Pipeline successfully deployed with name: {resp.json().get('name')}")
