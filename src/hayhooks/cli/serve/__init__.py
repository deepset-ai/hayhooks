from pathlib import Path

import click
import requests


@click.command()
@click.option('-n', '--name')
@click.argument('pipeline_file', type=click.File('r'))
def serve(name, pipeline_file):
    if name is None:
        name = Path(pipeline_file.name).stem
    resp = requests.post("http://localhost:1416/serve", json={"name": name, "source_code": str(pipeline_file.read())})
    click.echo(name)
    click.echo(resp.text)
