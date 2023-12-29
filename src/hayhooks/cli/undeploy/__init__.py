from pathlib import Path

import click
import requests
from requests import ConnectionError


@click.command()
@click.argument('pipeline_name')
def undeploy(pipeline_name):
    try:
        resp = requests.post(f"http://localhost:1416/undeploy/{pipeline_name}")

        if resp.status_code >= 400:
            click.echo(f"Cannot undeploy pipeline: {resp.json().get('detail')}")
        else:
            click.echo(f"Pipeline successfully undeployed")
    except ConnectionError:
        click.echo("Hayhooks server is not responding. To start one, run `hayooks run`")
