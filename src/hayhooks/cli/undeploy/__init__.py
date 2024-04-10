from pathlib import Path

import click
import requests
from requests import ConnectionError


@click.command()
@click.pass_obj
@click.argument('pipeline_name')
def undeploy(server, pipeline_name):
    try:
        resp = requests.post(f"{server}/undeploy/{pipeline_name}")

        if resp.status_code >= 400:
            click.echo(f"Cannot undeploy pipeline: {resp.json().get('detail')}")
        else:
            click.echo(f"Pipeline successfully undeployed")
    except ConnectionError:
        click.echo("Hayhooks server is not responding. To start one, run `hayooks run`")
