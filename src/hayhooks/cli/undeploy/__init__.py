from pathlib import Path
from urllib.parse import urljoin

import click
import requests
from requests import ConnectionError


@click.command()
@click.pass_obj
@click.argument('pipeline_name')
def undeploy(server_conf, pipeline_name):
    server, disable_ssl = server_conf
    try:
        resp = requests.post(urljoin(server, f"undeploy/{pipeline_name}"), verify=not disable_ssl)

        if resp.status_code >= 400:
            click.echo(f"Cannot undeploy pipeline: {resp.json().get('detail')}")
        else:
            click.echo(f"Pipeline successfully undeployed")
    except ConnectionError:
        click.echo("Hayhooks server is not responding. To start one, run `hayooks run`")
