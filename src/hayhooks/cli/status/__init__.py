import click
import requests


@click.command()
def status():
    r = requests.get("http://localhost:1416/status")
    click.echo(r.status_code)
