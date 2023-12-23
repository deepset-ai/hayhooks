import click
import uvicorn


@click.command()
@click.argument("host", default="localhost")
@click.argument('port', default=1416)
def run(host, port):
    uvicorn.run("hayhooks.server:app", host=host, port=port)
