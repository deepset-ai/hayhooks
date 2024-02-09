import click
import uvicorn


@click.command()
@click.option('-h', '--host', default="localhost")
@click.option('-p', '--port', default=1416)
def run(host, port):
    uvicorn.run("hayhooks.server:app", host=host, port=port)
