import click


@click.command()
@click.argument('pipeline_path', type=click.File('r'))
def serve(pipeline_path):
    while True:
        chunk = pipeline_path.read(1024)
        if not chunk:
            break
        click.echo(chunk)
