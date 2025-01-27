import click
import requests
from pathlib import Path
from urllib.parse import urljoin
from hayhooks.server.utils.deploy_utils import read_pipeline_files_from_folder


@click.command()
@click.pass_obj
@click.option('-n', '--name', required=True, help="Name of the pipeline to deploy")
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
def deploy_files(server_conf, name, folder):
    """Deploy all pipeline files from a folder to the Hayhooks server."""
    server, disable_ssl = server_conf

    files_dict = {}
    try:
        files_dict = read_pipeline_files_from_folder(folder)

        if not files_dict:
            click.echo("Error: No valid files found in the specified folder")
            return

        resp = requests.post(
            urljoin(server, "deploy_files"), json={"name": name, "files": files_dict}, verify=not disable_ssl
        )

        if resp.status_code >= 400:
            click.echo(f"Error deploying pipeline: {resp.json().get('detail')}")
        else:
            click.echo(f"Pipeline successfully deployed with name: {resp.json().get('name')}")

    except Exception as e:
        click.echo(f"Error processing folder: {str(e)}")
