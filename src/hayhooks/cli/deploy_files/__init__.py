import click
import requests
from pathlib import Path
from urllib.parse import urljoin


def should_skip_file(file_path: Path) -> bool:
    """
    Determine if a file should be skipped during deployment.
    Handles hidden files across different operating systems.
    """
    if file_path.is_dir():
        return True

    if file_path.name.startswith('.'):
        return True

    return False


@click.command()
@click.pass_obj
@click.option('-n', '--name', required=True, help="Name of the pipeline to deploy")
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
def deploy_files(server_conf, name, folder):
    """Deploy all pipeline files from a folder to the Hayhooks server."""
    server, disable_ssl = server_conf

    files_dict = {}
    try:
        for file_path in folder.iterdir():
            if should_skip_file(file_path):
                continue

            try:
                files_dict[file_path.name] = file_path.read_text()
            except Exception as e:
                click.echo(f"Error reading file {file_path}: {str(e)}")
                return

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
