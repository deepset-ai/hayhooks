import requests
import typer
from typing import Optional, Dict, Any
from rich.console import Console
from urllib.parse import urljoin

console = Console()


def get_server_url(host: str, port: int, https: bool = False) -> str:
    if https:
        return f"https://{host}:{port}"
    else:
        return f"http://{host}:{port}"


def make_request(
    host: str,
    port: int,
    endpoint: str,
    method: str = "GET",
    json: Optional[Dict[str, Any]] = None,
    disable_ssl: bool = False,
) -> Dict[str, Any]:
    """Make HTTP request to Hayhooks server with error handling.

    Args:
        host: Server hostname
        port: Server port
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc)
        json: Optional JSON payload
        disable_ssl: Whether to disable SSL verification
    """
    server_url = get_server_url(host, port, disable_ssl)
    url = urljoin(server_url, endpoint)

    try:
        response = requests.request(method=method, url=url, json=json, verify=not disable_ssl)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        console.print("[red][bold]Hayhooks server is not responding.[/bold]\nTo start one, run `hayhooks run`[/red].")
        raise typer.Abort()
    except requests.HTTPError:
        error_detail = response.json().get("detail", "Unknown error")
        console.print(f"[red][bold]Server error[/bold]\n{error_detail}[/red]")
        raise typer.Abort()
    except Exception as e:
        console.print(f"[red][bold]Unexpected error[/bold]\n{str(e)}[/red]")
        raise typer.Abort()
