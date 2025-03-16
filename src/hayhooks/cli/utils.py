import requests
import typer
from typing import Optional, Dict, Any, Callable
from rich.console import Console
from urllib.parse import urljoin
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

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


def show_error_and_abort(message: str, highlight: str = "") -> None:
    """Display error message in a panel and abort."""
    if highlight:
        message = message.replace(highlight, f"[red]{highlight}[/red]")
    console.print(Panel.fit(message, border_style="red", title="Error"))
    raise typer.Abort()


def show_success_panel(message: str, title: str = "Success") -> None:
    """Display success message in a panel."""
    console.print(Panel.fit(message, border_style="green", title=title))


def show_warning_panel(message: str, title: str = "Warning") -> None:
    """Display warning message in a panel."""
    console.print(Panel.fit(message, border_style="yellow", title=title))


def with_progress_spinner(description: str, operation: Callable, *args, **kwargs):
    """Execute an operation with a progress spinner.

    Args:
        description: Description to show in the spinner
        operation: Function to call
        *args, **kwargs: Arguments to pass to the operation

    Returns:
        The result of the operation
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description=description, total=None)
        return operation(*args, **kwargs)
