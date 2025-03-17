import io
import requests
import typer
import time
import mimetypes
from typing import Optional, Dict, Any, Callable, List, Tuple
from pathlib import Path
from rich.console import Console
from urllib.parse import urljoin
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

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


class ProgressFileReader:
    """File-like object wrapper that updates progress bar when read."""

    def __init__(self, file_obj, progress, task_id, file_size):
        self.file_obj = file_obj
        self.progress = progress
        self.task_id = task_id
        self.file_size = file_size
        self.bytes_read = 0

    def read(self, size=-1):
        chunk = self.file_obj.read(size)
        chunk_size = len(chunk)
        self.bytes_read += chunk_size
        self.progress.update(self.task_id, advance=chunk_size)
        return chunk

    def seek(self, offset, whence=0):
        return self.file_obj.seek(offset, whence)

    def tell(self):
        return self.file_obj.tell()

    def close(self):
        return self.file_obj.close()


def prepare_files_with_progress(files: Dict[str, Path], progress, task_id) -> Tuple[List[Tuple], List]:
    """Prepare files for upload with progress tracking.

    Args:
        files: Dictionary mapping file keys to Path objects
        progress: Rich Progress instance
        task_id: ID of the progress task to update

    Returns:
        Tuple containing (files_list for requests, file_handles to close later)
    """
    # Initialize mimetypes
    mimetypes.init()

    files_list = []
    file_handles = []

    for file_key, file_path in files.items():
        # Determine content type using mimetypes module
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = "application/octet-stream"  # Default if type cannot be determined

        # Get file size and open file
        file_size = file_path.stat().st_size
        file_obj = open(file_path, 'rb')
        file_handles.append(file_obj)

        # Create progress tracking wrapper
        progress_reader = ProgressFileReader(file_obj, progress, task_id, file_size)

        # Add to files list with repeating 'files' field name
        files_list.append(('files', (file_path.name, progress_reader, content_type)))

    return files_list, file_handles


def upload_files_with_progress(
    url: str, files: Dict[str, Path], form_data: Optional[Dict[str, Any]] = None, verify_ssl: bool = True
) -> Tuple[Any, float]:
    """Upload files with progress bar tracking.

    Args:
        url: The URL to upload to
        files: Dictionary mapping file keys to Path objects
        form_data: Optional form data to include with the request
        verify_ssl: Whether to verify SSL certificates

    Returns:
        Tuple containing (response json, elapsed time in seconds)
    """
    if not form_data:
        form_data = {}

    # Calculate total file size
    total_size_bytes = sum(file_path.stat().st_size for file_path in files.values())
    total_size_mb = total_size_bytes / (1024 * 1024)

    console.print(f"Uploading {len(files)} files ({total_size_mb:.2f} MB)...")

    start_time = time.time()
    result = None
    file_handles: List[io.BufferedReader] = []

    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Create a single progress bar for all files
            upload_task = progress.add_task(f"[cyan]Uploading {len(files)} files", total=total_size_bytes, completed=0)

            # Prepare tracking wrappers for each file
            files_list, file_handles = prepare_files_with_progress(files, progress, upload_task)

            # Upload files with progress tracking
            response = requests.post(
                url,
                data=form_data,  # Form fields
                files=files_list,  # Multiple files as a list of tuples
                verify=verify_ssl,
            )

            response.raise_for_status()
            result = response.json()

    except requests.ConnectionError:
        show_error_and_abort("Server is not responding.")
    except requests.HTTPError:
        try:
            error_detail = response.json().get("detail", "Unknown error")
            show_error_and_abort(f"Server error: {error_detail}")
        except:
            show_error_and_abort(f"Server error: {response.text}")
    except Exception as e:
        show_error_and_abort(f"Unexpected error: {str(e)}")
    finally:
        # Close all file handles
        for file_obj in file_handles:
            try:
                file_obj.close()
            except:
                pass

    elapsed_time = time.time() - start_time
    console.print(f"Upload and execution completed in {elapsed_time:.2f} seconds")

    return result, elapsed_time
