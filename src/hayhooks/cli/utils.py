import builtins
import contextlib
import io
import mimetypes
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Literal, TypeVar, overload
from urllib.parse import urljoin

import requests
import typer
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

_console = None


def get_console():
    global _console  # noqa: PLW0603
    if _console is None:
        from rich.console import Console

        from hayhooks.cli.theme import hayhooks_theme

        _console = Console(theme=hayhooks_theme)
    return _console


@contextlib.contextmanager
def padded_output() -> Generator[None, None, None]:
    """Add a blank line before and after the wrapped output."""
    console = get_console()
    console.print()
    try:
        yield
    finally:
        console.print()


def get_server_url(host: str, port: int, https: bool = False) -> str:
    if https:
        return f"https://{host}:{port}"
    else:
        return f"http://{host}:{port}"


# We need to overload the make_request function to handle whether the response is a dictionary or a streaming response.
# The streaming response is a requests.Response object, while the dictionary response is a dict[str, Any].
# We use the overloads to make the type checker happy.


@overload
def make_request(
    host: str,
    port: int,
    endpoint: str,
    method: str = "GET",
    json: dict[str, Any] | None = None,
    use_https: bool = False,
    disable_ssl: bool = False,
    stream: Literal[False] = False,
) -> dict[str, Any]: ...


@overload
def make_request(
    host: str,
    port: int,
    endpoint: str,
    method: str = "GET",
    json: dict[str, Any] | None = None,
    use_https: bool = False,
    disable_ssl: bool = False,
    stream: Literal[True] = ...,
) -> requests.Response: ...


def make_request(  # noqa: PLR0913
    host: str,
    port: int,
    endpoint: str,
    method: str = "GET",
    json: dict[str, Any] | None = None,
    use_https: bool = False,
    disable_ssl: bool = False,
    stream: bool = False,
) -> dict[str, Any] | requests.Response:
    """
    Make HTTP request to Hayhooks server with error handling.

    Args:
        host: Server hostname
        port: Server port
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc)
        json: Optional JSON payload
        use_https: Whether to use HTTPS for the connection.
        disable_ssl: Whether to disable SSL certificate verification.
        stream: Whether to return a streaming response (returns Response object instead of dict)
    """
    server_url = get_server_url(host=host, port=port, https=use_https)
    url = urljoin(server_url, endpoint)

    try:
        response = requests.request(method=method, url=url, json=json, verify=not disable_ssl, stream=stream)  # noqa: S113
        response.raise_for_status()

        if stream:
            return response

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        elif "text/plain" in content_type:
            show_warning(
                "This endpoint returns a streaming response. "
                "Use --stream flag to see tokens as they arrive."
            )
            return {"result": response.text}
        else:
            try:
                return response.json()
            except ValueError:
                return {"result": response.text}

    except requests.ConnectionError as connection_error:
        show_error("Hayhooks server is not responding. To start one, run `hayhooks run`.")
        raise typer.Abort() from connection_error
    except requests.HTTPError as http_error:
        try:
            error_detail = response.json().get("detail", "Unknown error")
        except ValueError:
            error_detail = response.text or "Unknown error"
        show_error(f"Server error: {error_detail}")
        raise typer.Abort() from http_error
    except Exception as e:
        show_error(f"Unexpected error: {e!s}")
        raise typer.Abort() from e


def show_error(message: str) -> None:
    """Display an error message with a prefix."""
    from hayhooks.cli.theme import ERROR_PREFIX

    get_console().print(f"{ERROR_PREFIX} [error]{message}[/error]")


def show_error_and_abort(message: str, highlight: str = "") -> None:
    """Display error message and abort."""
    if highlight:
        message = message.replace(highlight, f"[error.bold]{highlight}[/error.bold]")
    show_error(message)
    raise typer.Abort()


def show_success(message: str) -> None:
    """Display a success message with a prefix."""
    from hayhooks.cli.theme import SUCCESS_PREFIX

    get_console().print(f"{SUCCESS_PREFIX} {message}")


def show_warning(message: str) -> None:
    """Display a warning message with a prefix."""
    from hayhooks.cli.theme import WARNING_PREFIX

    get_console().print(f"{WARNING_PREFIX} [warning]{message}[/warning]")


T = TypeVar("T")


def with_progress_spinner(description: str, operation: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Execute an operation with a progress spinner.

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
        console=get_console(),
    ) as progress:
        progress.add_task(description=description, total=None)
        return operation(*args, **kwargs)


class ProgressFileReader:
    """File-like object wrapper that updates progress bar when read."""

    def __init__(self, file_obj: io.IOBase, progress: Progress, task_id: TaskID, file_size: int) -> None:
        self.file_obj = file_obj
        self.progress = progress
        self.task_id = task_id
        self.file_size = file_size
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.file_obj.read(size)
        chunk_size = len(chunk)
        self.bytes_read += chunk_size
        self.progress.update(self.task_id, advance=chunk_size)
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.file_obj.seek(offset, whence)

    def tell(self) -> int:
        return self.file_obj.tell()

    def close(self) -> None:
        return self.file_obj.close()


def prepare_files_with_progress(
    files: dict[str, Path], progress: Progress, task_id: TaskID
) -> tuple[list[tuple[str, tuple[str, ProgressFileReader, str]]], list[io.BufferedReader]]:
    """
    Prepare files for upload with progress tracking.

    Args:
        files: Dictionary mapping file keys to Path objects
        progress: Rich Progress instance
        task_id: ID of the progress task to update

    Returns:
        Tuple containing (files_list for requests, file_handles to close later)
    """
    mimetypes.init()

    files_list = []
    file_handles = []

    for file_path in files.values():
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = "application/octet-stream"

        file_size = file_path.stat().st_size
        file_obj = file_path.open("rb")
        file_handles.append(file_obj)

        progress_reader = ProgressFileReader(file_obj, progress, task_id, file_size)
        files_list.append(("files", (file_path.name, progress_reader, content_type)))

    return files_list, file_handles


def upload_files_with_progress(
    url: str, files: dict[str, Path], form_data: dict[str, Any] | None = None, verify_ssl: bool = True
) -> tuple[dict[str, Any], float]:
    """
    Upload files with progress bar tracking.

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

    total_size_bytes = sum(file_path.stat().st_size for file_path in files.values())
    total_size_mb = total_size_bytes / (1024 * 1024)

    get_console().print(f"[muted]Uploading {len(files)} files ({total_size_mb:.2f} MB)...[/muted]")

    start_time = time.time()
    result: dict[str, Any] = {}
    file_handles: list[io.BufferedReader] = []

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=get_console(),
        ) as progress:
            upload_task = progress.add_task(
                f"[accent]Uploading {len(files)} files", total=total_size_bytes, completed=0
            )

            files_list, file_handles = prepare_files_with_progress(files, progress, upload_task)

            response = requests.post(  # noqa: S113
                url,
                data=form_data,
                files=files_list,
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
        except Exception:
            show_error_and_abort(f"Server error: {response.text}")
    except Exception as e:
        show_error_and_abort(f"Unexpected error: {e!s}")
    finally:
        for file_obj in file_handles:
            with contextlib.suppress(builtins.BaseException):
                file_obj.close()

    elapsed_time = time.time() - start_time
    get_console().print(f"[muted]Completed in {elapsed_time:.2f}s[/muted]")

    return result, elapsed_time
