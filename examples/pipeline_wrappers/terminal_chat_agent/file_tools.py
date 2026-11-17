import os
import stat
import tempfile
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

from haystack.tools import Tool

MAX_TEXT_BYTES = 65_536
MAX_LIST_ENTRIES = 200


def build_file_tools(root: Path) -> list[Tool]:  # noqa: C901, PLR0915
    try:
        root = root.expanduser().resolve(strict=True)
        if not root.is_dir():
            message = "The workspace must be an existing directory."
            raise ValueError(message)
    except OSError as exc:
        message = "The workspace must be an existing, accessible directory."
        raise ValueError(message) from exc

    @contextmanager
    def operation(name: str, path: str):
        try:
            yield
        except FileNotFoundError as exc:
            message = f"{name} {path!r}: file or parent directory not found. Use list_files to find a valid path."
            raise ValueError(message) from exc
        except PermissionError as exc:
            message = f"{name} {path!r}: permission denied. Check server-side workspace permissions."
            raise ValueError(message) from exc
        except UnicodeError as exc:
            message = f"{name} {path!r}: only UTF-8 text is supported. Use a UTF-8 text file or valid text content."
            raise ValueError(message) from exc
        except OSError as exc:
            detail = (
                "Write completion is unknown; inspect the file before retrying."
                if name == "write_file"
                else "Check the path and server-side workspace access."
            )
            message = f"{name} {path!r}: filesystem operation failed. {detail}"
            raise ValueError(message) from exc

    def checked_path(path: str, name: str) -> Path:
        if "\x00" in path:
            message = f"{name} {path!r}: invalid path. Use a relative path without null characters."
            raise ValueError(message)
        relative = Path(path)
        if relative.anchor or ".." in relative.parts:
            message = f"{name} {path!r}: use a relative path within the configured workspace; traversal is not allowed."
            raise ValueError(message)
        target = root
        for part in relative.parts:
            target /= part
            if target.is_symlink():
                message = f"{name} {path!r}: symlinks are not allowed. Use a path directly within the workspace."
                raise ValueError(message)
        return target

    def require_file(target: Path, path: str, name: str) -> None:
        if not stat.S_ISREG(target.stat().st_mode):
            message = f"{name} {path!r}: only regular UTF-8 text files are supported. Use list_files to choose a file."
            raise ValueError(message)

    def list_files(path: str = ".") -> str:
        with operation("list_files", path):
            target = checked_path(path, "list_files")
            if not target.is_dir():
                if not target.exists():
                    raise FileNotFoundError(path)
                message = f"list_files {path!r}: expected a directory. Use list_files on its parent directory."
                raise ValueError(message)
            with os.scandir(target) as entries:
                selected = list(islice(entries, MAX_LIST_ENTRIES + 1))
            truncated = len(selected) > MAX_LIST_ENTRIES
            lines = []
            for entry in sorted(selected[:MAX_LIST_ENTRIES], key=lambda entry: entry.name):
                suffix = " [symlink]" if entry.is_symlink() else "/" if entry.is_dir(follow_symlinks=False) else ""
                lines.append(entry.name + suffix)
            if truncated:
                lines.append(f"[Truncated: showing {MAX_LIST_ENTRIES} entries; more entries exist.]")
            return "\n".join(lines) if lines else "[Empty directory]"

    def read_file(path: str) -> str:
        with operation("read_file", path):
            target = checked_path(path, "read_file")
            require_file(target, path, "read_file")
            with target.open("rb") as source:
                data = source.read(MAX_TEXT_BYTES + 1)
            if len(data) > MAX_TEXT_BYTES:
                message = f"read_file {path!r}: text exceeds the 65,536-byte limit. Choose a smaller UTF-8 file."
                raise ValueError(message)
            return data.decode("utf-8")

    def write_file(path: str, content: str) -> str:
        with operation("write_file", path):
            data = content.encode("utf-8")
            if len(data) > MAX_TEXT_BYTES:
                message = f"write_file {path!r}: content exceeds the 65,536-byte limit. Send less UTF-8 text."
                raise ValueError(message)
            target = checked_path(path, "write_file")
            if target.exists():
                require_file(target, path, "write_file")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=target.parent, prefix=".hayhooks-", delete=False
                ) as stage:
                    temporary = Path(stage.name)
                    stage.write(data)
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return f"Wrote {len(data)} UTF-8 bytes to {path!r}."

    path_property = {"type": "string", "description": "Relative path within the server workspace; no symlinks."}
    return [
        Tool(
            name="list_files",
            description="List up to 200 immediate directory entries. Reports truncation and does not follow symlinks.",
            parameters={"type": "object", "properties": {"path": path_property}},
            function=list_files,
        ),
        Tool(
            name="read_file",
            description="Read one regular UTF-8 text file, up to 65,536 encoded bytes.",
            parameters={"type": "object", "properties": {"path": path_property}, "required": ["path"]},
            function=read_file,
        ),
        Tool(
            name="write_file",
            description="Replace one UTF-8 text file, up to 65,536 encoded bytes. The parent directory must exist.",
            parameters={
                "type": "object",
                "properties": {"path": path_property, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            function=write_file,
        ),
    ]
