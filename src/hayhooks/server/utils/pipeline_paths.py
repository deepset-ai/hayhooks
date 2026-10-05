from pathlib import Path, PureWindowsPath

from hayhooks.server.exceptions import PipelinePathError


def validate_pipeline_name(name: str) -> str:
    """Require a single directory name on both POSIX and Windows."""
    if (
        not name
        or name in {".", ".."}
        or any(char in name for char in ("/", "\\", "\0"))
        or PureWindowsPath(name).drive
    ):
        msg = "Pipeline name must be a single non-empty name without path separators"
        raise PipelinePathError(msg)
    return name


def validate_file_keys(files: dict[str, str]) -> dict[str, str]:
    """Allow relative nested files, rejecting traversal and Windows paths on all hosts."""
    for name in files:
        if (
            not name
            or "\\" in name
            or "\0" in name
            or PureWindowsPath(name).drive
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            msg = f"Pipeline file must be a relative path without traversal: {name!r}"
            raise PipelinePathError(msg)
    return files


def require_contained_path(path: Path, root: Path) -> Path:
    """Check the resolved target, including existing and dangling symlinks, before I/O."""
    try:
        resolved, resolved_root = path.resolve(), root.resolve()
    except (OSError, RuntimeError) as error:
        msg = f"Cannot resolve pipeline path: {path}"
        raise PipelinePathError(msg) from error
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        msg = f"Pipeline path escapes its directory: {path}"
        raise PipelinePathError(msg)
    return path


def validate_pipeline_files(pipeline_name: str, files: dict[str, str], pipelines_dir: str) -> None:
    """Validate the entire upload before any existing source can be moved or removed."""
    validate_pipeline_name(pipeline_name)
    validate_file_keys(files)
    root = Path(pipelines_dir)
    pipeline_dir = require_contained_path(root / pipeline_name, root)
    for name in files:
        require_contained_path(pipeline_dir / name, pipeline_dir)
