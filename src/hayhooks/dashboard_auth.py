"""A private, same-user credential for local dashboard connections."""
# ruff: noqa: EM101, PLR2004

import os
import secrets
import stat
import tempfile
from contextlib import suppress
from pathlib import Path

from fastapi import HTTPException, Request

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def local_dashboard_token(*, create: bool = False) -> str | None:
    if not hasattr(os, "getuid"):
        return None  # Explicit credentials are required where POSIX ownership is unavailable.
    directory = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hayhooks"
    path = directory / "dashboard-token"
    if create:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not directory.exists():
        return None
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise PermissionError("Dashboard credential directory must be private to the current user")
    if create:
        # Publish a complete credential atomically; concurrent workers reuse the winner.
        with tempfile.NamedTemporaryFile(mode="w", dir=directory) as temporary:
            temporary.write(secrets.token_urlsafe(32))
            temporary.flush()
            with suppress(FileExistsError):
                os.link(temporary.name, path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor) as credential:
        info = os.fstat(credential.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise PermissionError("Dashboard credential must be private to the current user")
        token = credential.read(129).strip()
        if not 32 <= len(token) <= 128 or not all(char.isalnum() or char in "-_" for char in token):
            raise ValueError("Invalid local dashboard credential")
        return token


def require_tracing_access(request: Request) -> None:
    """Authenticate process-wide operator inspection; hosts may override this dependency."""
    from hayhooks.settings import settings

    if settings.tracing_token:
        token = settings.tracing_token.get_secret_value()
    elif getattr(request.state, "tracing_operator", False) is True:
        return
    else:
        token = getattr(request.app.state, "dashboard_token", None)
    if not token:
        raise HTTPException(status_code=403, detail="Authenticated tracing/operator dependency is required")
    expected = "Bearer " + token
    if not secrets.compare_digest(request.headers.get("Authorization", "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid tracing credentials")
