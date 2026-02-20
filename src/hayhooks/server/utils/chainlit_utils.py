"""
Chainlit integration utilities for Hayhooks.

This module provides utilities to mount Chainlit as an embedded frontend
for Hayhooks, allowing users to interact with deployed pipelines through
a chat interface.
"""

import os
from pathlib import Path

from hayhooks.server.logger import log

# Path to the default Chainlit app directory and file
DEFAULT_CHAINLIT_APP_DIR = Path(__file__).parent.parent / "chainlit_app"
DEFAULT_CHAINLIT_APP = DEFAULT_CHAINLIT_APP_DIR / "app.py"


def is_chainlit_available() -> bool:
    """
    Check if Chainlit is available.

    Returns:
        bool: True if Chainlit is installed, False otherwise.
    """
    try:
        import chainlit  # noqa: F401  # ty: ignore[unresolved-import]

        return True
    except ImportError:
        return False


def mount_chainlit_app(
    app,
    target: str | Path | None = None,
    path: str = "/ui",
) -> None:
    """
    Mount a Chainlit app as a sub-application on a FastAPI app.

    This allows users to access a chat interface at the specified path
    that communicates with Hayhooks' OpenAI-compatible endpoints.

    Args:
        app: FastAPI application instance to mount Chainlit on.
        target: Path to the Chainlit app file. If None, uses the default app.
        path: URL path where Chainlit will be mounted (default: "/ui").

    Raises:
        ImportError: If Chainlit is not installed.

    Example:
        >>> from fastapi import FastAPI
        >>> from hayhooks.server.utils.chainlit_utils import mount_chainlit_app
        >>> app = FastAPI()
        >>> mount_chainlit_app(app, path="/chat")
    """
    try:
        from chainlit.utils import mount_chainlit  # ty: ignore[unresolved-import]
    except ImportError as e:
        msg = "Run 'pip install \"hayhooks[ui]\"' to install Chainlit UI support."
        raise ImportError(msg) from e

    if target is None:
        target = str(DEFAULT_CHAINLIT_APP)
        app_root = str(DEFAULT_CHAINLIT_APP_DIR)
    else:
        target = str(target)
        app_root = str(Path(target).parent)

    # Verify the target file exists
    target_path = Path(target)
    if not target_path.exists():
        msg = f"Chainlit app file not found: {target}"
        raise FileNotFoundError(msg)

    # Ensure CHAINLIT_APP_ROOT points to the correct directory for theme/config
    current_root = os.environ.get("CHAINLIT_APP_ROOT", "")
    if current_root != app_root:
        os.environ["CHAINLIT_APP_ROOT"] = app_root
        log.debug("Set CHAINLIT_APP_ROOT to '{}'", app_root)


    log.info("Mounting Chainlit UI at path '{}' using app: {}", path, target)

    mount_chainlit(app=app, target=target, path=path)

    log.info("Chainlit UI successfully mounted at '{}'", path)
