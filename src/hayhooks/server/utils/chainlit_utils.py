"""
Chainlit integration utilities for Hayhooks.

This module provides utilities to mount Chainlit as an embedded frontend
for Hayhooks, allowing users to interact with deployed pipelines through
a chat interface.
"""

from pathlib import Path

from haystack.lazy_imports import LazyImport

from hayhooks.server.logger import log

# Lazily import Chainlit so the optional dependency is only required when used
with LazyImport("Run 'pip install \"hayhooks[ui]\"' to install Chainlit UI support.") as chainlit_import:
    from chainlit.utils import mount_chainlit


# Path to the default Chainlit app
DEFAULT_CHAINLIT_APP = Path(__file__).parent.parent / "chainlit_app" / "app.py"


def is_chainlit_available() -> bool:
    """
    Check if Chainlit is available.

    Returns:
        bool: True if Chainlit is installed, False otherwise.
    """
    try:
        chainlit_import.check()
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
    chainlit_import.check()

    target = str(DEFAULT_CHAINLIT_APP) if target is None else str(target)

    # Verify the target file exists
    target_path = Path(target)
    if not target_path.exists():
        msg = f"Chainlit app file not found: {target}"
        raise FileNotFoundError(msg)

    log.info("Mounting Chainlit UI at path '{}' using app: {}", path, target)

    mount_chainlit(app=app, target=target, path=path)

    log.info("Chainlit UI successfully mounted at '{}'", path)
