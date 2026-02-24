"""
Chainlit integration utilities for Hayhooks.

This module provides utilities to mount Chainlit as an embedded frontend
for Hayhooks, allowing users to interact with deployed pipelines through
a chat interface.
"""

import os
import shutil
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from hayhooks.server.logger import log
from hayhooks.settings import settings

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


def _seed_public_assets(app_root: str) -> None:
    """
    Copy built-in public assets into the app root without overwriting.

    Seeds the ``public/`` directory with default logos, favicons, and theme so
    that custom Chainlit apps inherit branding automatically.  Files already
    present in the target directory are left untouched.
    """
    builtin_public = DEFAULT_CHAINLIT_APP_DIR / "public"
    if not builtin_public.is_dir():
        return

    target_public = Path(app_root) / "public"
    target_public.mkdir(parents=True, exist_ok=True)

    for src_file in builtin_public.iterdir():
        if not src_file.is_file():
            continue
        dest = target_public / src_file.name
        if dest.exists():
            continue
        shutil.copy2(src_file, dest)
        log.debug("Seeded default asset: {}", src_file.name)


def _merge_custom_elements(app_root: str) -> None:
    """
    Copy custom ``.jsx`` element files into the Chainlit ``public/elements/`` directory.

    Reads the source directory from ``settings.chainlit_custom_elements_dir``.
    Only files with a ``.jsx`` extension are copied.  If a custom file has the
    same name as a built-in element, the built-in is overridden and a warning
    is logged.
    """
    custom_dir_str = settings.chainlit_custom_elements_dir
    if not custom_dir_str:
        return

    custom_dir = Path(custom_dir_str).resolve()
    if not custom_dir.is_dir():
        log.warning("HAYHOOKS_CHAINLIT_CUSTOM_ELEMENTS_DIR '{}' is not a directory, skipping", custom_dir)
        return

    jsx_files = list(custom_dir.glob("*.jsx"))
    if not jsx_files:
        log.warning("No .jsx files found in '{}'", custom_dir)
        return

    elements_dir = Path(app_root) / "public" / "elements"
    elements_dir.mkdir(parents=True, exist_ok=True)

    for jsx_file in jsx_files:
        dest = elements_dir / jsx_file.name
        if dest.exists():
            log.warning("Custom element '{}' overrides built-in element", jsx_file.name)
        shutil.copy2(jsx_file, dest)
        log.info("Loaded custom Chainlit element: {}", jsx_file.stem)


def mount_chainlit_app(
    app,
    target: str | Path | None = None,
    path: str = "/chat",
) -> None:
    """
    Mount a Chainlit app as a sub-application on a FastAPI app.

    This allows users to access a chat interface at the specified path
    that communicates with Hayhooks' OpenAI-compatible endpoints.

    Args:
        app: FastAPI application instance to mount Chainlit on.
        target: Path to the Chainlit app file. If None, uses the default app.
        path: URL path where Chainlit will be mounted (default: "/chat").

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
        msg = "Run 'pip install \"hayhooks[chainlit]\"' to install Chainlit UI support."
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

    # Seed the public directory with built-in assets (logos, favicons, theme)
    # so that custom apps inherit them without having to copy everything manually.
    _seed_public_assets(app_root)

    # Merge user-provided custom elements into public/elements/
    _merge_custom_elements(app_root)

    log.info("Mounting Chainlit UI at path '{}' using app: {}", path, target)

    # Mount the public directory as static files for theme, logos, etc.
    public_dir = Path(app_root) / "public"
    if public_dir.exists() and public_dir.is_dir():
        app.mount("/public", StaticFiles(directory=str(public_dir)), name="chainlit_public")
        log.debug("Mounted Chainlit public directory at '/public'")

    mount_chainlit(app=app, target=target, path=path)

    log.success("Chainlit UI available at {}{}", settings.chainlit_base_url, path)
