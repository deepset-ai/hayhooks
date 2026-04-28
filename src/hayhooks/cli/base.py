import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from fastapi import FastAPI

from hayhooks.cli.mcp import mcp
from hayhooks.cli.pipeline import pipeline
from hayhooks.cli.theme import apply_typer_theme
from hayhooks.cli.utils import get_console, get_server_url, make_request, padded_output, show_success

apply_typer_theme()

hayhooks_cli = typer.Typer(name="hayhooks", rich_markup_mode="rich")
hayhooks_cli.add_typer(pipeline, name="pipeline")
hayhooks_cli.add_typer(mcp, name="mcp")


def _set_env(key: str, value: str | None) -> None:
    """Set an environment variable if the value is truthy, so worker child processes inherit it."""
    if value:
        os.environ[key] = value


def _resolve_dashboard_source_dir(dashboard_dist_dir: str) -> Path | None:
    """Find dashboard source directory containing package.json."""
    configured_dist = Path(dashboard_dist_dir).expanduser()
    candidates = [configured_dist.parent, *[parent / "dashboard" for parent in Path(__file__).resolve().parents]]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "package.json").exists():
            return resolved
    return None


def _build_dashboard_assets(source_dir: Path) -> Path:
    """Build dashboard frontend assets from source directory."""
    npm = shutil.which("npm")
    if npm is None:
        msg = "npm was not found in PATH. Install Node.js/npm or provide prebuilt dashboard assets."
        raise RuntimeError(msg)

    if not (source_dir / "node_modules").exists():
        install_cmd = [npm, "ci"] if (source_dir / "package-lock.json").exists() else [npm, "install"]
        subprocess.run(install_cmd, cwd=source_dir, check=True)  # noqa: S603

    subprocess.run([npm, "run", "build"], cwd=source_dir, check=True)  # noqa: S603
    return source_dir / "dist"


def _dashboard_assets_available(dashboard_dist_dir: str) -> bool:
    """Return True when configured dashboard assets are available."""
    configured_index = Path(dashboard_dist_dir).expanduser() / "index.html"
    return configured_index.exists()


def _prepare_tracing_dashboard_assets(dashboard_dist_dir: str) -> str | None:
    """Build tracing dashboard assets when source is available."""
    source_dir = _resolve_dashboard_source_dir(dashboard_dist_dir)
    if source_dir is None:
        return None
    built_dist = _build_dashboard_assets(source_dir)
    return str(built_dist)


def get_app() -> FastAPI:
    """
    Factory function to create the FastAPI app.
    """
    # Lazy import to avoid importing FastAPI and related dependencies on CLI startup
    from hayhooks.server.app import create_app

    return create_app()


@hayhooks_cli.command()
def run(  # noqa: C901, PLR0912, PLR0913, PLR0915
    host: Annotated[str | None, typer.Option("--host", "-h", help="Host to run the server on")] = None,
    port: Annotated[int | None, typer.Option("--port", "-p", help="Port to run the server on")] = None,
    pipelines_dir: Annotated[
        str | None, typer.Option("--pipelines-dir", "-d", help="Directory containing the pipelines")
    ] = None,
    root_path: Annotated[str | None, typer.Option(help="Root path of the server")] = None,
    additional_python_path: Annotated[
        str | None, typer.Option(help="Additional Python path to add to sys.path")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of workers to run the server with")] = 1,
    reload: Annotated[
        bool, typer.Option("--reload", "-r", help="Whether to reload the server on file changes")
    ] = False,
    with_chainlit: Annotated[
        bool, typer.Option("--with-chainlit", help="Enable embedded Chainlit UI (requires hayhooks[chainlit])")
    ] = False,
    with_tracing_dashboard: Annotated[
        bool,
        typer.Option(
            "--with-tracing-dashboard",
            "--with-dashboard",
            "--tracing-dashboard",
            help="Enable tracing dashboard UI and build dashboard assets when source is available",
        ),
    ] = False,
    chainlit_path: Annotated[
        str | None, typer.Option("--chainlit-path", help="URL path for the Chainlit UI (default: /chat)")
    ] = None,
    chainlit_custom_elements_dir: Annotated[
        str | None,
        typer.Option(
            "--chainlit-custom-elements-dir",
            help="Directory with custom .jsx element files for the Chainlit UI",
        ),
    ] = None,
    dashboard_path: Annotated[
        str | None, typer.Option("--dashboard-path", help="URL path for the tracing dashboard (default: /dashboard)")
    ] = None,
) -> None:
    """
    Run the Hayhooks server.
    """
    from hayhooks.server.logger import log
    from hayhooks.settings import settings

    # Fill defaults from settings only when command is executed
    host = host or settings.host
    port = port or settings.port
    pipelines_dir = pipelines_dir or settings.pipelines_dir
    root_path = root_path or settings.root_path

    # Propagate CLI overrides via env vars so that worker child processes
    # (spawned by uvicorn when workers > 1) pick them up when they create
    # their own AppSettings instance.
    _set_env("HAYHOOKS_HOST", host)
    _set_env("HAYHOOKS_PORT", str(port))
    _set_env("HAYHOOKS_PIPELINES_DIR", pipelines_dir)
    _set_env("HAYHOOKS_ROOT_PATH", root_path)

    settings.host = host
    settings.port = port
    settings.pipelines_dir = pipelines_dir
    settings.root_path = root_path

    if additional_python_path:
        _set_env("HAYHOOKS_ADDITIONAL_PYTHON_PATH", additional_python_path)
        settings.additional_python_path = additional_python_path
        sys.path.append(additional_python_path)
        log.trace("Added '{}' to sys.path", additional_python_path)

    # Handling Chainlit CLI flags
    if chainlit_path and not with_chainlit:
        log.warning("--chainlit-path was provided but --with-chainlit is not set. The UI will not be mounted.")

    if chainlit_custom_elements_dir and not with_chainlit:
        log.warning(
            "--chainlit-custom-elements-dir was provided but --with-chainlit is not set. "
            "Custom elements will not be loaded."
        )

    if dashboard_path and not with_tracing_dashboard:
        log.warning(
            "--dashboard-path was provided but --with-tracing-dashboard is not set. The UI will not be mounted."
        )

    if with_chainlit:
        if workers > 1:
            log.warning(
                "Chainlit UI uses WebSockets (socket.io) which requires sticky sessions. "
                "With --workers {}, requests may hit different worker processes, causing WebSocket failures. "
                "Use --workers 1 when running with --with-chainlit, "
                "or place a reverse proxy with sticky sessions in front.",
                workers,
            )
        _set_env("HAYHOOKS_CHAINLIT_ENABLED", "true")
        settings.chainlit_enabled = True

        if chainlit_path:
            _set_env("HAYHOOKS_CHAINLIT_PATH", chainlit_path)
            settings.chainlit_path = chainlit_path

        if chainlit_custom_elements_dir:
            _set_env("HAYHOOKS_CHAINLIT_CUSTOM_ELEMENTS_DIR", chainlit_custom_elements_dir)
            settings.chainlit_custom_elements_dir = chainlit_custom_elements_dir

    if with_tracing_dashboard:
        _set_env("HAYHOOKS_DASHBOARD_ENABLED", "true")
        settings.dashboard_enabled = True

        if dashboard_path:
            _set_env("HAYHOOKS_DASHBOARD_PATH", dashboard_path)
            settings.dashboard_path = dashboard_path

        try:
            built_dashboard_dist = _prepare_tracing_dashboard_assets(settings.dashboard_dist_dir)
            if built_dashboard_dist is not None:
                _set_env("HAYHOOKS_DASHBOARD_DIST_DIR", built_dashboard_dist)
                settings.dashboard_dist_dir = built_dashboard_dist
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            if _dashboard_assets_available(settings.dashboard_dist_dir):
                log.warning(
                    "Failed to build tracing dashboard assets: {}. "
                    "Falling back to available prebuilt dashboard assets.",
                    exc,
                )
            else:
                log.error(
                    "Failed to build tracing dashboard assets and no prebuilt assets were found: {}",
                    exc,
                )
                raise typer.Exit(code=1) from exc

    if workers > 1 or reload:
        # Multi-worker and auto-reload require a string import path so each
        # uvicorn worker/subprocess re-imports the app module independently.
        import uvicorn

        uvicorn.run(
            "hayhooks.server.app:create_app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            factory=True,
            log_config=None,
        )
    else:
        from hayhooks.server.app import run_app

        run_app(get_app(), host=host, port=port)


@hayhooks_cli.command()
def status(ctx: typer.Context) -> None:
    """Get the status of the Hayhooks server."""
    with padded_output():
        response = make_request(
            host=ctx.obj["host"],
            port=ctx.obj["port"],
            endpoint="status",
            use_https=ctx.obj["use_https"],
            disable_ssl=ctx.obj["disable_ssl"],
        )

        server_url = get_server_url(host=ctx.obj["host"], port=ctx.obj["port"], https=ctx.obj["use_https"])
        show_success(f"Hayhooks server is up and running at: [accent.bold]{server_url}[/accent.bold]")

        assert isinstance(response, dict), "Status endpoint must return JSON"

        if pipes := response.get("pipelines"):
            from rich import box
            from rich.table import Table

            console = get_console()
            console.print()
            console.print("[heading]Pipelines:[/heading]")

            table = Table(
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="table.header",
                padding=(0, 2),
                show_edge=False,
            )
            table.add_column("№", style="muted", width=4)
            table.add_column("Name", style="pipeline.name")
            table.add_column("Status", style="pipeline.status")

            for idx, pipeline in enumerate(pipes, 1):
                table.add_row(str(idx), pipeline, "Active")

            console.print(table)
        else:
            get_console().print("[warning]  No pipelines currently deployed[/warning]")


@hayhooks_cli.callback()
def callback(ctx: typer.Context) -> None:
    # Lazy import settings so it's only loaded on actual CLI invocation
    from hayhooks.settings import settings

    ctx.obj = {
        "host": settings.host,
        "port": settings.port,
        "disable_ssl": settings.disable_ssl,
        "use_https": settings.use_https,
    }
