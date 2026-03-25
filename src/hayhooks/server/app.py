import os
import sys
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from os import PathLike
from pathlib import Path

# Set CHAINLIT_APP_ROOT before any Chainlit imports (must be done before import)
# ruff: noqa: E402

_chainlit_app_dir = Path(__file__).parent / "chainlit_app"
if _chainlit_app_dir.exists():
    os.environ.setdefault("CHAINLIT_APP_ROOT", str(_chainlit_app_dir))

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from hayhooks.server.logger import log, log_elapsed
from hayhooks.server.routers import deploy_router, draw_router, openai_router, status_router, undeploy_router
from hayhooks.server.utils.deploy_utils import (
    commit_prepared_pipeline,
    deploy_pipeline_files,
    deploy_pipeline_yaml,
    prepare_pipeline_files,
    prepare_pipeline_yaml,
    read_pipeline_files_from_dir,
    rebuild_openapi,
)
from hayhooks.server.utils.models import PreparedPipeline
from hayhooks.settings import APP_DESCRIPTION, APP_TITLE, StartupDeployStrategy, check_cors_settings, settings


def deploy_yaml_pipeline(app: FastAPI, pipeline_file_path: Path) -> dict:
    """
    Deploy a pipeline from a YAML file.

    Args:
        app: FastAPI application instance
        pipeline_file_path: Path to the YAML pipeline definition

    Returns:
        dict: Deployment result containing pipeline name
    """
    name = pipeline_file_path.stem
    with open(pipeline_file_path, encoding="utf-8") as pipeline_file:
        source_code = pipeline_file.read()

    deployed_pipeline = deploy_pipeline_yaml(pipeline_name=name, source_code=source_code, app=app)
    log.info("Deployed pipeline from YAML: '{}'", deployed_pipeline["name"])

    return deployed_pipeline


def deploy_files_pipeline(app: FastAPI, pipeline_dir: Path) -> dict | None:
    """
    Deploy a pipeline from a directory containing multiple files.

    Args:
        app: FastAPI application instance
        pipeline_dir: Path to the pipeline directory

    Returns:
        dict: Deployment result containing pipeline name
    """
    files = read_pipeline_files_from_dir(pipeline_dir)

    if files:
        deployed_pipeline = deploy_pipeline_files(
            app=app, pipeline_name=pipeline_dir.name, files=files, save_files=False
        )
        log.info("Deployed pipeline from dir: '{}'", pipeline_dir)
        return deployed_pipeline
    else:
        log.warning("No files found in pipeline directory: '{}'", pipeline_dir)
        return None


def init_pipeline_dir(pipelines_dir: PathLike | str) -> str:
    """
    Create a directory for pipelines if it doesn't exist.

    If the directory doesn't exist, it will be created.
    If the directory exists but is not a directory, an error will be raised.

    Args:
        pipelines_dir: Path to the pipelines directory

    Returns:
        str: Path to the pipelines directory
    """
    pipelines_dir = Path(pipelines_dir)

    if not pipelines_dir.exists():
        log.info("Creating pipelines dir: '{}'", pipelines_dir)
        pipelines_dir.mkdir(parents=True, exist_ok=True)

    if not pipelines_dir.is_dir():
        msg = f"pipelines_dir '{pipelines_dir}' exists but is not a directory"
        raise ValueError(msg)

    return str(pipelines_dir)


def _deploy_pipelines_sequential(app: FastAPI, yaml_files: list[Path], pipeline_dirs: list[Path]) -> int:
    """Deploy pipelines one at a time (original behaviour). Returns count of deployed."""
    deployed = 0
    for pipeline_file_path in yaml_files:
        try:
            deploy_yaml_pipeline(app, pipeline_file_path)
            deployed += 1
        except Exception as e:
            log.warning("Skipping pipeline file '{}': {}", pipeline_file_path, e)

    for pipeline_dir in pipeline_dirs:
        try:
            deploy_files_pipeline(app, pipeline_dir)
            deployed += 1
        except Exception as e:
            log.warning("Skipping pipeline directory '{}': {}", pipeline_dir, e)
    return deployed


def _prepare_one(path: Path) -> PreparedPipeline | None:
    """Prepare a single pipeline from a YAML file or a directory. Thread-safe."""
    if path.is_file():
        return prepare_pipeline_yaml(path.stem, source_code=path.read_text(encoding="utf-8"))

    files = read_pipeline_files_from_dir(path)
    if not files:
        log.warning("No files found in pipeline directory: '{}'", path)
        return None
    return prepare_pipeline_files(path.name, files=files, save_files=False)


def _safe_prepare(path: Path) -> PreparedPipeline | None:
    try:
        return _prepare_one(path)
    except Exception as e:
        log.warning("Skipping pipeline '{}' (prepare failed): {}", path, e)
        return None


def _deploy_pipelines_parallel(app: FastAPI, yaml_files: list[Path], pipeline_dirs: list[Path]) -> int:
    """
    Deploy pipelines with parallel prepare + serial commit.

    The expensive work (file I/O, YAML/module loading, wrapper ``setup()``) runs
    in a bounded thread pool.  Results are committed one-by-one on the calling
    thread so ``app.routes`` and the registry are never mutated concurrently.
    The OpenAPI schema is rebuilt exactly once after all commits.
    """
    max_workers = max(1, settings.startup_deploy_workers)
    sources: list[Path] = [*yaml_files, *pipeline_dirs]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        prepared = [pipeline for pipeline in pool.map(_safe_prepare, sources) if pipeline is not None]

    deployed = 0
    for p in prepared:
        try:
            commit_prepared_pipeline(p, app=app, _defer_openapi_rebuild=True)
            deployed += 1
        except Exception as e:
            log.warning("Skipping pipeline '{}' (commit failed): {}", p.name, e)

    if deployed:
        rebuild_openapi(app)

    return deployed


@log_elapsed("INFO")
def deploy_pipelines(app: FastAPI, pipelines_dir: PathLike | str) -> None:
    """
    Deploy all pipelines from the specified directory.

    Respects ``startup_deploy_strategy``:
    - *sequential*: deploy one pipeline at a time (original behaviour).
    - *parallel* (default): deploy in a bounded thread pool, rebuild OpenAPI once.

    Args:
        app: FastAPI application instance
        pipelines_dir: Path to the pipelines directory
    """
    pipelines_dir = init_pipeline_dir(pipelines_dir)
    log.info("Pipelines dir set to: '{}'", pipelines_dir)
    pipelines_path = Path(pipelines_dir)

    yaml_files = list(pipelines_path.glob("*.y*ml"))
    pipeline_dirs = [d for d in pipelines_path.iterdir() if d.is_dir()]

    total = len(yaml_files) + len(pipeline_dirs)
    if total == 0:
        return

    strategy = settings.startup_deploy_strategy
    is_parallel = strategy == StartupDeployStrategy.PARALLEL
    deploy_fn = _deploy_pipelines_parallel if is_parallel else _deploy_pipelines_sequential

    log.info("Deploying {} pipeline(s) using '{}' strategy", total, strategy.value)
    deployed = deploy_fn(app, yaml_files, pipeline_dirs)
    log.info("Startup deploy complete: {}/{} pipelines deployed", deployed, total)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if settings.pipelines_dir:
        deploy_pipelines(app, settings.pipelines_dir)

    yield


@lru_cache(maxsize=1)
def get_package_version() -> str:
    """
    Get the version of the package using package metadata.
    """
    try:
        from importlib.metadata import version

        version_str = version("hayhooks")
        # Fallback to a safe default if metadata lookup returns empty/None
        if not version_str or not isinstance(version_str, str):
            msg = "Invalid package version metadata"
            raise ValueError(msg)
        log.debug("Version from package metadata: {}", version_str)
        return version_str
    except Exception as e:
        log.debug("Could not get version from package metadata: {}", e)

    # Return a PEP440-compliant default
    return "0.0.0"


def create_app() -> FastAPI:
    """
    Create and configure a FastAPI application.

    This function initializes a FastAPI application with the following features:
    - Configures root path from settings if provided
    - Includes all router endpoints (status, draw, deploy, undeploy)

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    if additional_path := settings.additional_python_path:
        sys.path.append(additional_path)
        log.trace("Added '{}' to sys.path", additional_path)

    app_params: dict = {
        "lifespan": lifespan,
        "title": APP_TITLE,
        "description": APP_DESCRIPTION,
        "version": get_package_version(),
    }

    if root_path := settings.root_path:
        app_params["root_path"] = root_path

    app = FastAPI(**app_params)

    # Check CORS settings before adding middleware
    check_cors_settings()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,  # ty: ignore[invalid-argument-type]
        allow_origins=settings.cors_allow_origins,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
        allow_credentials=settings.cors_allow_credentials,
        allow_origin_regex=settings.cors_allow_origin_regex,
        expose_headers=settings.cors_expose_headers,
        max_age=settings.cors_max_age,
    )

    # Include all routers
    app.include_router(status_router)
    app.include_router(draw_router)
    app.include_router(deploy_router)
    app.include_router(undeploy_router)
    app.include_router(openai_router)

    # Mount Chainlit UI if enabled
    if settings.chainlit_enabled:
        _mount_chainlit_ui(app)

    return app


def run_app(
    app: FastAPI,
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """
    Run a Hayhooks FastAPI application with Uvicorn.

    Reads host/port defaults from Hayhooks settings when not provided.
    For multi-worker or auto-reload setups, use the ``hayhooks run`` CLI
    or call ``uvicorn.run()`` directly with a string import path.

    Args:
        app: A FastAPI application (e.g. from ``create_app()``).
        host: Bind address. Defaults to ``settings.host``.
        port: Bind port. Defaults to ``settings.port``.
    """
    import uvicorn

    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
    )


def _mount_chainlit_ui(app: FastAPI) -> None:
    """
    Mount Chainlit UI as a sub-application if enabled and available.

    Args:
        app: FastAPI application instance
    """
    from hayhooks.server.utils.chainlit_utils import is_chainlit_available, mount_chainlit_app

    if not is_chainlit_available():
        log.warning("Chainlit UI is enabled but not installed. Install with: pip install 'hayhooks[chainlit]'")
        return

    try:
        custom_app = settings.chainlit_app if settings.chainlit_app else None
        mount_chainlit_app(app, target=custom_app, path=settings.chainlit_path)
    except Exception as e:
        log.error("Failed to mount Chainlit UI: {}", e)
        if settings.show_tracebacks:
            import traceback

            log.error(traceback.format_exc())
