import asyncio
import os
import socket
import sys
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from contextvars import Context, copy_context
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# Set CHAINLIT_APP_ROOT before any Chainlit imports (must be done before import)
# ruff: noqa: E402

_chainlit_app_dir = Path(__file__).parent / "chainlit_app"
if _chainlit_app_dir.exists():
    os.environ.setdefault("CHAINLIT_APP_ROOT", str(_chainlit_app_dir))

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from starlette.routing import Match, Mount, compile_path

from hayhooks.server.exceptions import PipelineModeError
from hayhooks.server.logger import RequestIdMiddleware, intercept_stdlib_logging, log, log_elapsed
from hayhooks.server.pipelines.loader import build_immutable_host
from hayhooks.server.pipelines.registry import ImmutablePipelineRegistry, PipelineRegistry, registry
from hayhooks.server.routers import (
    create_openai_router,
    dashboard_router,
    deploy_router,
    draw_router,
    status_router,
    undeploy_router,
)
from hayhooks.server.tracing import (
    SPAN_PIPELINE_STARTUP_DEPLOY,
    build_trace_tags,
    configure_tracing,
    instrument_fastapi_app,
    trace_durable_runner,
    trace_operation,
)
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.deploy_utils import (
    _build_run_route,
    commit_prepared_pipeline,
    deploy_pipeline_files,
    deploy_pipeline_yaml,
    prepare_pipeline_files,
    prepare_pipeline_yaml,
    read_pipeline_files_from_dir,
    rebuild_openapi,
    require_live_deployment,
)
from hayhooks.server.utils.live_trace_stream import get_trace_stream_broadcaster
from hayhooks.server.utils.models import PreparedPipeline
from hayhooks.server.utils.module_loader import inspect_durable_runner, is_durable_wrapper
from hayhooks.settings import APP_DESCRIPTION, APP_TITLE, StartupDeployStrategy, check_cors_settings, settings

if TYPE_CHECKING:
    from hayhooks.durable.store import ExecutionStore, StoreConfig

# Worker commands and connection attempts are short; redis-py 8 uses the same default.
_REDIS_TIMEOUT_SECONDS = 5.0
# An extra viewer waits this long for a pooled connection before its stream ends with an error event.
_VIEWER_POOL_WAIT_SECONDS = 1.0


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

    Raises:
        PipelineModeError: In durable mode, which never creates the directory.
    """
    require_live_deployment()
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
        except PipelineModeError:
            raise
        except Exception as e:
            log.warning("Skipping pipeline file '{}': {}", pipeline_file_path, e)

    for pipeline_dir in pipeline_dirs:
        try:
            deploy_files_pipeline(app, pipeline_dir)
            deployed += 1
        except PipelineModeError:
            raise
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
    except PipelineModeError:
        raise
    except Exception as e:
        log.warning("Skipping pipeline '{}' (prepare failed): {}", path, e)
        return None


def _prepare_with_context(source: Path, context: Context) -> PreparedPipeline | None:
    return context.run(_safe_prepare, source)


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
        contexts = [copy_context() for _ in sources]
        prepared = [pipeline for pipeline in pool.map(_prepare_with_context, sources, contexts) if pipeline is not None]

    deployed = 0
    for p in prepared:
        try:
            commit_prepared_pipeline(p, app=app, _defer_openapi_rebuild=True)
            deployed += 1
        except PipelineModeError:
            raise
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

    Raises:
        PipelineModeError: In durable mode, or if a wrapper is durable.
    """
    require_live_deployment(app)
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
    with trace_operation(
        SPAN_PIPELINE_STARTUP_DEPLOY,
        tags=build_trace_tags(
            {
                "hayhooks.transport": "startup",
                "hayhooks.deploy.strategy": strategy.value,
                "hayhooks.deploy.total": total,
                "hayhooks.deploy.workers": settings.startup_deploy_workers if is_parallel else None,
            }
        ),
    ):
        deployed = deploy_fn(app, yaml_files, pipeline_dirs)

    log.info("Startup deploy complete: {}/{} pipelines deployed", deployed, total)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Capture the running loop so synchronous span recording can wake SSE
    # subscribers via call_soon_threadsafe.
    broadcaster = get_trace_stream_broadcaster()
    broadcaster.set_loop(asyncio.get_running_loop())
    try:
        runtime = app.state.durable_runtime
        try:
            if not app.state.durable_mode and settings.pipelines_dir:
                deploy_pipelines(app, settings.pipelines_dir)
            if runtime is not None:
                await runtime.start()
            yield
        finally:
            if runtime is not None:
                try:
                    # Uvicorn drains or cancels open requests, SSE streams included, before lifespan shutdown;
                    # closing ends any stream still waiting, so none outlives its viewer client.
                    await runtime.close()
                finally:
                    # ponytail: retained threads can delay graceful shutdown past the grace period;
                    # process isolation is the upgrade path for an enforceable execution deadline.
                    await runtime.wait_drained()
                    # Reached only after drainage: retained work keeps Redis until it no longer owns claims.
                    async with AsyncExitStack() as clients:
                        for client in app.state.durable_redis_clients:
                            clients.push_async_callback(client.aclose)
    finally:
        broadcaster.clear_loop()


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

    With ``durable_mode`` enabled, the pipelines load once from ``pipelines_dir``: deploy and
    undeploy routes are absent, and durable wrappers are hosted by a fixed durable runtime.
    The mode is read once here; changing the setting later does not change a built app.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Durable mode runs wrapper setup() while building the app, so logging, shared code, and tracing come first.
    intercept_stdlib_logging(
        settings.intercepted_loggers,
        access_log_excluded_path_prefixes=settings.access_log_excluded_path_prefixes,
    )

    if additional_path := settings.additional_python_path:
        sys.path.append(additional_path)
        log.trace("Added '{}' to sys.path", additional_path)

    configure_tracing()

    if settings.durable_mode:
        return build_immutable_host(settings.pipelines_dir, _build_app)
    return _build_app(registry)


def _build_app(pipeline_registry: PipelineRegistry) -> FastAPI:
    durable_mode = isinstance(pipeline_registry, ImmutablePipelineRegistry)
    app_params: dict = {
        "lifespan": lifespan,
        "title": APP_TITLE,
        "description": APP_DESCRIPTION,
        "version": get_package_version(),
    }

    if root_path := settings.root_path:
        app_params["root_path"] = root_path

    app = FastAPI(**app_params)
    app.state.durable_mode = durable_mode
    app.state.pipeline_registry = pipeline_registry
    app.state.durable_runtime = None
    app.state.durable_redis_clients = ()
    app.state.durable_health = None

    app.add_middleware(RequestIdMiddleware)

    # Check CORS settings before adding middleware
    check_cors_settings()

    app.add_middleware(
        CORSMiddleware,
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
    if not durable_mode:
        app.include_router(deploy_router)
        app.include_router(undeploy_router)
    app.include_router(create_openai_router(pipeline_registry))
    app.include_router(dashboard_router, prefix=settings.dashboard_path)

    _mount_dashboard_ui(app)

    # Mount Chainlit UI if enabled
    if settings.chainlit_enabled:
        _mount_chainlit_ui(app)

    if isinstance(pipeline_registry, ImmutablePipelineRegistry):
        _add_immutable_pipelines(app, pipeline_registry)

    instrument_fastapi_app(app)

    return app


def _add_immutable_pipelines(app: FastAPI, pipeline_registry: ImmutablePipelineRegistry) -> None:
    """Add the fixed run routes and durable deployments after every fixed route and mount exists."""
    wrappers = {
        name: wrapper for name in pipeline_registry.get_names() if (wrapper := pipeline_registry.get(name)) is not None
    }
    # A fixed route or mount that matches a pipeline's run path would shadow the routes under /{name}/.
    if conflicts := [name for name in wrappers if _matches_existing_route(app, f"/{name}/run")]:
        msg = f"Pipeline names {conflicts} conflict with server routes or mounts; rename their definitions"
        raise PipelineModeError(msg)

    # Durable-only and chat-only wrappers have no ordinary run endpoint.
    for name, wrapper in wrappers.items():
        metadata = pipeline_registry.get_metadata(name) or {}
        if route := _build_run_route(name, wrapper, request_model=metadata.get("request_model")):
            route_kwargs, _metadata = route
            app.add_api_route(**route_kwargs)
    _add_durable_deployments(app, {name: wrapper for name, wrapper in wrappers.items() if is_durable_wrapper(wrapper)})


def _matches_existing_route(app: FastAPI, path: str) -> bool:
    scope = {"type": "http", "method": "POST", "path": path, "root_path": ""}
    path_regex, _, _ = compile_path(path)
    return any(
        route.matches(scope)[0] is not Match.NONE
        # Also catch mounts within a parameterized path, e.g. /jobs/executions/<id>.
        or (isinstance(route, Mount) and path_regex.fullmatch(route.path) is not None)
        for route in app.router.routes
    )


def _add_durable_deployments(app: FastAPI, durable_wrappers: dict[str, BasePipelineWrapper]) -> None:
    """Construct one fixed durable deployment and router per durable wrapper; nothing connects yet."""
    if not durable_wrappers:
        return

    from hayhooks.durable.fastapi import create_durable_router
    from hayhooks.durable.haystack import HaystackDurableAdapter
    from hayhooks.durable.runtime import DurableDeployment, DurableRuntime, RuntimeConfig
    from hayhooks.durable.store import MemoryExecutionStore, StoreConfig

    store_config = StoreConfig(
        lease_commit_safety_ms=settings.durable_lease_commit_safety_ms,
        terminal_ttl_seconds=settings.durable_terminal_ttl_seconds,
        max_nonterminal_executions=settings.durable_max_nonterminal_executions,
        max_payload_bytes=settings.durable_max_payload_bytes,
        max_progress_events=settings.durable_max_progress_events,
        max_progress_event_bytes=settings.durable_max_progress_event_bytes,
        max_stream_chunks=settings.durable_max_stream_chunks,
        max_stream_chunk_bytes=settings.durable_max_stream_chunk_bytes,
    )
    runtime_config = RuntimeConfig(
        worker_concurrency=settings.durable_worker_concurrency,
        poll_interval_seconds=settings.durable_poll_interval_seconds,
        maintenance_interval_seconds=settings.durable_maintenance_interval_seconds,
        shutdown_grace_seconds=settings.durable_shutdown_grace_seconds,
        lease_duration_ms=settings.durable_lease_duration_ms,
        max_run_attempts=settings.durable_max_run_attempts,
        max_application_retries=settings.durable_max_application_retries,
        retry_base_delay_seconds=settings.durable_retry_base_delay_seconds,
        retry_max_delay_seconds=settings.durable_retry_max_delay_seconds,
        release_running_on_close=settings.durable_release_running_on_shutdown,
    )

    deployments = []
    for name, wrapper in durable_wrappers.items():
        try:
            runner, request_model, result_model = inspect_durable_runner(wrapper)
            revision = cast(str, wrapper.durable_revision)
            adapter = HaystackDurableAdapter(wrapper.pipeline)
            store: ExecutionStore
            if settings.durable_store == "memory":
                store = MemoryExecutionStore(name, config=store_config)
            else:
                store = _redis_store(app, name, store_config)
            deployment = DurableDeployment(
                name,
                revision,
                store,
                request_model,
                trace_durable_runner(name, revision, adapter.kind.value, runner),
                kind=adapter.kind,
                result_model=result_model,
                resume_model=wrapper.durable_resume_model,
                adapter=adapter,
                config=runtime_config,
            )
        except Exception as error:
            msg = f"Failed to build the durable deployment of pipeline '{name}': {error}"
            raise PipelineModeError(msg) from error
        router = create_durable_router(deployment, owner_id_dependency=None)
        if conflicts := [
            route.path
            for route in router.routes
            if isinstance(route, APIRoute) and _matches_existing_route(app, f"/{name}{route.path}")
        ]:
            msg = f"Pipeline '{name}' routes {conflicts} conflict with server routes or mounts; rename its definition"
            raise PipelineModeError(msg)
        app.include_router(router, prefix=f"/{name}")
        deployments.append(deployment)
    app.state.durable_runtime = DurableRuntime(tuple(deployments))


def _redis_store(app: FastAPI, name: str, store_config: "StoreConfig") -> "ExecutionStore":
    """Build a Redis store on the app's worker and viewer clients, created on first use and unconnected."""
    # Imported first: it raises the install hint when the durable extra is missing.
    from hayhooks.durable.redis import RedisExecutionStore

    if not app.state.durable_redis_clients:
        app.state.durable_redis_clients = _redis_clients(settings.durable_redis_url)
    worker_client, viewer_client = app.state.durable_redis_clients
    return RedisExecutionStore(
        worker_client,
        name,
        viewer_client=viewer_client,
        config=store_config,
        key_prefix=settings.durable_redis_key_prefix,
    )


def _redis_clients(url: str) -> tuple[Any, Any]:
    """
    Build the worker and viewer clients with explicit timeouts; query options in *url* override them.

    Blocking SSE reads use the viewer client, so viewers cannot starve worker heartbeats of connections.
    Neither client retries commands or negotiates RESP3, on every supported redis-py version.
    """
    from redis.asyncio import BlockingConnectionPool, Redis

    from hayhooks.durable.fastapi import _STREAM_BLOCK_SECONDS

    common: dict[str, Any] = {
        "protocol": 2,
        "socket_connect_timeout": _REDIS_TIMEOUT_SECONDS,
        "socket_keepalive": True,
        "socket_keepalive_options": _keepalive_options(),
    }
    # ponytail: unbounded worker pool (redis-py 8 caps it at 100); bound HTTP concurrency upstream if Redis
    # connections must be capped, since exhausting this pool would fail heartbeats.
    worker = Redis.from_url(url, socket_timeout=_REDIS_TIMEOUT_SECONDS, max_connections=2**31, **common)
    viewer = Redis.from_pool(
        BlockingConnectionPool.from_url(
            url,
            max_connections=settings.durable_redis_max_viewers,
            timeout=_VIEWER_POOL_WAIT_SECONDS,
            socket_timeout=_STREAM_BLOCK_SECONDS + 15,
            **common,
        )
    )
    viewer_timeout = viewer.connection_pool.connection_kwargs.get("socket_timeout")
    if viewer_timeout is not None and viewer_timeout <= _STREAM_BLOCK_SECONDS:
        msg = f"socket_timeout in the durable Redis URL must exceed {_STREAM_BLOCK_SECONDS:g} seconds, the SSE block"
        raise ValueError(msg)
    return worker, viewer


def _keepalive_options() -> dict[int, int]:
    # redis-py 8's defaults, so 5-7 also detect dead idle connections; macOS names the idle option TCP_KEEPALIVE.
    idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    options = {idle: 30, getattr(socket, "TCP_KEEPINTVL", None): 5, getattr(socket, "TCP_KEEPCNT", None): 3}
    return {option: value for option, value in options.items() if option is not None}


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

    intercept_stdlib_logging(
        settings.intercepted_loggers,
        access_log_excluded_path_prefixes=settings.access_log_excluded_path_prefixes,
    )
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        timeout_graceful_shutdown=settings.graceful_shutdown_timeout,
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


def _mount_dashboard_ui(app: FastAPI) -> None:
    """
    Mount dashboard static files if enabled and available.

    Args:
        app: FastAPI application instance
    """
    if not settings.dashboard_enabled:
        return

    dashboard_dist_dir = _resolve_dashboard_dist_dir()
    if dashboard_dist_dir is None:
        return

    app.mount(
        settings.dashboard_path,
        StaticFiles(directory=str(dashboard_dist_dir), html=True),
        name="dashboard-ui",
    )


def _resolve_dashboard_dist_dir() -> Path | None:
    """Resolve configured dashboard dist directory when available."""
    configured_dist_dir = Path(settings.dashboard_dist_dir).expanduser()
    if configured_dist_dir.exists() and configured_dist_dir.is_dir():
        return configured_dist_dir

    log.warning("Dashboard UI enabled but dist dir was not found: '{}'", configured_dist_dir)
    return None
