import sys
from typing import Annotated, Literal

import typer

a2a = typer.Typer(rich_markup_mode="rich")


@a2a.command()
def run(  # noqa: C901, PLR0913
    host: Annotated[str | None, typer.Option("--host", "-h", help="Host to run the A2A server on")] = None,
    port: Annotated[int | None, typer.Option("--port", "-p", help="Port to run the A2A server on")] = None,
    pipelines_dir: Annotated[
        str | None, typer.Option("--pipelines-dir", "-d", help="Directory containing the pipelines")
    ] = None,
    additional_python_path: Annotated[
        str | None, typer.Option(help="Additional Python path to add to sys.path")
    ] = None,
    external_url: Annotated[
        str | None,
        typer.Option("--external-url", help="Base URL advertised in agent cards (e.g. behind a reverse proxy)"),
    ] = None,
    task_store_provider: Annotated[
        str | None,
        typer.Option(
            "--task-store-provider",
            help="Custom A2A TaskStoreProvider as module:ClassName",
        ),
    ] = None,
    task_store: Annotated[
        Literal["auto", "memory", "redis"] | None,
        typer.Option("--task-store", help="Built-in A2A task-store backend"),
    ] = None,
    a2a_redis_url: Annotated[
        str | None,
        typer.Option("--a2a-redis-url", help="Redis URL for the built-in A2A task store"),
    ] = None,
    a2a_redis_key_prefix: Annotated[
        str | None,
        typer.Option("--a2a-redis-key-prefix", help="Redis key prefix for the built-in A2A task store"),
    ] = None,
    execution_store: Annotated[
        Literal["memory", "redis"] | None,
        typer.Option("--execution-store", help="Built-in durable execution-store backend"),
    ] = None,
    execution_redis_url: Annotated[
        str | None,
        typer.Option("--execution-redis-url", help="Redis URL for durable execution storage"),
    ] = None,
    execution_redis_key_prefix: Annotated[
        str | None,
        typer.Option("--execution-redis-key-prefix", help="Redis key prefix for durable execution storage"),
    ] = None,
    durable_execution_concurrency: Annotated[
        int | None,
        typer.Option(
            "--durable-execution-concurrency",
            min=1,
            help="Maximum concurrent durable Agent executions per deployed agent",
        ),
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="If true, tracebacks should be returned on errors")] = False,
) -> None:
    """
    Run the A2A server, exposing deployed pipelines as A2A agents.
    """
    # Lazy imports of settings, logger and uvicorn
    import uvicorn

    from hayhooks.server.logger import intercept_stdlib_logging, log
    from hayhooks.server.utils.a2a_utils import a2a_import, create_a2a_app
    from hayhooks.server.utils.deploy_utils import deploy_pipelines
    from hayhooks.settings import settings

    a2a_import.check()

    # Fill defaults from settings when command executes
    host = host or settings.a2a_host
    port = port or settings.a2a_port
    pipelines_dir = pipelines_dir or settings.pipelines_dir

    settings.a2a_host = host
    settings.a2a_port = port
    settings.pipelines_dir = pipelines_dir
    settings.show_tracebacks = debug

    if external_url:
        settings.a2a_external_url = external_url

    if task_store_provider is not None:
        settings.a2a_task_store_provider = task_store_provider

    if task_store is not None:
        settings.a2a_task_store = task_store

    if a2a_redis_url is not None:
        settings.a2a_redis_url = a2a_redis_url

    if a2a_redis_key_prefix is not None:
        settings.a2a_redis_key_prefix = a2a_redis_key_prefix

    if execution_store is not None:
        settings.durable_store = execution_store

    if execution_redis_url is not None:
        settings.durable_redis_url = execution_redis_url

    if execution_redis_key_prefix is not None:
        settings.durable_redis_key_prefix = execution_redis_key_prefix

    if durable_execution_concurrency is not None:
        settings.durable_execution_concurrency = durable_execution_concurrency

    if task_store_provider is not None and task_store is not None:
        msg = "--task-store and --task-store-provider cannot be used together"
        raise typer.BadParameter(msg)
    if settings.a2a_task_store_provider and settings.a2a_task_store not in {"auto", "memory"}:
        msg = "--task-store and --task-store-provider cannot be used together"
        raise typer.BadParameter(msg)

    if additional_python_path:
        settings.additional_python_path = additional_python_path
        sys.path.append(additional_python_path)
        log.trace("Added '{}' to sys.path", additional_python_path)

    # Deploy the pipelines
    deploy_pipelines()

    # Setup the Starlette app exposing pipelines as A2A agents
    log.debug(
        "Starting A2A server with host={}, port={}, pipelines_dir={}, external_url={}, "
        "v0.3_compat={}, task_store={}, task_store_provider={}, durable_store={}, durable_execution_concurrency={}",
        host,
        port,
        pipelines_dir,
        settings.a2a_external_url or "<derived>",
        settings.a2a_v0_3_compat,
        settings.a2a_task_store,
        settings.a2a_task_store_provider or f"<built-in:{settings.a2a_task_store}>",
        settings.durable_store,
        settings.durable_execution_concurrency,
    )
    app = create_a2a_app(debug=debug)

    # Run the A2A server
    # NOTE: reload and workers options are not supported in this context
    intercept_stdlib_logging(
        settings.intercepted_loggers,
        access_log_excluded_path_prefixes=settings.access_log_excluded_path_prefixes,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        timeout_graceful_shutdown=settings.graceful_shutdown_timeout,
    )
