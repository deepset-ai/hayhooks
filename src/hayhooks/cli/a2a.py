import sys
from typing import Annotated

import typer

a2a = typer.Typer(rich_markup_mode="rich")


@a2a.command()
def run(  # noqa: PLR0913
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

    if additional_python_path:
        settings.additional_python_path = additional_python_path
        sys.path.append(additional_python_path)
        log.trace("Added '{}' to sys.path", additional_python_path)

    # Deploy the pipelines
    deploy_pipelines()

    # Setup the Starlette app exposing pipelines as A2A agents
    log.debug(
        "Starting A2A server with host={}, port={}, pipelines_dir={}, external_url={}, v0.3_compat={}",
        host,
        port,
        pipelines_dir,
        settings.a2a_external_url or "<derived>",
        settings.a2a_v0_3_compat,
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
