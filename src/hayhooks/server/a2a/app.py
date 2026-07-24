from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from hayhooks.a2a import TaskStoreProvider
from hayhooks.durable.mode import DurableAuthoringMode, durable_authoring_mode
from hayhooks.durable.runtime import durable_runtime
from hayhooks.server.a2a.cards import create_agent_card, get_a2a_base_url, is_a2a_exposable
from hayhooks.server.a2a.executor import create_agent_executor
from hayhooks.server.a2a.imports import (
    DefaultRequestHandler,
    a2a_import,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from hayhooks.server.a2a.runtime import A2ARuntime, create_task_store_provider
from hayhooks.server.logger import log
from hayhooks.server.pipelines.lifecycle import close_pipeline_wrappers, start_pipeline_wrappers
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.tracing import configure_tracing, instrument_starlette_app
from hayhooks.settings import settings

# Path reserved by the A2A app itself; a pipeline with this name cannot be mounted
_RESERVED_PATHS = frozenset({"status"})


def _create_agent_mount(pipeline_name: str, base_url: str, runtime: A2ARuntime) -> Mount:
    wrapper = registry.get(pipeline_name)
    if wrapper is None:
        msg = f"Pipeline '{pipeline_name}' not found"
        raise ValueError(msg)

    card = create_agent_card(
        pipeline_name,
        base_url,
        push_notifications=runtime.push_notifications_enabled,
    )
    task_store = runtime.create_task_store(pipeline_name)
    agent_executor = create_agent_executor(wrapper, pipeline_name, runtime=runtime, task_store=task_store)
    runtime.register_agent_executor(agent_executor)
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
        agent_card=card,
        request_context_builder=runtime.create_request_context_builder(task_store),
    )
    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(request_handler, rpc_url="/", enable_v0_3_compat=settings.a2a_v0_3_compat),
    ]
    log.debug(
        "Created A2A mount for pipeline '{}' at '/{}' with v0.3_compat={}",
        pipeline_name,
        pipeline_name,
        settings.a2a_v0_3_compat,
    )
    return Mount(f"/{pipeline_name}", routes=routes)


def _create_app_task_store_provider(durable_agents_deployed: bool) -> TaskStoreProvider:
    backend = settings.a2a_task_store
    redis_url = settings.a2a_redis_url
    redis_key_prefix = settings.a2a_redis_key_prefix
    durable_redis = durable_agents_deployed and settings.durable_store == "redis"
    if backend == "auto" and durable_redis and not settings.a2a_task_store_provider:
        backend = "redis"
        redis_url = settings.durable_redis_url
        redis_key_prefix = f"{settings.durable_redis_key_prefix.rstrip(':')}:a2a"
    elif backend == "auto":
        backend = "memory"

    shared_redis = (
        durable_runtime.shared_redis_client()
        if backend == "redis" and durable_redis and redis_url == settings.durable_redis_url
        else None
    )
    if shared_redis is None:
        return create_task_store_provider(
            backend=backend,
            custom_provider=settings.a2a_task_store_provider,
            redis_url=redis_url,
            redis_key_prefix=redis_key_prefix,
        )
    return create_task_store_provider(
        backend=backend,
        custom_provider=settings.a2a_task_store_provider,
        redis_url=redis_url,
        redis_key_prefix=redis_key_prefix,
        redis=shared_redis,
        close_redis=False,
    )


def _create_agent_mounts(base_url: str, runtime: A2ARuntime) -> tuple[list[str], list[Mount]]:
    agent_names: list[str] = []
    mounts: list[Mount] = []
    for pipeline_name in registry.get_names():
        if not is_a2a_exposable(pipeline_name):
            continue
        if pipeline_name in _RESERVED_PATHS:
            log.warning("Skipping pipeline '{}': the path is reserved by the A2A server", pipeline_name)
            continue
        try:
            mounts.append(_create_agent_mount(pipeline_name, base_url, runtime))
        except Exception as error:
            log.opt(exception=True).warning(
                "Skipping pipeline '{}': failed to build A2A agent: {}",
                pipeline_name,
                error,
            )
            continue
        agent_names.append(pipeline_name)
        log.info("Exposing pipeline '{}' as A2A agent at {}/{}/", pipeline_name, base_url, pipeline_name)
    return agent_names, mounts


def create_a2a_app(*, base_url: str | None = None, debug: bool = False, runtime: A2ARuntime | None = None) -> Starlette:
    """
    Create a Starlette app exposing deployed pipelines as A2A agents.

    Each exposable pipeline is mounted under ``/{pipeline_name}/`` with its
    agent card at ``/{pipeline_name}/.well-known/agent-card.json`` and the
    JSON-RPC binding at ``POST /{pipeline_name}/``.
    """
    a2a_import.check()

    durable_agents_deployed = any(
        wrapper is not None and durable_authoring_mode(wrapper) is DurableAuthoringMode.MANAGED_AGENT
        for name in registry.get_names()
        if (wrapper := registry.get(name)) is not None
    )
    runtime = runtime or A2ARuntime(
        task_store_provider=_create_app_task_store_provider(durable_agents_deployed),
    )
    log.info("Using A2A task store provider '{}'", type(runtime.task_store_provider).__name__)
    base_url = (base_url or get_a2a_base_url()).rstrip("/")
    if "//0.0.0.0" in base_url or "//[::]" in base_url:
        log.warning(
            "Agent cards will advertise the wildcard bind address ({}) which remote clients cannot connect to. "
            "Set HAYHOOKS_A2A_EXTERNAL_URL (or --external-url) to the server's reachable base URL.",
            base_url,
        )

    agent_names, mounts = _create_agent_mounts(base_url, runtime)

    if not agent_names:
        log.warning(
            "No pipelines exposable as A2A agents. "
            "A pipeline must implement create_a2a_agent_executor, run_chat_completion, "
            "or run_chat_completion_async."
        )

    async def handle_status(request: Request) -> JSONResponse:  # noqa: ARG001
        return JSONResponse({"status": "ok", "agents": agent_names})

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:  # noqa: ARG001
        await start_pipeline_wrappers()
        try:
            await runtime.start()
            yield
        finally:
            try:
                await runtime.close()
            finally:
                await close_pipeline_wrappers()

    app = Starlette(debug=debug, routes=[Route("/status", endpoint=handle_status), *mounts], lifespan=lifespan)
    log.debug("Created A2A Starlette app with {} mounted agent(s): {}", len(agent_names), agent_names)

    configure_tracing()
    instrument_starlette_app(app)
    return app
