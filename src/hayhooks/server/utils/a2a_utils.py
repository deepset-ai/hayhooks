import traceback
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any

from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from haystack.dataclasses import StreamingChunk
from haystack.lazy_imports import LazyImport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from hayhooks.server.logger import log
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.tracing import (
    SPAN_A2A_RUN_AGENT,
    build_trace_tags,
    configure_tracing,
    instrument_starlette_app,
    trace_operation,
)
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings

# Lazily import A2A modules so the optional dependency is only required when used
with LazyImport("Run 'pip install \"hayhooks[a2a]\"' to install A2A support.") as a2a_import:
    from a2a.helpers import get_message_text, new_task_from_user_message, new_text_part
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role

RESPONSE_ARTIFACT_NAME = "response"

# Path reserved by the A2A app itself; a pipeline with this name cannot be mounted
_RESERVED_PATHS = frozenset({"status"})


def get_a2a_base_url() -> str:
    """Base URL advertised in agent cards, without trailing slash."""
    base_url = settings.a2a_external_url or f"http://{settings.a2a_host}:{settings.a2a_port}"
    return base_url.rstrip("/")


def is_a2a_exposable(pipeline_name: str) -> bool:
    """
    Whether a deployed pipeline can be exposed as an A2A agent.

    A pipeline is exposable when it implements ``run_chat_completion`` or
    ``run_chat_completion_async`` and does not set ``skip_a2a = True``.
    """
    pipeline_wrapper = registry.get(pipeline_name)
    if pipeline_wrapper is None:
        return False

    metadata = registry.get_metadata(name=pipeline_name) or {}
    if metadata.get("skip_a2a"):
        log.debug("Skipping pipeline '{}': skip_a2a is set", pipeline_name)
        return False

    exposable = (
        pipeline_wrapper._is_run_chat_completion_implemented
        or pipeline_wrapper._is_run_chat_completion_async_implemented
    )
    if not exposable:
        log.debug("Skipping pipeline '{}': no chat completion method implemented", pipeline_name)
    return exposable


def create_agent_card(pipeline_name: str, base_url: str) -> "AgentCard":
    """
    Build an A2A agent card for a deployed pipeline.

    Card fields are derived from the pipeline's registry metadata and can be
    overridden via the wrapper's ``a2a_card`` class attribute.
    """
    a2a_import.check()

    metadata = registry.get_metadata(name=pipeline_name) or {}
    overrides = metadata.get("a2a_card") or {}

    name = overrides.get("name") or pipeline_name
    description = (
        overrides.get("description")
        or metadata.get("description")
        or f"Haystack pipeline '{pipeline_name}' deployed with Hayhooks"
    )
    version = overrides.get("version") or "1.0.0"
    agent_url = f"{base_url.rstrip('/')}/{pipeline_name}/"

    skills_spec: list[dict[str, Any]] = overrides.get("skills") or [
        {"id": pipeline_name, "name": name, "description": description, "tags": ["haystack", "hayhooks"]}
    ]
    skills = [
        AgentSkill(
            id=skill.get("id", pipeline_name),
            name=skill.get("name", name),
            description=skill.get("description", description),
            tags=list(skill.get("tags", [])),
            examples=list(skill.get("examples", [])),
        )
        for skill in skills_spec
    ]

    log.debug(
        "Built A2A agent card for pipeline '{}' with name='{}', url='{}', skills={}",
        pipeline_name,
        name,
        agent_url,
        [skill.id for skill in skills],
    )

    return AgentCard(
        name=name,
        description=description,
        version=version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=agent_url)],
        skills=skills,
    )


def _stream_item_to_text(item: Any) -> str | None:
    """
    Map a chat-completion stream item to response text.

    Returns None for items that carry no response text (UI events, empty chunks).
    """
    if isinstance(item, StreamingChunk):
        return item.content or None
    if isinstance(item, str):
        return item or None
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace") or None
    # PipelineEvent / dict items are UI-oriented events, not part of the text response
    return None


def _build_openai_messages(context: "RequestContext") -> list[dict]:
    """Map the A2A task history and current message to OpenAI-format messages."""
    messages: list[dict] = []

    history = list(context.current_task.history) if context.current_task else []
    history_message_ids = {message.message_id for message in history}
    for message in history:
        text = get_message_text(message)
        if text:
            role = "assistant" if message.role == Role.ROLE_AGENT else "user"
            messages.append({"role": role, "content": text})

    # Append the incoming message unless it is already part of the task history
    current = context.message
    if current is not None and current.message_id not in history_message_ids:
        current_text = get_message_text(current)
        if current_text:
            messages.append({"role": "user", "content": current_text})

    log.debug(
        "Mapped A2A request context to {} OpenAI message(s): history={}, current_message={}",
        len(messages),
        len(history),
        current is not None,
    )
    return messages


async def _run_chat_completion(pipeline_name: str, context: "RequestContext") -> Any:
    """Run the pipeline's chat completion method (async preferred, sync via threadpool)."""
    pipeline_wrapper: BasePipelineWrapper | None = registry.get(pipeline_name)
    if pipeline_wrapper is None:
        msg = f"Pipeline '{pipeline_name}' not found"
        raise ValueError(msg)

    messages = _build_openai_messages(context)

    if pipeline_wrapper._is_run_chat_completion_async_implemented:
        log.debug("Running pipeline '{}' as A2A agent via async chat completion", pipeline_name)
        return await pipeline_wrapper.run_chat_completion_async(model=pipeline_name, messages=messages, body={})
    log.debug("Running pipeline '{}' as A2A agent via sync chat completion in threadpool", pipeline_name)
    return await run_in_threadpool(
        pipeline_wrapper.run_chat_completion, model=pipeline_name, messages=messages, body={}
    )


async def _iter_text_chunks(result: Any) -> AsyncGenerator[str, None]:
    """
    Normalize a chat completion result (str or sync/async iterator) into text chunks.

    Raises ValueError for results outside the ``run_chat_completion`` contract
    (e.g. None) so wrapper bugs surface as failed tasks instead of a completed
    task whose response text is ``"None"``.
    """
    if isinstance(result, str):
        yield result
    elif isinstance(result, AsyncIterator):
        async for item in result:
            text = _stream_item_to_text(item)
            if text is not None:
                yield text
    elif isinstance(result, Iterator):
        # Drain in a threadpool to keep the event loop free
        async for item in iterate_in_threadpool(result):
            text = _stream_item_to_text(item)
            if text is not None:
                yield text
    else:
        msg = f"run_chat_completion returned unsupported type '{type(result).__name__}'; expected str or generator"
        raise ValueError(msg)


async def _stream_result_as_artifact(result: Any, updater: "TaskUpdater") -> None:
    """
    Emit the chat completion result as a single ``response`` artifact.

    Generator results are streamed incrementally as artifact chunks
    (``append=True``) so SSE clients receive text as it is produced;
    the task manager aggregates chunks for non-streaming clients.
    The last chunk is emitted with ``last_chunk=True``, so chunks are
    held back one iteration until the end of the stream is known.
    """
    artifact_id = str(uuid.uuid4())
    first = True
    pending: str | None = None

    async def emit(text: str, *, last: bool) -> None:
        nonlocal first
        log.debug(
            "Emitting A2A artifact chunk: artifact_id={}, append={}, last={}, chars={}",
            artifact_id,
            not first,
            last,
            len(text),
        )
        await updater.add_artifact(
            [new_text_part(text)],
            artifact_id=artifact_id,
            name=RESPONSE_ARTIFACT_NAME,
            append=not first,
            last_chunk=last,
        )
        first = False

    async for text in _iter_text_chunks(result):
        if pending is not None:
            await emit(pending, last=False)
        pending = text
    await emit(pending if pending is not None else "", last=True)


async def _execute_agent_task(pipeline_name: str, context: "RequestContext", event_queue: "EventQueue") -> None:
    """
    Run a pipeline's chat completion as an A2A task.

    Emits the event sequence required by the A2A spec: the Task first,
    then a working status, artifact chunk(s), and a terminal state.
    """
    if context.current_task is not None:
        task = context.current_task
        log.debug("Continuing A2A task '{}' for pipeline '{}'", task.id, pipeline_name)
    elif context.message is not None:
        task = new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        log.debug("Created A2A task '{}' for pipeline '{}'", task.id, pipeline_name)
    else:
        msg = "A2A request has neither a current task nor a message"
        raise ValueError(msg)

    updater = TaskUpdater(event_queue, task.id, task.context_id)
    await updater.start_work()

    with trace_operation(
        SPAN_A2A_RUN_AGENT,
        tags=build_trace_tags({"hayhooks.transport": "a2a", "hayhooks.pipeline.name": pipeline_name}),
    ):
        try:
            result = await _run_chat_completion(pipeline_name, context)
            await _stream_result_as_artifact(result, updater)
        except Exception as exc:
            msg = f"Error running pipeline '{pipeline_name}' as A2A agent: {exc}"
            if settings.show_tracebacks:
                msg += f"\n{traceback.format_exc()}"
            log.opt(exception=True).error(msg)
            await updater.failed(message=updater.new_agent_message([new_text_part(msg)]))
            return

    await updater.complete()
    log.debug("Completed A2A task '{}' for pipeline '{}'", task.id, pipeline_name)


def create_agent_executor(pipeline_name: str) -> "AgentExecutor":
    """
    Create an ``AgentExecutor`` bridging A2A requests to the given pipeline.

    The class is defined inside this factory (instead of at module level) so
    the module stays importable when the optional ``a2a-sdk`` dependency is
    not installed.
    """
    a2a_import.check()

    class HayhooksAgentExecutor(AgentExecutor):
        """Runs a deployed pipeline's chat completion method as an A2A task."""

        def __init__(self, name: str) -> None:
            self.pipeline_name = name

        async def execute(self, context: "RequestContext", event_queue: "EventQueue") -> None:
            await _execute_agent_task(self.pipeline_name, context, event_queue)

        async def cancel(self, context: "RequestContext", event_queue: "EventQueue") -> None:
            # Best-effort: Hayhooks has no pipeline interruption primitive
            task = context.current_task
            if task is not None:
                await TaskUpdater(event_queue, task.id, task.context_id).cancel()

    return HayhooksAgentExecutor(pipeline_name)


def _create_agent_mount(pipeline_name: str, base_url: str) -> Mount:
    card = create_agent_card(pipeline_name, base_url)
    request_handler = DefaultRequestHandler(
        agent_executor=create_agent_executor(pipeline_name),
        task_store=InMemoryTaskStore(),
        agent_card=card,
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


def create_a2a_app(*, base_url: str | None = None, debug: bool = False) -> Starlette:
    """
    Create a Starlette app exposing deployed pipelines as A2A agents.

    Each exposable pipeline is mounted under ``/{pipeline_name}/`` with its
    agent card at ``/{pipeline_name}/.well-known/agent-card.json`` and the
    JSON-RPC binding at ``POST /{pipeline_name}/``.

    NOTE: mounts are built from the registry at startup; pipelines deployed or
    undeployed at runtime require a restart to be reflected.
    """
    a2a_import.check()

    base_url = (base_url or get_a2a_base_url()).rstrip("/")
    if "//0.0.0.0" in base_url or "//[::]" in base_url:
        log.warning(
            "Agent cards will advertise the wildcard bind address ({}) which remote clients cannot connect to. "
            "Set HAYHOOKS_A2A_EXTERNAL_URL (or --external-url) to the server's reachable base URL.",
            base_url,
        )

    agent_names: list[str] = []
    mounts: list[Mount] = []
    for pipeline_name in registry.get_names():
        if not is_a2a_exposable(pipeline_name):
            continue

        if pipeline_name in _RESERVED_PATHS:
            log.warning("Skipping pipeline '{}': the path is reserved by the A2A server", pipeline_name)
            continue

        # One failing agent card (e.g. a malformed a2a_card override) must not
        # take down the other agents
        try:
            mounts.append(_create_agent_mount(pipeline_name, base_url))
        except Exception as e:
            log.opt(exception=True).warning("Skipping pipeline '{}': failed to build A2A agent: {}", pipeline_name, e)
            continue

        agent_names.append(pipeline_name)
        log.info("Exposing pipeline '{}' as A2A agent at {}/{}/", pipeline_name, base_url, pipeline_name)

    if not agent_names:
        log.warning(
            "No pipelines exposable as A2A agents. "
            "A pipeline must implement run_chat_completion or run_chat_completion_async."
        )

    async def handle_status(request: Request) -> JSONResponse:  # noqa: ARG001
        return JSONResponse({"status": "ok", "agents": agent_names})

    app = Starlette(debug=debug, routes=[Route("/status", endpoint=handle_status), *mounts])
    log.debug("Created A2A Starlette app with {} mounted agent(s): {}", len(agent_names), agent_names)

    configure_tracing()
    instrument_starlette_app(app)
    return app
