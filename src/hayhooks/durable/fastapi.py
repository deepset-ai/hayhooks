"""FastAPI routes for one application-owned durable deployment."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import wraps
from typing import Annotated, Any, cast

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError, create_model

from hayhooks.durable.backend import CHUNK_CURSOR_START, ChunkCursorExpiredError, chunk_read_count, parse_chunk_cursor
from hayhooks.durable.engine import RUN_ID_PATTERN
from hayhooks.durable.models import ExecutionAdmissionError, ExecutionResult, ExecutionStoreError
from hayhooks.durable.runtime import DefinitionRevisionConflictError, DurableDeployment, IdempotencyConflictError
from hayhooks.server.logger import log

_MAX_OWNER_LENGTH = 512
# The chunk read deliberately does not block server-side: a blocking read pins one
# Redis connection per attached viewer, which caps concurrent streams at the pool
# size and starves the engine sharing that pool. Poll instead -- fast while chunks
# are moving, so tokens still arrive smoothly rather than in visible bursts...
_STREAM_POLL_SECONDS = 0.05
# ...and backed off once the stream is quiet, because an execution can sit in
# `waiting` for hours with a viewer attached.
_STREAM_IDLE_POLL_SECONDS = 0.5
# Empty polls before a stream counts as idle, at the fast interval above.
_STREAM_IDLE_AFTER = 20
# Rereading the record is three round trips and a heartbeat is a wasted frame, so an
# idle stream does both every few idle polls. This is the worst-case delay on the
# terminal event, on top of the time it takes to go idle.
_STREAM_RECHECK_EVERY = 2
# An SSE comment line: keeps the connection warm while an execution is quiet.
_SSE_HEARTBEAT = ":\n\n"
# Proxies buffer ``text/event-stream`` by default, which stalls the whole stream.
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_MAX_OWNER_SCOPED_IDEMPOTENCY_KEY_LENGTH = 63
ExecutionId = Annotated[str, Path(pattern=rf"^{RUN_ID_PATTERN}$", min_length=1, max_length=128)]
OwnerIdDependency = Callable[..., str | Awaitable[str]]


def _unscoped_owner() -> None:
    return None


def _validated_owner(owner_id: Any, *, enforce_owner: bool) -> str | None:
    if not enforce_owner:
        return None
    if not isinstance(owner_id, str) or not owner_id or len(owner_id) > _MAX_OWNER_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The configured owner dependency must return a non-empty string of at most 512 characters",
        )
    return owner_id


def _translate_errors(handler: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Map durable domain failures onto the HTTP contract shared by every handler."""

    @wraps(handler)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await handler(*args, **kwargs)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found") from error
        except (IdempotencyConflictError, DefinitionRevisionConflictError) as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except (ValidationError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
        except ExecutionAdmissionError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
                headers={"Retry-After": str(error.retry_after_seconds)},
            ) from error
        except ExecutionStoreError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Durable execution store is unavailable",
            ) from error

    return wrapper


def _sse(event: str, data: str, *, entry_id: str | None = None) -> str:
    """Frame one SSE event around data that is already serialized JSON."""
    prefix = f"id: {entry_id}\n" if entry_id else ""
    return f"{prefix}event: {event}\ndata: {data}\n\n"


def _durable_response_model(deployment: DurableDeployment) -> type[ExecutionResult]:
    if deployment.result_type is Any:
        return ExecutionResult
    return create_model(
        f"{deployment.name.title().replace('-', '').replace('_', '')}ExecutionResult",
        __base__=ExecutionResult,
        result=(deployment.result_type | None, None),
    )


def create_durable_router(  # noqa: C901, PLR0915 - one factory owns every generated route for a deployment
    deployment: DurableDeployment,
    *,
    owner_id_dependency: OwnerIdDependency | None,
) -> APIRouter:
    """
    Return the typed HTTP API for one durable deployment.

    Passing ``None`` explicitly enables unscoped bearer-by-execution-ID access.
    The caller owns the deployment and its runtime lifecycle.
    """
    router = APIRouter()
    owner_dependency = owner_id_dependency or _unscoped_owner
    enforce_owner = owner_id_dependency is not None
    response_model = _durable_response_model(deployment)
    route_names = {
        "submit": f"hayhooks.durable.{deployment.name}.submit",
        "inspect": f"hayhooks.durable.{deployment.name}.inspect",
        "cancel": f"hayhooks.durable.{deployment.name}.cancel",
        "resume": f"hayhooks.durable.{deployment.name}.resume",
        "stream": f"hayhooks.durable.{deployment.name}.stream",
    }

    def execution_links(request: Request, execution_id: str) -> dict[str, str]:
        root = request.url_for(route_names["inspect"], execution_id=execution_id).path
        return {
            "self": root,
            "cancel": request.url_for(route_names["cancel"], execution_id=execution_id).path,
            "resume": request.url_for(route_names["resume"], execution_id=execution_id).path,
            "stream": request.url_for(route_names["stream"], execution_id=execution_id).path,
        }

    def execution_result(
        request: Request,
        record: Any,
        *,
        model: type[ExecutionResult] = ExecutionResult,
    ) -> ExecutionResult:
        return model.model_validate(record.safe_view(links=execution_links(request, record.execution_id)))

    async def get_execution(execution_id: str, owner_id: str | None) -> Any:
        return await deployment.get(
            execution_id,
            owner_id=owner_id,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
        )

    async def submit(
        run_req: Any,
        response: Response,
        request: Request,
        owner_id: Any = Depends(owner_dependency),  # noqa: B008
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            pattern=rf"^{RUN_ID_PATTERN}$",
        ),
    ) -> ExecutionResult:
        owner_id = _validated_owner(owner_id, enforce_owner=enforce_owner)
        if (
            enforce_owner
            and idempotency_key is not None
            and len(idempotency_key) > _MAX_OWNER_SCOPED_IDEMPOTENCY_KEY_LENGTH
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Idempotency-Key must be at most 63 characters when owner scoping is enabled",
            )
        try:
            created, record = await deployment.submit(
                run_req.model_dump(mode="json"),
                execution_id=idempotency_key,
                owner_id=owner_id,
            )
        except (
            IdempotencyConflictError,
            DefinitionRevisionConflictError,
            ExecutionAdmissionError,
            ExecutionStoreError,
        ):
            raise  # the shared translator owns these status codes
        except RuntimeError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        response.status_code = status.HTTP_200_OK if not created and record.terminal else status.HTTP_202_ACCEPTED
        result = execution_result(request, record, model=response_model)
        response.headers["Location"] = result.links["self"]
        if not created:
            response.headers["Idempotent-Replay"] = "true"
        return result

    submit.__annotations__["run_req"] = deployment.request_type

    async def inspect_execution(
        execution_id: ExecutionId,
        request: Request,
        owner_id: Any = Depends(owner_dependency),  # noqa: B008
    ) -> ExecutionResult:
        owner_id = _validated_owner(owner_id, enforce_owner=enforce_owner)
        return execution_result(request, await get_execution(execution_id, owner_id))

    async def cancel_execution(
        execution_id: ExecutionId,
        response: Response,
        request: Request,
        owner_id: Any = Depends(owner_dependency),  # noqa: B008
    ) -> ExecutionResult:
        owner_id = _validated_owner(owner_id, enforce_owner=enforce_owner)
        accepted = await deployment.request_cancel(execution_id, owner_id=owner_id, enforce_owner=enforce_owner)
        record = await get_execution(execution_id, owner_id)
        response.status_code = status.HTTP_202_ACCEPTED if accepted else status.HTTP_200_OK
        return execution_result(request, record)

    async def resume_execution(
        execution_id: ExecutionId,
        response: Response,
        request: Request,
        owner_id: Any = Depends(owner_dependency),  # noqa: B008
        update: Any = Body(default=None),  # noqa: B008
    ) -> ExecutionResult:
        owner_id = _validated_owner(owner_id, enforce_owner=enforce_owner)
        if not await deployment.resume(execution_id, update, owner_id=owner_id, enforce_owner=enforce_owner):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Execution is not waiting")
        response.status_code = status.HTTP_202_ACCEPTED
        return execution_result(request, await get_execution(execution_id, owner_id))

    async def stream_execution(
        execution_id: ExecutionId,
        request: Request,
        owner_id: Any = Depends(owner_dependency),  # noqa: B008
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        owner_id = _validated_owner(owner_id, enforce_owner=enforce_owner)
        cursor = last_event_id or CHUNK_CURSOR_START
        # Validate the client cursor and resolve ownership before the response
        # starts, while HTTP status codes are still available.
        parse_chunk_cursor(cursor)
        record = await get_execution(execution_id, owner_id)
        return StreamingResponse(
            _chunk_events(request, record, execution_id, owner_id, cursor),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    async def _chunk_events(  # noqa: C901 - one loop owns cursor, attempt, and lifecycle ordering
        request: Request, record: Any, execution_id: str, owner_id: str | None, cursor: str
    ) -> AsyncIterator[str]:
        page = chunk_read_count(deployment.store.config)
        visible_attempt = record.attempt
        quiet = 0
        try:
            # Prove the stream is live before the first poll interval elapses.
            yield _SSE_HEARTBEAT
            while True:
                terminal = record.terminal
                # Draining after the terminal read, never before, is what keeps the
                # last chunks of an execution from being lost.
                try:
                    entries = await deployment.store.read_chunks(execution_id, cursor)
                except ChunkCursorExpiredError:
                    yield _sse("gap", '{"detail":"Requested stream history is no longer available"}')
                    cursor = CHUNK_CURSOR_START
                    continue
                if entries and not terminal:
                    # A lease-lost worker can append after its replacement was
                    # claimed but before that replacement emits its first chunk.
                    # Highest-attempt-seen cannot identify that ordering, so anchor
                    # every non-empty page to the current control attempt first.
                    current_attempt = await deployment.store.current_attempt(execution_id)
                    if current_attempt is not None:
                        visible_attempt = max(visible_attempt, current_attempt)
                for entry_id, attempt, data in entries:
                    cursor = entry_id
                    if attempt < visible_attempt:
                        continue
                    visible_attempt = attempt
                    yield _sse("chunk", f'{{"attempt":{attempt},"payload":{data.decode()}}}', entry_id=entry_id)
                # One read is capped at a page, so a full page means the terminal
                # drain is not finished; stopping here would silently truncate a
                # client reattaching to an already-finished execution.
                if terminal and len(entries) < page:
                    yield _sse(record.status.value, execution_result(request, record).model_dump_json())
                    return
                if entries:
                    quiet = 0
                    continue
                quiet += 1
                idle = quiet > _STREAM_IDLE_AFTER
                # Flowing chunks already mean running, so the lifecycle is only
                # rechecked once the stream goes idle -- and then on a cadence,
                # because an execution can sit in `waiting` for a very long time.
                if idle and quiet % _STREAM_RECHECK_EVERY == 0:
                    yield _SSE_HEARTBEAT
                    record = await get_execution(execution_id, owner_id)
                    visible_attempt = max(visible_attempt, record.attempt)
                await asyncio.sleep(_STREAM_IDLE_POLL_SECONDS if idle else _STREAM_POLL_SECONDS)
        except Exception:
            # The response has already begun, so no status code is left to raise
            # with. Name the break and let the client reattach with Last-Event-ID.
            log.bind(execution_id=execution_id).opt(exception=True).warning("Durable execution stream failed")
            yield _sse("error", '{"detail":"Execution stream interrupted"}')

    if deployment.resume_type is not None:
        resume_execution.__annotations__["update"] = deployment.resume_type
        signature = inspect.signature(resume_execution)
        update_parameter = signature.parameters["update"].replace(annotation=deployment.resume_type, default=Body())
        cast(Any, resume_execution).__signature__ = signature.replace(
            parameters=[
                update_parameter if parameter.name == "update" else parameter
                for parameter in signature.parameters.values()
            ]
        )

    for path, endpoint, methods, name, model in (
        ("/run-durable", submit, ["POST"], route_names["submit"], response_model),
        ("/executions/{execution_id}", inspect_execution, ["GET"], route_names["inspect"], ExecutionResult),
        ("/executions/{execution_id}/cancel", cancel_execution, ["POST"], route_names["cancel"], ExecutionResult),
        ("/executions/{execution_id}/resume", resume_execution, ["POST"], route_names["resume"], ExecutionResult),
        ("/executions/{execution_id}/stream", stream_execution, ["GET"], route_names["stream"], None),
    ):
        router.add_api_route(
            path,
            _translate_errors(endpoint),
            methods=methods,
            name=name,
            response_model=model,
            tags=["durable executions"],
            status_code=status.HTTP_202_ACCEPTED if methods == ["POST"] else status.HTTP_200_OK,
        )
    return router


__all__ = ["OwnerIdDependency", "create_durable_router"]
