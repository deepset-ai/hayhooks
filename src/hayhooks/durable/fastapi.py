"""Portable FastAPI routes for one durable deployment."""
# ruff: noqa: B008

from __future__ import annotations

import inspect
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import wraps
from typing import Annotated, Any, cast

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Request, Response, status
from fastapi.responses import StreamingResponse
from loguru import logger as log

from hayhooks.durable.engine import (
    RUN_ID_PATTERN,
    ExecutionControl,
    ExecutionNotFoundError,
    ExecutionPayloadSizeError,
    ExecutionStatus,
    InvalidExecutionTransitionError,
)
from hayhooks.durable.models import ExecutionResult, decode_json, project_execution
from hayhooks.durable.runtime import DurableDeployment
from hayhooks.durable.store import (
    CHUNK_CURSOR_START,
    ChunkCursorExpiredError,
    ExecutionAdmissionError,
    ExecutionIdempotencyConflictError,
    ExecutionStoreCorruptionError,
    ExecutionStoreError,
    StoredExecution,
    chunk_read_count,
    parse_chunk_cursor,
)

_MAX_HEADER_BYTES = 512
_STREAM_BLOCK_SECONDS = 15.0
_SSE_HEARTBEAT = ": heartbeat\n\n"
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
ExecutionId = Annotated[str, Path(pattern=rf"^{RUN_ID_PATTERN}$")]
OwnerIdDependency = Callable[..., str | Awaitable[str]]


def _unscoped_owner() -> None:
    return None


def _validated_owner(owner_id: object, *, enforce_owner: bool) -> str | None:
    if not enforce_owner:
        return None
    try:
        valid = isinstance(owner_id, str) and bool(owner_id) and len(owner_id.encode()) <= _MAX_HEADER_BYTES
    except UnicodeError:
        valid = False
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The owner dependency must return a non-empty UTF-8 string of at most 512 bytes",
        )
    return cast(str, owner_id)


def _translate_errors(handler: Callable[..., Awaitable[Any]], deployment: str) -> Callable[..., Awaitable[Any]]:
    @wraps(handler)
    async def translated(*args: Any, **kwargs: Any) -> Any:
        try:
            return await handler(*args, **kwargs)
        except ExecutionNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found") from error
        except (ExecutionIdempotencyConflictError, InvalidExecutionTransitionError) as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except ExecutionPayloadSizeError as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
        except ExecutionAdmissionError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
                headers={"Retry-After": "1"},
            ) from error
        except ExecutionStoreCorruptionError as error:
            _log_failure("Durable execution state is invalid", error, deployment, kwargs.get("execution_id"))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Durable execution state is invalid",
            ) from error
        except ExecutionStoreError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Durable execution store is unavailable",
            ) from error
        except RuntimeError as error:
            _log_failure("Durable execution request failed", error, deployment, kwargs.get("execution_id"))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Durable execution service is unavailable",
            ) from error

    return translated


def _log_failure(message: str, error: BaseException, deployment: str, run_id: object) -> None:
    log.opt(exception=error).bind(
        deployment=deployment,
        run_id=run_id,
        exception_type=type(error).__name__,
        error=str(error),
    ).error(message)


def _project(
    request: Request,
    deployment: DurableDeployment,
    route_names: dict[str, str],
    stored: StoredExecution,
    response_model: type[ExecutionResult],
) -> ExecutionResult:
    execution_id = stored.control.run_id
    links = {
        key: request.url_for(route_names[key], execution_id=execution_id).path
        for key in ("self", "cancel", "resume", "stream")
    }
    try:
        public = project_execution(
            stored,
            links=links,
            # Reads never re-apply write limits, which may have been lowered since the write.
            max_payload_bytes=sys.maxsize,
        )
        if (
            deployment.result_model is not None
            and stored.control.definition_revision == deployment.revision
            and stored.control.status is ExecutionStatus.COMPLETED
        ):
            deployment.result_model.model_validate(public.result)
        return response_model.model_validate(public.model_dump(mode="python"))
    except (ExecutionPayloadSizeError, OSError, OverflowError, TypeError, ValueError) as error:
        raise ExecutionStoreCorruptionError("stored execution cannot be projected") from error  # noqa: EM101


def _sse(event: str, data: str, *, cursor: str | None = None) -> str:
    prefix = f"id: {cursor}\n" if cursor is not None else ""
    return f"{prefix}event: {event}\ndata: {data}\n\n"


async def _stream_events(  # noqa: C901, PLR0913
    request: Request,
    deployment: DurableDeployment,
    route_names: dict[str, str],
    response_model: type[ExecutionResult],
    control: ExecutionControl,
    owner_id: str | None,
    enforce_owner: bool,
    cursor: str,
) -> AsyncIterator[str]:
    """
    Push chunks as workers flush them and end on the terminal marker.

    A resumed cursor, or a run that is already terminal, first catches up with bounded
    pages; a terminal run with no marker left ends once caught up. Otherwise every
    iteration blocks on the stream, and a block timeout sends a keepalive and checks the
    control, so a run that ends without a visible marker still ends its stream.
    """
    execution_id = control.run_id
    visible_attempt = control.run_attempt
    terminal = control.terminal
    store = deployment.store
    page_size = chunk_read_count(store.config)
    chunk_bytes = sys.maxsize

    async def terminal_event() -> str:
        stored = await deployment.get(
            execution_id,
            owner_id=owner_id,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
        )
        public = _project(request, deployment, route_names, stored, response_model)
        return _sse(stored.control.status.value, public.model_dump_json())

    try:
        yield _SSE_HEARTBEAT
        catching_up = terminal or cursor != CHUNK_CURSOR_START
        while True:
            try:
                if catching_up:
                    chunks = await store.read_chunks(execution_id, cursor)
                    catching_up = len(chunks) == page_size
                elif terminal:
                    yield await terminal_event()
                    return
                else:
                    waited = await deployment.wait_chunks(execution_id, cursor, _STREAM_BLOCK_SECONDS)
                    if waited is None:
                        return
                    if not waited:
                        yield _SSE_HEARTBEAT
                        control = await deployment.get_control(
                            execution_id,
                            owner_id=owner_id,
                            enforce_owner=enforce_owner,
                        )
                        terminal = catching_up = control.terminal
                        continue
                    chunks = waited
            except ChunkCursorExpiredError:
                yield _sse("gap", '{"detail":"Requested stream history is no longer available"}')
                cursor, catching_up = CHUNK_CURSOR_START, terminal
                continue

            for chunk in chunks:
                cursor = chunk.cursor
                if chunk.terminal:
                    yield await terminal_event()
                    return
                if chunk.skipped or chunk.attempt < visible_attempt:
                    continue
                try:
                    payload = decode_json(chunk.data, max_bytes=chunk_bytes)
                except (ExecutionPayloadSizeError, ValueError):
                    log.bind(run_id=execution_id, cursor=chunk.cursor).warning(
                        "Skipped an undecodable durable stream chunk"
                    )
                    continue
                visible_attempt = chunk.attempt
                yield _sse(
                    "chunk",
                    json.dumps(
                        {"attempt": chunk.attempt, "payload": payload},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    cursor=chunk.cursor,
                )
    except Exception as error:
        log.bind(run_id=execution_id, exception_type=type(error).__name__, error=str(error)).warning(
            "Durable execution stream failed"
        )
        yield _sse("error", '{"detail":"Execution stream interrupted"}')


def create_durable_router(  # noqa: C901
    deployment: DurableDeployment,
    *,
    owner_id_dependency: OwnerIdDependency | None,
) -> APIRouter:
    """Expose one caller-owned deployment without managing its lifecycle."""
    router = APIRouter()
    owner_dependency = owner_id_dependency or _unscoped_owner
    enforce_owner = owner_id_dependency is not None
    response_model = ExecutionResult
    route_names = {
        key: f"hayhooks.durable.{deployment.name}.{key}" for key in ("submit", "self", "cancel", "resume", "stream")
    }

    async def submit_execution(
        payload: Any,
        response: Response,
        request: Request,
        owner_id: object = Depends(owner_dependency),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", min_length=1),
    ) -> ExecutionResult:
        owner = _validated_owner(owner_id, enforce_owner=enforce_owner)
        if idempotency_key is not None:
            try:
                valid_key = len(idempotency_key.encode()) <= _MAX_HEADER_BYTES
            except UnicodeError:
                valid_key = False
            if not valid_key:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Idempotency-Key must be at most 512 UTF-8 bytes",
                )
        submission = await deployment.submit(payload, owner_id=owner, idempotency_key=idempotency_key)
        # A new execution is queued with no payloads or progress, so the submitted control is its snapshot.
        stored = (
            StoredExecution(submission.control, {}, ())
            if submission.created
            else await deployment.get(
                submission.control.run_id,
                owner_id=owner,
                enforce_owner=enforce_owner,
                allow_revision_mismatch=True,
            )
        )
        public = _project(request, deployment, route_names, stored, response_model)
        response.status_code = (
            status.HTTP_200_OK if not submission.created and stored.control.terminal else status.HTTP_202_ACCEPTED
        )
        response.headers["Location"] = public.links["self"]
        if not submission.created:
            response.headers["Idempotent-Replay"] = "true"
        return public

    submit_execution.__annotations__["payload"] = deployment.request_model

    async def inspect_execution(
        execution_id: ExecutionId,
        request: Request,
        owner_id: object = Depends(owner_dependency),
    ) -> ExecutionResult:
        owner = _validated_owner(owner_id, enforce_owner=enforce_owner)
        stored = await deployment.get(
            execution_id,
            owner_id=owner,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
        )
        return _project(request, deployment, route_names, stored, response_model)

    async def cancel_execution(
        execution_id: ExecutionId,
        response: Response,
        request: Request,
        owner_id: object = Depends(owner_dependency),
    ) -> ExecutionResult:
        owner = _validated_owner(owner_id, enforce_owner=enforce_owner)
        plan = await deployment.cancel(execution_id, owner_id=owner, enforce_owner=enforce_owner)
        response.status_code = status.HTTP_200_OK if plan.next_control.terminal else status.HTTP_202_ACCEPTED
        stored = await deployment.get(
            execution_id,
            owner_id=owner,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
        )
        return _project(request, deployment, route_names, stored, response_model)

    async def resume_execution(
        execution_id: ExecutionId,
        response: Response,
        request: Request,
        owner_id: object = Depends(owner_dependency),
        resume_input: Any = Body(default=None),
    ) -> ExecutionResult:
        owner = _validated_owner(owner_id, enforce_owner=enforce_owner)
        await deployment.resume(execution_id, resume_input, owner_id=owner, enforce_owner=enforce_owner)
        response.status_code = status.HTTP_202_ACCEPTED
        stored = await deployment.get(
            execution_id,
            owner_id=owner,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
        )
        return _project(request, deployment, route_names, stored, response_model)

    if deployment.resume_model is not None:
        resume_execution.__annotations__["resume_input"] = deployment.resume_model
        signature = inspect.signature(resume_execution)
        resume_parameter = signature.parameters["resume_input"].replace(
            annotation=deployment.resume_model,
            default=Body(),
        )
        cast(Any, resume_execution).__signature__ = signature.replace(
            parameters=[
                resume_parameter if parameter.name == "resume_input" else parameter
                for parameter in signature.parameters.values()
            ]
        )

    async def stream_execution(
        execution_id: ExecutionId,
        request: Request,
        owner_id: object = Depends(owner_dependency),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        owner = _validated_owner(owner_id, enforce_owner=enforce_owner)
        cursor = CHUNK_CURSOR_START if last_event_id is None else last_event_id
        parse_chunk_cursor(cursor)
        control = await deployment.get_control(execution_id, owner_id=owner, enforce_owner=enforce_owner)
        return StreamingResponse(
            _stream_events(
                request,
                deployment,
                route_names,
                response_model,
                control,
                owner,
                enforce_owner,
                cursor,
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    for path, endpoint, methods, name, model, status_code in (
        ("/run-durable", submit_execution, ["POST"], route_names["submit"], response_model, status.HTTP_202_ACCEPTED),
        (
            "/executions/{execution_id}",
            inspect_execution,
            ["GET"],
            route_names["self"],
            response_model,
            status.HTTP_200_OK,
        ),
        (
            "/executions/{execution_id}/cancel",
            cancel_execution,
            ["POST"],
            route_names["cancel"],
            response_model,
            status.HTTP_202_ACCEPTED,
        ),
        (
            "/executions/{execution_id}/resume",
            resume_execution,
            ["POST"],
            route_names["resume"],
            response_model,
            status.HTTP_202_ACCEPTED,
        ),
        (
            "/executions/{execution_id}/stream",
            stream_execution,
            ["GET"],
            route_names["stream"],
            None,
            status.HTTP_200_OK,
        ),
    ):
        router.add_api_route(
            path,
            _translate_errors(endpoint, deployment.name),
            methods=methods,
            name=name,
            response_model=model,
            status_code=status_code,
            tags=["durable executions"],
        )
    return router


__all__ = ["OwnerIdDependency", "create_durable_router"]
