"""Portable FastAPI routes for one durable deployment."""
# ruff: noqa: B008

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import wraps
from typing import Annotated, Any, cast

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Request, Response, status
from fastapi.responses import StreamingResponse
from loguru import logger as log

from hayhooks.durable.engine import (
    RUN_ID_PATTERN,
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
_STREAM_POLL_SECONDS = 0.1
_STREAM_IDLE_POLL_SECONDS = 1.0
_STREAM_IDLE_AFTER = 10
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


def _translate_errors(handler: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
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
        except ExecutionStoreError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Durable execution store is unavailable",
            ) from error
        except RuntimeError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    return translated


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
            max_payload_bytes=deployment.store.config.max_payload_bytes,
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


async def _stream_events(  # noqa: PLR0913
    request: Request,
    deployment: DurableDeployment,
    route_names: dict[str, str],
    response_model: type[ExecutionResult],
    stored: StoredExecution,
    owner_id: str | None,
    enforce_owner: bool,
    cursor: str,
) -> AsyncIterator[str]:
    execution_id = stored.control.run_id
    visible_attempt = stored.control.run_attempt
    page_size = chunk_read_count(deployment.store.config)
    quiet = 0
    try:
        yield _SSE_HEARTBEAT
        while True:
            try:
                chunks = await deployment.store.read_chunks(execution_id, cursor)
            except ChunkCursorExpiredError:
                yield _sse("gap", '{"detail":"Requested stream history is no longer available"}')
                cursor = CHUNK_CURSOR_START
                continue

            if chunks:
                stored = await deployment.get(
                    execution_id,
                    owner_id=owner_id,
                    enforce_owner=enforce_owner,
                    allow_revision_mismatch=True,
                )
                visible_attempt = max(visible_attempt, stored.control.run_attempt)
                for chunk in chunks:
                    cursor = chunk.cursor
                    if chunk.attempt < visible_attempt:
                        continue
                    visible_attempt = chunk.attempt
                    yield _sse(
                        "chunk",
                        json.dumps(
                            {
                                "attempt": chunk.attempt,
                                "payload": decode_json(
                                    chunk.data,
                                    max_bytes=deployment.store.config.max_stream_chunk_bytes,
                                ),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        cursor=chunk.cursor,
                    )
                quiet = 0
                if len(chunks) == page_size:
                    continue
            else:
                quiet += 1
                stored = await deployment.get(
                    execution_id,
                    owner_id=owner_id,
                    enforce_owner=enforce_owner,
                    allow_revision_mismatch=True,
                )
                visible_attempt = max(visible_attempt, stored.control.run_attempt)

            if stored.control.terminal:
                public = _project(request, deployment, route_names, stored, response_model)
                yield _sse(stored.control.status.value, public.model_dump_json())
                return
            yield _SSE_HEARTBEAT
            await asyncio.sleep(_STREAM_IDLE_POLL_SECONDS if quiet > _STREAM_IDLE_AFTER else _STREAM_POLL_SECONDS)
    except Exception as error:
        log.bind(run_id=execution_id, exception_type=type(error).__name__).warning("Durable execution stream failed")
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
        stored = await deployment.get(
            submission.control.run_id,
            owner_id=owner,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
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
        stored = await deployment.get(
            execution_id,
            owner_id=owner,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
        )
        return StreamingResponse(
            _stream_events(
                request,
                deployment,
                route_names,
                response_model,
                stored,
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
            _translate_errors(endpoint),
            methods=methods,
            name=name,
            response_model=model,
            status_code=status_code,
            tags=["durable executions"],
        )
    return router


__all__ = ["OwnerIdDependency", "create_durable_router"]
