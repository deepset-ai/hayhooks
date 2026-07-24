"""Typed REST resources that project the durable execution record."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from fastapi import Body, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response
from fastapi.routing import APIRoute
from pydantic import ValidationError, create_model

from hayhooks.durable import ExecutionResult
from hayhooks.durable.models import ExecutionStoreError
from hayhooks.durable.runtime import (
    DefinitionRevisionConflictError,
    DurableDeployment,
    IdempotencyConflictError,
    durable_runtime,
)
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings

DURABLE_ROUTE_SUFFIXES = (
    "/run-durable",
    "/executions/{execution_id}",
    "/executions/{execution_id}/cancel",
    "/executions/{execution_id}/resume",
)
_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._~-]{1,256}$")
_MAX_DURABLE_OWNER_LENGTH = 512


def _execution_links(pipeline_name: str, execution_id: str) -> dict[str, str]:
    root = f"/{pipeline_name}/executions/{quote(execution_id, safe='-._~')}"
    return {"self": root, "cancel": f"{root}/cancel", "resume": f"{root}/resume"}


def _execution_result(
    deployment: DurableDeployment,
    record: Any,
    *,
    response_model: type[ExecutionResult] = ExecutionResult,
) -> ExecutionResult:
    return response_model.model_validate(record.safe_view(links=_execution_links(deployment.name, record.execution_id)))


def _durable_response_model(deployment: DurableDeployment) -> type[ExecutionResult]:
    if deployment.result_type is Any:
        return ExecutionResult
    return create_model(
        f"{deployment.name.title().replace('-', '').replace('_', '')}ExecutionResult",
        __base__=ExecutionResult,
        result=(deployment.result_type | None, None),
    )


def _durable_owner(request: Request) -> tuple[str | None, bool]:
    header = settings.durable_trusted_owner_header.strip()
    if not header:
        return None, False
    owner = request.headers.get(header)
    if not owner:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authenticated owner header '{header}' is required",
        )
    if len(owner) > _MAX_DURABLE_OWNER_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authenticated owner header '{header}' exceeds 512 characters",
        )
    return owner, True


def _remove_pipeline_route(app: FastAPI, path: str, method: str) -> None:
    for route in list(app.routes):
        if isinstance(route, APIRoute) and route.path == path and route.methods is not None and method in route.methods:
            app.routes.remove(route)


def _remove_durable_api_routes(app: FastAPI, pipeline_name: str) -> None:
    root = f"/{pipeline_name}"
    durable_paths = {f"{root}{suffix}" for suffix in DURABLE_ROUTE_SUFFIXES}
    app.routes[:] = [route for route in app.routes if not (isinstance(route, APIRoute) and route.path in durable_paths)]


def add_durable_api_routes(  # noqa: C901, PLR0915 - route-local handlers share generated models
    app: FastAPI,
    pipeline_name: str,
    pipeline_wrapper: BasePipelineWrapper,
    *,
    deployment: DurableDeployment | None = None,
    _defer_openapi_rebuild: bool,
) -> None:
    """Register typed durable submission and control resources when opted in."""
    _remove_durable_api_routes(app, pipeline_name)
    if not durable_runtime.has_capability(pipeline_wrapper):
        if not _defer_openapi_rebuild:
            app.openapi_schema = None
            app.setup()
        return
    deployment = deployment or durable_runtime.deployment(pipeline_name, pipeline_wrapper)
    request_model = deployment.request_type
    response_model = _durable_response_model(deployment)
    root = f"/{pipeline_name}"

    async def submit(
        run_req: Any,
        response: Response,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> ExecutionResult:
        if idempotency_key is not None and _IDEMPOTENCY_KEY_PATTERN.fullmatch(idempotency_key) is None:
            raise HTTPException(
                status_code=422,
                detail="Idempotency-Key must contain 1-256 URL-safe letters, digits, '.', '_', '~', or '-'",
            )
        try:
            owner_id, _ = _durable_owner(request)
            created, record = await deployment.submit(
                run_req.model_dump(mode="json"),
                execution_id=idempotency_key,
                owner_id=owner_id,
            )
        except IdempotencyConflictError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except DefinitionRevisionConflictError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except (ValidationError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
        except (ExecutionStoreError, RuntimeError) as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        response.status_code = status.HTTP_200_OK if not created and record.terminal else status.HTTP_202_ACCEPTED
        response.headers["Location"] = _execution_links(deployment.name, record.execution_id)["self"]
        if not created:
            response.headers["Idempotent-Replay"] = "true"
        return _execution_result(deployment, record, response_model=response_model)

    # FastAPI consumes the runtime annotation; keep the Python type checker on
    # the stable BaseModel boundary while preserving the generated request schema.
    submit.__annotations__["run_req"] = request_model

    async def inspect_execution(execution_id: str, request: Request) -> ExecutionResult:
        try:
            owner_id, enforce_owner = _durable_owner(request)
            record = await deployment.get(execution_id, owner_id=owner_id, enforce_owner=enforce_owner)
            return _execution_result(deployment, record, response_model=response_model)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Execution not found") from error
        except DefinitionRevisionConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ExecutionStoreError as error:
            raise HTTPException(status_code=503, detail="Durable execution store is unavailable") from error

    async def cancel_execution(execution_id: str, response: Response, request: Request) -> ExecutionResult:
        try:
            owner_id, enforce_owner = _durable_owner(request)
            accepted = await deployment.request_cancel(
                execution_id,
                owner_id=owner_id,
                enforce_owner=enforce_owner,
            )
            record = await deployment.get(execution_id, owner_id=owner_id, enforce_owner=enforce_owner)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Execution not found") from error
        except DefinitionRevisionConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ExecutionStoreError as error:
            raise HTTPException(status_code=503, detail="Durable execution store is unavailable") from error
        response.status_code = status.HTTP_202_ACCEPTED if accepted else status.HTTP_200_OK
        return _execution_result(deployment, record, response_model=response_model)

    async def resume_execution(
        execution_id: str,
        response: Response,
        request: Request,
        update: Any = Body(default=None),  # noqa: B008
    ) -> ExecutionResult:
        try:
            owner_id, enforce_owner = _durable_owner(request)
            resumed = await deployment.resume(
                execution_id,
                update,
                owner_id=owner_id,
                enforce_owner=enforce_owner,
            )
            if not resumed:
                raise HTTPException(status_code=409, detail="Execution is not waiting")
            record = await deployment.get(execution_id, owner_id=owner_id, enforce_owner=enforce_owner)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Execution not found") from error
        except DefinitionRevisionConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ValidationError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ExecutionStoreError as error:
            raise HTTPException(status_code=503, detail="Durable execution store is unavailable") from error
        response.status_code = status.HTTP_202_ACCEPTED
        return _execution_result(deployment, record, response_model=response_model)

    if deployment.resume_type is not None:
        resume_execution.__annotations__["update"] = deployment.resume_type | None

    routes = [
        (f"{root}/run-durable", submit, ["POST"], f"{pipeline_name}_run_durable"),
        (f"{root}/executions/{{execution_id}}", inspect_execution, ["GET"], f"{pipeline_name}_execution"),
        (f"{root}/executions/{{execution_id}}/cancel", cancel_execution, ["POST"], f"{pipeline_name}_cancel"),
        (f"{root}/executions/{{execution_id}}/resume", resume_execution, ["POST"], f"{pipeline_name}_resume"),
    ]
    for path, endpoint, methods, name in routes:
        _remove_pipeline_route(app, path, methods[0])
        app.add_api_route(
            path,
            endpoint,
            methods=methods,
            name=name,
            response_model=response_model,
            tags=["durable executions"],
            status_code=status.HTTP_202_ACCEPTED if methods == ["POST"] else status.HTTP_200_OK,
        )

    registry.update_metadata(
        pipeline_name,
        {"durable_request_model": request_model, "durable_response_model": response_model},
    )
    if not _defer_openapi_rebuild:
        app.openapi_schema = None
        app.setup()


__all__ = ["DURABLE_ROUTE_SUFFIXES", "add_durable_api_routes"]
