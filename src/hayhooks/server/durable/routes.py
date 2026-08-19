"""Hayhooks-specific composition for the public durable FastAPI adapter."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.routing import APIRoute

from hayhooks.durable.fastapi import OwnerIdDependency, create_durable_router
from hayhooks.durable.runtime import DurableDeployment
from hayhooks.settings import settings

DURABLE_ROUTE_SUFFIXES = (
    "/run-durable",
    "/executions/{execution_id}",
    "/executions/{execution_id}/cancel",
    "/executions/{execution_id}/resume",
    "/executions/{execution_id}/stream",
)
_MAX_OWNER_LENGTH = 512


def _trusted_owner_dependency() -> OwnerIdDependency | None:
    header = settings.durable_trusted_owner_header.strip()
    if not header:
        return None

    def trusted_owner(request: Request) -> str:
        owner = request.headers.get(header)
        if not owner:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authenticated owner header '{header}' is required",
            )
        if len(owner) > _MAX_OWNER_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Authenticated owner header '{header}' exceeds 512 characters",
            )
        return owner

    return trusted_owner


def remove_durable_api_routes(app: FastAPI, pipeline_name: str) -> None:
    root = f"/{pipeline_name}"
    durable_paths = {f"{root}{suffix}" for suffix in DURABLE_ROUTE_SUFFIXES}
    app.routes[:] = [
        route
        for route in app.routes
        if not (
            (isinstance(route, APIRoute) and route.path in durable_paths)
            or getattr(getattr(route, "original_router", None), "_hayhooks_durable_pipeline", None) == pipeline_name
        )
    ]


def add_durable_api_routes(
    app: FastAPI,
    pipeline_name: str,
    deployment: DurableDeployment | None,
    *,
    _defer_openapi_rebuild: bool,
) -> None:
    """Replace one pipeline's durable route family with the public adapter."""
    remove_durable_api_routes(app, pipeline_name)
    if deployment is not None:
        router = create_durable_router(
            deployment,
            owner_id_dependency=_trusted_owner_dependency(),
        )
        router._hayhooks_durable_pipeline = pipeline_name  # ty: ignore[unresolved-attribute]
        app.include_router(
            router,
            prefix=f"/{pipeline_name}",
        )
    if not _defer_openapi_rebuild:
        app.openapi_schema = None
        app.setup()


__all__ = ["DURABLE_ROUTE_SUFFIXES", "add_durable_api_routes", "remove_durable_api_routes"]
