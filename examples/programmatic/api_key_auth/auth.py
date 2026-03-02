"""
Auth helpers for API-key-protected Hayhooks apps.

This module is meant as a starting point. Replace or extend it to build
your own key verification system (database lookups, scoped keys, etc.).

The three building blocks are:

1. **Key loading**  - ``_load_api_keys()`` reads keys from env vars.
   Swap this with your own store (database, secrets manager, config file).

2. **Middleware**   - ``_add_api_key_middleware()`` intercepts every HTTP
   request and rejects those without a valid ``X-API-Key`` header.
   Certain paths (Swagger UI, OpenAPI schema) are kept public so docs
   remain accessible.  Edit ``PUBLIC_PATHS`` to change what's exempt.

3. **OpenAPI hook** - ``_add_openapi_security()`` patches the generated
   OpenAPI schema so the Swagger UI at ``/docs`` shows an **Authorize**
   button.  This is optional — remove it if you don't need it.
"""

import os
import secrets
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

# --- Configuration ---------------------------------------------------------
# Change these to match your env-var naming convention.
API_KEYS_ENV_VAR = "HAYHOOKS_API_KEYS"  # comma-separated list of keys
API_KEY_ENV_VAR = "HAYHOOKS_API_KEY"  # single-key fallback
API_KEY_HEADER = "X-API-Key"  # HTTP header clients must send

# Paths that can be reached without an API key.
# Swagger UI (/docs, /redoc) and the OpenAPI schema (/openapi.json) are
# public by default so users can open the docs and use the Authorize button.
# Add or remove paths here as needed.
PUBLIC_PATHS = {"/docs", "/redoc", "/openapi.json"}


# --- Path helpers ----------------------------------------------------------
# When Hayhooks is deployed behind a reverse proxy with a root_path prefix
# (e.g. /api), the incoming request path includes that prefix.  These helpers
# strip it before matching against PUBLIC_PATHS so the allowlist stays simple.


def _normalize(path: str) -> str:
    path = f"/{path.lstrip('/')}" if path else "/"
    return path.rstrip("/") or "/"


def _is_public_path(path: str, root_path: str = "") -> bool:
    candidate = _normalize(path)
    normalized_root = _normalize(root_path)

    if normalized_root != "/":
        if candidate == normalized_root:
            candidate = "/"
        elif candidate.startswith(f"{normalized_root}/"):
            candidate = candidate[len(normalized_root) :]

    return any(candidate == p or candidate.startswith(f"{p}/") for p in PUBLIC_PATHS)


# --- Key loading -----------------------------------------------------------
# Reads keys from environment variables.  Replace this function if you want
# to load keys from a database, config file, or secrets manager instead.


def _load_api_keys() -> set[str]:
    keys = {value.strip() for value in os.environ.get(API_KEYS_ENV_VAR, "").split(",") if value.strip()}
    single_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if single_key:
        keys.add(single_key)
    return keys


# --- Middleware -------------------------------------------------------------
# Registers an HTTP middleware on the FastAPI app.  Every request passes
# through it *before* reaching any route handler.
#
# Middleware is the right layer for auth when you need to protect ALL routes
# (including ones registered by create_app() before you have access to the
# router).  For per-route auth, use FastAPI dependencies instead:
#   https://fastapi.tiangolo.com/tutorial/security/
#
# secrets.compare_digest() is used instead of == to prevent timing attacks.


def _add_api_key_middleware(app: Any, valid_keys: set[str], root_path: str) -> None:
    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if _is_public_path(request.url.path, root_path):
            return await call_next(request)

        key = request.headers.get(API_KEY_HEADER)
        if not key:
            return JSONResponse(status_code=403, content={"detail": "Not authenticated"})
        if not any(secrets.compare_digest(key, allowed) for allowed in valid_keys):
            return JSONResponse(status_code=403, content={"detail": "Invalid API key"})

        return await call_next(request)


# --- OpenAPI security scheme -----------------------------------------------
# Patches the auto-generated OpenAPI schema so Swagger UI shows an
# "Authorize" button where users can enter their API key.  After authorizing,
# every "Try it out" request from Swagger includes the X-API-Key header.
#
# This function merges into the existing schema (using setdefault) so it
# won't overwrite security schemes added by other middleware or plugins.
#
# This step is entirely optional.  If you don't need the Swagger Authorize
# button, simply don't call this function in app.py.


def _add_openapi_security(app: Any) -> None:
    original_openapi = app.openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        schema = original_openapi()
        components = schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes.setdefault(
            "APIKeyHeader",
            {
                "type": "apiKey",
                "in": "header",
                "name": API_KEY_HEADER,
            },
        )

        security = schema.get("security")
        if not isinstance(security, list):
            security = []
            schema["security"] = security

        api_key_requirement = {"APIKeyHeader": []}
        if api_key_requirement not in security:
            security.append(api_key_requirement)

        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
