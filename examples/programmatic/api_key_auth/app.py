"""
Hayhooks with API Key Authentication.

Demonstrates how to use ``create_app()`` programmatically and protect
all endpoints with API-key authentication.

Usage::

    cd examples/programmatic/api_key_auth
    export HAYHOOKS_API_KEYS="my-secret-key,another-secret-key"
    # Optional fallback for a single key:
    # export HAYHOOKS_API_KEY="my-secret-key"
    export OPENAI_API_KEY="sk-..."
    python app.py

Clients must include the key as::

    X-API-Key: my-secret-key

How it works
------------
1. ``create_app()`` returns a standard FastAPI application with all Hayhooks
   routes already registered.
2. We register an HTTP **middleware** that checks the ``X-API-Key`` header on
   every request (except Swagger docs).  See ``auth.py`` for details.
3. We patch the **OpenAPI schema** so the Swagger UI shows an Authorize button.
"""

import sys

import uvicorn
from auth import API_KEY_ENV_VAR, API_KEYS_ENV_VAR, _add_api_key_middleware, _add_openapi_security, _load_api_keys

from hayhooks import create_app, log
from hayhooks.settings import settings


def create_authenticated_app():
    # 1. Load API keys from environment variables.
    valid_keys = _load_api_keys()
    if not valid_keys:
        log.error(
            "Set {} (comma-separated) or {} (single key) before starting the server.",
            API_KEYS_ENV_VAR,
            API_KEY_ENV_VAR,
        )
        sys.exit(1)

    # 2. Create the Hayhooks app (all default routes are registered here).
    app = create_app()

    # 3. Add the auth middleware — protects every route globally.
    _add_api_key_middleware(app, valid_keys, settings.root_path)

    # 4. (Optional) Patch OpenAPI so /docs shows an "Authorize" button.
    _add_openapi_security(app)

    return app


# The app object is created at module level so that Uvicorn can import it
# directly (e.g. ``uvicorn app:hayhooks``).  We pass it as an object to
# uvicorn.run() below to avoid double-import issues.
hayhooks = create_authenticated_app()

if __name__ == "__main__":
    uvicorn.run(hayhooks, host=settings.host, port=settings.port)
