# API Key Authentication Example

A complete end-to-end example that protects all Hayhooks API endpoints with API-key authentication (single key or multiple keys) and deploys a weather agent you can query.

## What's Included

```text
api_key_auth/
├── app.py                                  # Thin entrypoint
├── auth.py                                 # Auth middleware and OpenAPI helpers
├── pipelines/
│   └── weather_agent/
│       └── pipeline_wrapper.py             # Haystack Agent with a weather tool
├── test_api_key_auth.py                    # Integration tests
└── README.md
```

## How It Works

`create_app()` returns a standard FastAPI application. This example keeps `app.py` thin and moves auth internals into `auth.py`:

1. **`APIKeyMiddleware`** — checks `X-API-Key` on every request.
2. **Public docs paths** — `/docs`, `/redoc`, and `/openapi.json` remain public so documentation is reachable.
3. **Multi-key support** — accepts keys from:
   - `HAYHOOKS_API_KEYS` (comma-separated list), and/or
   - `HAYHOOKS_API_KEY` (single-key fallback)
4. **Swagger Authorize button** — OpenAPI schema is extended with an API-key security scheme so `/docs` shows **Authorize** and sends `X-API-Key`.

Key validation uses `secrets.compare_digest()` to prevent timing attacks.

At startup, Hayhooks auto-deploys the `weather_agent` pipeline from the `pipelines/` directory.

## Prerequisites

- An OpenAI API key (the agent uses `gpt-4o-mini`)
- Install pinned dependencies: `pip install -r requirements.txt`

## Run

```bash
cd examples/programmatic/api_key_auth

# One or more API keys:
export HAYHOOKS_API_KEYS="my-secret-key,another-key"

# Optional single-key fallback:
# export HAYHOOKS_API_KEY="my-secret-key"

export OPENAI_API_KEY="sk-..."

python app.py
```

You should see the weather agent being auto-deployed in the logs.

## Try It

### Check status

```bash
# Without a key → 403
curl -s http://localhost:1416/status
# {"detail":"Not authenticated"}

# Wrong key → 403
curl -s -H "X-API-Key: wrong" http://localhost:1416/status
# {"detail":"Invalid API key"}

# Correct key (any configured key) → 200
curl -s -H "X-API-Key: my-secret-key" http://localhost:1416/status
```

### Call the agent via /run

```bash
curl -s -X POST http://localhost:1416/weather_agent/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret-key" \
  -d '{"question": "What is the weather in Berlin?"}'
```

### Call the agent via OpenAI-compatible /chat/completions

```bash
curl -s -X POST http://localhost:1416/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret-key" \
  -d '{
    "model": "weather_agent",
    "messages": [{"role": "user", "content": "What is the weather in Berlin?"}]
  }'
```

### Use the Swagger UI

Open `http://localhost:1416/docs` (no key required), click **Authorize**, set your key in `X-API-Key`, and test protected endpoints directly from Swagger.

## Run Tests

The tests exercise the auth layer only and don't require OpenAI keys or any external service:

```bash
hatch run test:pytest examples/programmatic/api_key_auth/test_api_key_auth.py -v
```

## Customization Ideas

- **Multiple keys / roles** — Load keys from a database or config file and map them to roles.
- **Rate limiting** — Combine with a rate-limiting middleware like [slowapi](https://github.com/laurentS/slowapi).
- **Public paths** — Edit the `PUBLIC_PATHS` set in `auth.py` to allow unauthenticated access to additional endpoints.
- **Bearer token** — Check `Authorization: Bearer <key>` in addition to (or instead of) `X-API-Key`.
