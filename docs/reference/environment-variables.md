# Environment Variables

Hayhooks can be configured via environment variables. Most app settings use the `HAYHOOKS_` prefix (plus the legacy `LOG` alias for log level), while tracing uses standard OpenTelemetry `OTEL_*` variables. This page lists the canonical variables supported by the codebase.

## Server

### HAYHOOKS_HOST

- Default: `localhost`
- Description: Host for the FastAPI app

### HAYHOOKS_PORT

- Default: `1416`
- Description: Port for the FastAPI app

### HAYHOOKS_ROOT_PATH

- Default: `""`
- Description: Root path to mount the API under (FastAPI `root_path`)

### HAYHOOKS_PIPELINES_DIR

- Default: `./pipelines`
- Description: Directory containing pipelines to auto-deploy on startup

### HAYHOOKS_ADDITIONAL_PYTHON_PATH

- Default: `""`
- Description: Additional path appended to `sys.path` for wrapper imports

### HAYHOOKS_USE_HTTPS

- Default: `false`
- Description: Use HTTPS when the CLI calls the server (affects CLI only)

### HAYHOOKS_DISABLE_SSL

- Default: `false`
- Description: Disable SSL verification for CLI calls

### HAYHOOKS_SHOW_TRACEBACKS

- Default: `false`
- Description: Include tracebacks in error messages (server and MCP)

### HAYHOOKS_STREAMING_COMPONENTS

- Default: `""` (empty string)
- Description: Global configuration for which pipeline components should stream
- Options:
  - `""` (empty): Stream only the last capable component (default)
  - `"all"`: Stream all streaming-capable components
  - Comma-separated list: `"llm_1,llm_2"` to enable specific components

!!! note "Priority Order"
    Pipeline-specific settings (via `streaming_components` parameter or YAML) override this global default.

!!! tip "Component-Specific Control"
    For component-specific control, use the `streaming_components` parameter in your code or YAML configuration instead of the environment variable to specify exactly which components should stream.

**Examples:**

```bash
# Stream all components globally
export HAYHOOKS_STREAMING_COMPONENTS="all"

# Stream specific components (comma-separated, spaces are trimmed)
export HAYHOOKS_STREAMING_COMPONENTS="llm_1,llm_2"
export HAYHOOKS_STREAMING_COMPONENTS="llm_1, llm_2, llm_3"
```

## Deploy Performance

### HAYHOOKS_DEPLOY_CONCURRENCY

- Default: `serialized`
- Description: Controls how runtime deploy/undeploy operations (via HTTP API and MCP) are synchronized
- Options:
  - `"serialized"` (default): One deploy/undeploy at a time. Safest option; prevents race conditions on the pipeline registry and FastAPI route table.
  - `"parallel"`: Allow concurrent deploy/undeploy operations. Higher throughput for admin traffic, but carries a higher risk of race conditions.

### HAYHOOKS_STARTUP_DEPLOY_STRATEGY

- Default: `parallel`
- Description: Controls how pipelines are deployed from `HAYHOOKS_PIPELINES_DIR` at startup
- Options:
  - `"sequential"`: Deploy one pipeline at a time (original behavior).
  - `"parallel"` (default): Prepare pipelines in a bounded thread pool, then commit routes serially and rebuild the OpenAPI schema once at the end. Significantly faster when many pipelines are deployed.

### HAYHOOKS_STARTUP_DEPLOY_WORKERS

- Default: `4`
- Description: Maximum number of worker threads used for parallel startup deployment (only effective when `HAYHOOKS_STARTUP_DEPLOY_STRATEGY=parallel`)

**Examples:**

```bash
# Fastest startup: parallel with 8 workers
export HAYHOOKS_STARTUP_DEPLOY_STRATEGY=parallel
export HAYHOOKS_STARTUP_DEPLOY_WORKERS=8

# Safe runtime: serialize all deploy/undeploy (default)
export HAYHOOKS_DEPLOY_CONCURRENCY=serialized

# Allow concurrent runtime deploys (advanced)
export HAYHOOKS_DEPLOY_CONCURRENCY=parallel
```

## MCP

### HAYHOOKS_MCP_HOST

- Default: `localhost`
- Description: Host for the MCP server

### HAYHOOKS_MCP_PORT

- Default: `1417`
- Description: Port for the MCP server

## A2A

### HAYHOOKS_A2A_HOST

- Default: `localhost`
- Description: Host for the A2A server

### HAYHOOKS_A2A_PORT

- Default: `1418`
- Description: Port for the A2A server

### HAYHOOKS_A2A_EXTERNAL_URL

- Default: `""` (empty)
- Description: Base URL advertised in A2A agent cards (e.g. when Hayhooks runs behind a reverse proxy). When empty, defaults to `http://{HAYHOOKS_A2A_HOST}:{HAYHOOKS_A2A_PORT}`

### HAYHOOKS_A2A_V0_3_COMPAT

- Default: `true`
- Description: Accept A2A spec 0.3 requests on the same endpoints. Many clients and tools (e.g. the a2a-inspector) still speak 0.3 during the 1.0 transition

## Chainlit UI

### HAYHOOKS_CHAINLIT_ENABLED

- Default: `false`
- Description: Enable the embedded Chainlit chat UI

### HAYHOOKS_CHAINLIT_PATH

- Default: `/chat`
- Description: URL path where the Chainlit UI is mounted

### HAYHOOKS_CHAINLIT_APP

- Default: `""` (uses built-in default app)
- Description: Path to a custom Chainlit app file

### HAYHOOKS_CHAINLIT_DEFAULT_MODEL

- Default: `""` (auto-selects if only one pipeline is deployed)
- Description: Default pipeline/model to auto-select in the Chainlit UI

### HAYHOOKS_CHAINLIT_REQUEST_TIMEOUT

- Default: `120.0`
- Description: Timeout in seconds for chat completion requests from the Chainlit UI

### HAYHOOKS_CHAINLIT_CUSTOM_ELEMENTS_DIR

- Default: `""` (no custom elements)
- Description: Path to a directory containing custom `.jsx` element files. These files are copied into the Chainlit `public/elements/` directory at startup and become available as `cl.CustomElement` targets. See [Custom Elements](../features/chainlit-integration.md#custom-elements).

!!! note "Installation Required"
    The Chainlit UI requires the `chainlit` extra: `pip install "hayhooks[chainlit]"`

## Dashboard UI

### HAYHOOKS_DASHBOARD_ENABLED

- Default: `false`
- Description: Enable serving the built dashboard static frontend
- Notes:
  - `/dashboard/api/traces` always reads from Hayhooks' in-process live trace buffer.
  - The dashboard trace buffer is process-local. With `hayhooks run --workers >1`, each worker has its own
    buffer, so the UI can show partial traces and clear operations are worker-local.
  - For a consistent dashboard view, run Hayhooks with a single worker (`--workers 1`).

### HAYHOOKS_DASHBOARD_PATH

- Default: `/dashboard`
- Description: URL path where the dashboard static frontend is mounted

### HAYHOOKS_DASHBOARD_DIST_DIR

- Default: `./dashboard/dist`
- Description: Directory containing dashboard static assets (`index.html`, JS/CSS chunks). Override this when
  serving a custom or locally rebuilt dashboard frontend.

### HAYHOOKS_DASHBOARD_UI_POLL_MS

- Default: `2500`
- Description: Dashboard UI polling interval in milliseconds.

### HAYHOOKS_DASHBOARD_UI_LIST_CAP

- Default: `100`
- Description: Maximum number of traces the dashboard keeps in the browser list.

### HAYHOOKS_DASHBOARD_UI_FETCH_LIMIT

- Default: `50`
- Description: Number of traces the dashboard requests per poll. This is capped by
  `HAYHOOKS_DASHBOARD_TRACE_MAX_LIMIT` and `HAYHOOKS_DASHBOARD_UI_LIST_CAP`.

### HAYHOOKS_DASHBOARD_UI_FRESH_MS

- Default: `6000`
- Description: Duration (milliseconds) that newly seen traces keep the "NEW" highlight state.

### HAYHOOKS_DASHBOARD_UI_SLOW_COMPONENT_MIN_DURATION_MS

- Default: `1000`
- Description: Minimum duration (milliseconds) required for the dashboard to mark a trace component as "slow".
- Notes:
  - The UI highlights only the single slowest component per trace.
  - Highlighting is applied only when that component duration is strictly greater than this threshold.

### HAYHOOKS_DASHBOARD_TRACE_DEFAULT_LIMIT

- Default: `25`
- Description: Default number of traces returned from the in-process live trace buffer per dashboard query

### HAYHOOKS_DASHBOARD_TRACE_MAX_LIMIT

- Default: `100`
- Description: Upper bound for `limit` in `/dashboard/api/traces` (local buffer reads)

### HAYHOOKS_DASHBOARD_TRACE_BUFFER_CAPACITY

- Default: `200`
- Description: Maximum number of traces retained in the in-process live trace buffer.
- Notes:
  - Behaves like a capped collection: when full, older traces are evicted automatically.
  - Increase this (for example to `2000`) if you want a larger local history window.

### HAYHOOKS_DASHBOARD_TRACE_INCLUDE_HAYSTACK_SPANS

- Default: `true`
- Description: Include Haystack tracer spans in the dashboard live buffer in addition to Hayhooks operation spans.
- Notes:
  - Works in local capture mode even when no external tracing backend is configured.
  - If OpenTelemetry/Datadog tracing is configured, the same Haystack spans are also exported to that backend.
  - Keeps dashboard polling/filter behavior unchanged; it only broadens which spans appear in traces.

### HAYHOOKS_DASHBOARD_TRACE_INCLUDE_PAYLOAD_VALUES

- Default: `false`
- Description: Include raw pipeline request payload values in dashboard trace tags.
- Notes:
  - When disabled, dashboard traces include payload shape metadata such as `question=str(24)`, `top_k=int`, or
    `filters=dict(3)`, rather than the raw values.
  - When enabled, raw scalar values and JSON-serializable payload values are included in `hayhooks.payload.values`.
  - Common sensitive keys are still redacted in raw-value mode, including keys containing `api_key`, `token`,
    `authorization`, `password`, `secret`, `credential`, or related variants.
  - Enable this only for trusted local debugging sessions, because payloads can contain prompts, documents, or PII.

## CORS

These map 1:1 to FastAPI CORSMiddleware and the settings in `hayhooks.settings.AppSettings`.

### HAYHOOKS_CORS_ALLOW_ORIGINS

- Default: `["*"]`
- Description: List of allowed origins

### HAYHOOKS_CORS_ALLOW_METHODS

- Default: `["*"]`
- Description: List of allowed HTTP methods

### HAYHOOKS_CORS_ALLOW_HEADERS

- Default: `["*"]`
- Description: List of allowed headers

### HAYHOOKS_CORS_ALLOW_CREDENTIALS

- Default: `false`
- Description: Allow credentials

### HAYHOOKS_CORS_ALLOW_ORIGIN_REGEX

- Default: `null`
- Description: Regex pattern for allowed origins

### HAYHOOKS_CORS_EXPOSE_HEADERS

- Default: `["X-Hayhooks-Trace-Cursor"]`
- Description: Headers to expose in response

### HAYHOOKS_CORS_MAX_AGE

- Default: `600`
- Description: Maximum age for CORS preflight responses in seconds

## Logging

### HAYHOOKS_LOG_LEVEL

- Default: `INFO`
- Alias: `LOG` (legacy, lower priority)
- Description: Minimum log level to display (consumed by Loguru). When both `HAYHOOKS_LOG_LEVEL` and `LOG` are set, `HAYHOOKS_LOG_LEVEL` takes precedence.

### HAYHOOKS_LOG_FORMAT

- Default: `default`
- Description: Controls log output verbosity
- Options:
  - `"default"`: Timestamp, level, and message
  - `"verbose"`: Adds module, function, and line number metadata

### HAYHOOKS_INTERCEPTED_LOGGERS

- Default: `["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]`
- Description: List of stdlib loggers to intercept and route through Loguru. Only the listed loggers are patched; all others (httpx, haystack, etc.) keep their default behaviour.

### HAYHOOKS_ACCESS_LOG_EXCLUDED_PATH_PREFIXES

- Default: `["/dashboard/api/config", "/dashboard/api/entrypoints", "/dashboard/api/traces"]`
- Description: Path prefixes for `uvicorn.access` request logs that should be suppressed.
- Notes:
  - Useful to reduce noise from dashboard polling requests.
  - Set to `[]` to keep all access logs.

**Examples:**

```bash
# Set log level
export HAYHOOKS_LOG_LEVEL=DEBUG

# Enable verbose log format (includes module:function:line)
export HAYHOOKS_LOG_FORMAT=verbose

# Also intercept haystack logs
export HAYHOOKS_INTERCEPTED_LOGGERS='["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "haystack"]'

# Keep dashboard polling request logs
export HAYHOOKS_ACCESS_LOG_EXCLUDED_PATH_PREFIXES='[]'
```

## OpenTelemetry (Tracing)

Hayhooks tracing relies on Haystack tracing APIs and standard OpenTelemetry configuration.

For setup walkthroughs and backend examples (SigNoz, Jaeger), see [Tracing](tracing.md).

Install tracing extras first:

```bash
pip install "hayhooks[tracing]"
```

Then configure your exporter/provider using OpenTelemetry env vars (examples):

### OTEL_SERVICE_NAME

- Example: `hayhooks`
- Description: Service name shown in your tracing backend

### OTEL_EXPORTER_OTLP_ENDPOINT

- Example: `http://localhost:4318`
- Description: OTLP collector endpoint

!!! tip "Using OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    If you set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` directly for HTTP transport, use the full traces path
    (for example `http://localhost:4318/v1/traces`).

### OTEL_EXPORTER_OTLP_TRACES_PROTOCOL / OTEL_EXPORTER_OTLP_PROTOCOL

- Example: `http/protobuf`
- Description: OTLP transport protocol (`http/protobuf` or `grpc`). If both are set, `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` takes precedence for trace export.

### OTEL_TRACES_SAMPLER

- Example: `parentbased_traceidratio`
- Description: Sampling strategy

### OTEL_TRACES_SAMPLER_ARG

- Example: `1.0`
- Description: Sampler argument (e.g. ratio)

### HAYHOOKS_TRACING_EXCLUDED_SPANS

- Default: `["send", "receive"]`
- Description: Low-level ASGI child spans to suppress from FastAPI/Starlette instrumentation. This helps keep traces readable for streaming responses by default.
- Options:
  - `["send", "receive"]` (default): suppress both per-chunk response send spans and request receive spans
  - `["send"]`: keep receive spans, suppress response send spans
  - `[]`: disable this filtering and keep all framework spans

!!! note "No Hayhooks-specific OTel env vars"
    Hayhooks does not introduce a custom `HAYHOOKS_OTEL_*` exporter/provider namespace. Use standard OpenTelemetry
    variables for backend configuration. `HAYHOOKS_TRACING_EXCLUDED_SPANS` is a Hayhooks-specific instrumentation tuning
    setting for framework span noise, not an OpenTelemetry exporter setting.

When `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set, Hayhooks will also attempt
an automatic OpenTelemetry bootstrap at startup using the protocol from `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`
(falling back to `OTEL_EXPORTER_OTLP_PROTOCOL`)
(`http/protobuf` default, or `grpc`).

## Usage Examples

### Docker

```bash
docker run -d \
  -e HAYHOOKS_HOST=0.0.0.0 \
  -e HAYHOOKS_PORT=1416 \
  -e HAYHOOKS_PIPELINES_DIR=/app/pipelines \
  -v "$PWD/pipelines:/app/pipelines:ro" \
  -p 1416:1416 \
  deepset/hayhooks:main
```

!!! warning "Pipeline Directory Required"
    Without mounting a pipelines directory (or baking pipelines into the image), the server will start but no pipelines will be deployed.

### Development

```bash
export HAYHOOKS_HOST=127.0.0.1
export HAYHOOKS_PORT=1416
export HAYHOOKS_PIPELINES_DIR=./pipelines
export HAYHOOKS_LOG_LEVEL=DEBUG

hayhooks run
```

### MCP Server startup

```bash
export HAYHOOKS_MCP_HOST=0.0.0.0
export HAYHOOKS_MCP_PORT=1417

hayhooks mcp run
```

### .env file example

```env
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_MCP_HOST=0.0.0.0
HAYHOOKS_MCP_PORT=1417
HAYHOOKS_PIPELINES_DIR=./pipelines
HAYHOOKS_ADDITIONAL_PYTHON_PATH=./custom_code
HAYHOOKS_USE_HTTPS=false
HAYHOOKS_DISABLE_SSL=false
HAYHOOKS_SHOW_TRACEBACKS=false
HAYHOOKS_STREAMING_COMPONENTS=all
HAYHOOKS_DEPLOY_CONCURRENCY=serialized
HAYHOOKS_STARTUP_DEPLOY_STRATEGY=parallel
HAYHOOKS_STARTUP_DEPLOY_WORKERS=4
HAYHOOKS_DASHBOARD_ENABLED=true
HAYHOOKS_DASHBOARD_DIST_DIR=./dashboard/dist
HAYHOOKS_DASHBOARD_TRACE_DEFAULT_LIMIT=25
HAYHOOKS_DASHBOARD_TRACE_MAX_LIMIT=100
# Optional: only highlight slowest components above 1s
# HAYHOOKS_DASHBOARD_UI_SLOW_COMPONENT_MIN_DURATION_MS=1000
# Optional: exclude Haystack component spans (included by default)
# HAYHOOKS_DASHBOARD_TRACE_INCLUDE_HAYSTACK_SPANS=false
# Optional: include raw request payload values in dashboard trace tags (safe metadata only by default)
# HAYHOOKS_DASHBOARD_TRACE_INCLUDE_PAYLOAD_VALUES=true
HAYHOOKS_CORS_ALLOW_ORIGINS=["*"]
HAYHOOKS_LOG_LEVEL=INFO
HAYHOOKS_LOG_FORMAT=default
```

!!! info "Configuration Note"
    - Worker count, timeouts, and other server process settings are CLI flags (e.g., `hayhooks run --workers 4`).
    - YAML/file saving and MCP exposure are controlled per-deploy via API/CLI flags, not global env vars.

## Next Steps

- [Configuration](../getting-started/configuration.md)
- [Tracing](tracing.md)
- [Logging](logging.md)
