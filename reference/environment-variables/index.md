# Environment Variables

Hayhooks can be configured via environment variables (loaded with prefix `HAYHOOKS_` or `LOG`). This page lists the canonical variables supported by the codebase.

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

Priority Order

Pipeline-specific settings (via `streaming_components` parameter or YAML) override this global default.

Component-Specific Control

For component-specific control, use the `streaming_components` parameter in your code or YAML configuration instead of the environment variable to specify exactly which components should stream.

**Examples:**

```
# Stream all components globally
export HAYHOOKS_STREAMING_COMPONENTS="all"

# Stream specific components (comma-separated, spaces are trimmed)
export HAYHOOKS_STREAMING_COMPONENTS="llm_1,llm_2"
export HAYHOOKS_STREAMING_COMPONENTS="llm_1, llm_2, llm_3"
```

## MCP

### HAYHOOKS_MCP_HOST

- Default: `localhost`
- Description: Host for the MCP server

### HAYHOOKS_MCP_PORT

- Default: `1417`
- Description: Port for the MCP server

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
- Description: Path to a directory containing custom `.jsx` element files. These files are copied into the Chainlit `public/elements/` directory at startup and become available as `cl.CustomElement` targets. See [Custom Elements](https://deepset-ai.github.io/hayhooks/features/chainlit-integration/#custom-elements).

Installation Required

The Chainlit UI requires the `chainlit` extra: `pip install "hayhooks[chainlit]"`

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

- Default: `[]`
- Description: Headers to expose in response

### HAYHOOKS_CORS_MAX_AGE

- Default: `600`
- Description: Maximum age for CORS preflight responses in seconds

## Logging

### LOG (log level)

- Default: `INFO`
- Description: Global log level (consumed by Loguru). Example: `LOG=DEBUG hayhooks run`

Logging Configuration

Format/handlers are configured internally; Hayhooks does not expose `HAYHOOKS_LOG_FORMAT` or `HAYHOOKS_LOG_FILE` env vars at this time.

## Usage Examples

### Docker

```
docker run -d \
  -e HAYHOOKS_HOST=0.0.0.0 \
  -e HAYHOOKS_PORT=1416 \
  -e HAYHOOKS_PIPELINES_DIR=/app/pipelines \
  -v "$PWD/pipelines:/app/pipelines:ro" \
  -p 1416:1416 \
  deepset/hayhooks:main
```

Pipeline Directory Required

Without mounting a pipelines directory (or baking pipelines into the image), the server will start but no pipelines will be deployed.

### Development

```
export HAYHOOKS_HOST=127.0.0.1
export HAYHOOKS_PORT=1416
export HAYHOOKS_PIPELINES_DIR=./pipelines
export LOG=DEBUG

hayhooks run
```

### MCP Server startup

```
export HAYHOOKS_MCP_HOST=0.0.0.0
export HAYHOOKS_MCP_PORT=1417

hayhooks mcp run
```

### .env file example

```
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
HAYHOOKS_CORS_ALLOW_ORIGINS=["*"]
LOG=INFO
```

Configuration Note

- Worker count, timeouts, and other server process settings are CLI flags (e.g., `hayhooks run --workers 4`).
- YAML/file saving and MCP exposure are controlled per-deploy via API/CLI flags, not global env vars.

## Next Steps

- [Configuration](https://deepset-ai.github.io/hayhooks/getting-started/configuration/index.md)
- [Logging](https://deepset-ai.github.io/hayhooks/reference/logging/index.md)
