# CLI Commands

Hayhooks provides a comprehensive command-line interface for managing pipelines and the server. This section covers all available CLI commands and their usage.

## Overview

The Hayhooks CLI allows you to:

- Start and manage the Hayhooks server
- Deploy and undeploy pipelines
- Run pipelines with custom inputs
- Monitor server status and health
- Manage MCP server operations

## Installation

The CLI is automatically installed with the Hayhooks package:

```
pip install hayhooks
```

## Global Commands

### Help

Get help for any command:

```
# Show main help
hayhooks --help

# Show help for specific command
hayhooks run --help
hayhooks pipeline --help
```

### Version

Check the installed version:

```
hayhooks --version
```

## Server Commands

### run

Start the Hayhooks server:

```
# Basic server start
hayhooks run

# With custom host and port
hayhooks run --host 0.0.0.0 --port 1416

# With multiple workers in default mode
hayhooks run --workers 4

# With custom pipelines directory
hayhooks run --pipelines-dir ./my_pipelines

# With additional Python path
hayhooks run --additional-python-path ./custom_code

# Reload on changes (development)
hayhooks run --reload

# Disable local trace inspection if it is not needed
hayhooks run --without-tracing
```

Trace inspection reads a process-local buffer. With multiple workers, use a connection pinned to one process or an external OTLP backend for aggregate history.

Durable mode uses one worker per process

`hayhooks run --workers N` with `N > 1` exits with an error in durable mode. Scale with replicas that share Redis, and do not place a durable-mode app under another multi-worker supervisor such as `uvicorn --workers` or gunicorn.

#### Options for `run`

| Option                                   | Short | Description                                                             | Default                      |
| ---------------------------------------- | ----- | ----------------------------------------------------------------------- | ---------------------------- |
| `--host`                                 |       | Host to bind to                                                         | `localhost`                  |
| `--port`                                 |       | Port to listen on                                                       | `1416`                       |
| `--workers`                              |       | Number of worker processes; values above 1 are rejected in durable mode | `1`                          |
| `--pipelines-dir`                        |       | Directory for pipeline definitions                                      | `./pipelines`                |
| `--additional-python-path`               |       | Additional Python path                                                  | `None`                       |
| `--root-path`                            |       | Root path for API                                                       | `""`                         |
| `--reload`                               | `-r`  | Reload on code changes (development)                                    | `false`                      |
| `--with-chainlit`                        |       | Mount the embedded Chainlit UI                                          | `false`                      |
| `--chainlit-path`                        |       | Chainlit mount path                                                     | `/chat`                      |
| `--chainlit-custom-elements-dir`         |       | Directory of custom JSX elements                                        | Unset                        |
| `--with-monitoring/--without-monitoring` |       | Authenticated durable execution indexing and collection routes          | Enabled by settings (`true`) |
| `--with-tracing/--without-tracing`       |       | Authenticated local trace API for terminal inspection                   | Enabled by settings (`true`) |

### dashboard

For a visual tour, first-run setup, and remote connections, see the [terminal dashboard guide](https://deepset-ai.github.io/hayhooks/features/terminal-dashboard/index.md).

Browse deployments, ordinary and durable executions, and recent process traces using GET requests only:

```
pip install 'hayhooks[tui]'
hayhooks dashboard
hayhooks dashboard --url https://example.com/hayhooks --pipeline research_agent
hayhooks dashboard --traces
hayhooks dashboard --interval 5
hayhooks dashboard --once --json
```

`--traces` opens Traces initially; with `--once --json`, it adds a bounded `tracing` snapshot alongside status.

`--url` preserves reverse-proxy path prefixes. The default URL follows the existing host/port/HTTPS settings. TLS verification stays enabled, including when `HAYHOOKS_DISABLE_SSL` is set; use `REQUESTS_CA_BUNDLE` for a private CA. The client uses 3-second connection and 5-second read timeouts. Polling defaults to two seconds and backs off up to 30 seconds after failures.

Local POSIX servers and clients under the same OS account automatically share a private operator credential; see [local authentication](https://deepset-ai.github.io/hayhooks/reference/tracing/#operator-authorization). Explicit credentials come from `HAYHOOKS_DASHBOARD_TOKEN` (bearer token) or `HAYHOOKS_DASHBOARD_API_KEY` (`X-API-Key`, override its name with `HAYHOOKS_DASHBOARD_API_KEY_HEADER`). Keep credentials out of URLs. [Monitoring authentication](https://deepset-ai.github.io/hayhooks/deployment/durable-operations/#monitoring-authentication) shows the supported FastAPI dependency integration.

Deployments show the direct Haystack target's type, execution mode, description, API capabilities and durable health. Custom wrappers can declare `deployment_type = "agent"`, `"pipeline"`, or `"unknown"`; other values fail validation. A chat endpoint or MCP/A2A eligibility does not establish agent identity or prove a protocol server is running. `Loaded` means registered, not that external services were probed. Workers describe the responding process; store/revision counts are labeled separately.

Use up/down to select rows, left/right to scroll wide tables, Tab/Shift+Tab to move focus, Enter for details, Esc to return, `/` to search, `d` for Deployments, `e` for Executions and `v` for Traces, Space to pause, `r` to refresh, `?` for help, and `q` or Ctrl+C to exit. Text inputs keep ordinary shortcut characters as text; Ctrl+C always exits. Mouse selection and scrolling are supported. Details appear alongside the list on wide terminals and on a separate screen below 110 columns. `t` toggles themes. `v` opens Traces, where entrypoint/kind/state filters, newest/slowest sorting and span trees are available. `c` hides current traces locally. See [tracing setup and operator authorization](https://deepset-ai.github.io/hayhooks/reference/tracing/#terminal-trace-inspection).

The dashboard uses a charcoal blue dark palette and a warm light palette. Short terminals condense spacing to leave room for rows; wide terminals group metadata, process health and store counts in an inspector pane. Long IDs and revisions use ellipses in the table; details preserve their full values and timestamps.

Deployment searches and type filters apply to the complete deployment snapshot. Executions show all deployments by default. Ordinary runs come from the bounded recent process trace buffer; durable executions come from their indexed store. Durable attempts are not duplicated as ordinary runs. Choose a deployment to narrow the view. ID searches and status filters apply to both sources; revision filters apply only to stored durable history. Durable searches query the index beyond the visible page; ordinary searches cover only the recent sample. Next/Previous buttons (or `]`/`[`) navigate durable pages of up to 50 rows per deployment. Recent ordinary runs appear on page 1 only. A sparse durable page can have a next page. Durable kind, revision and attempt remain available in details; Enter on an ordinary run opens its span tree. Waiting, progress and errors are shown only for the selected item, bounded and rendered as literal text. Public results are discarded. Actual trace durations are shown for ordinary runs; durable creation/update timestamps are not treated as runtime duration.

Capture and durable indexing are enabled by default, with authentication required. They do not launch the terminal UI. Ordinary history is process-local and disappears on buffer eviction or server restart; durable retention follows the execution store. Multiple server workers have separate trace samples. All writing replicas must index submissions for complete durable coverage. See [index coverage and retention](https://deepset-ai.github.io/hayhooks/deployment/durable-operations/#execution-monitoring).

Standalone `hayhooks mcp run` and `hayhooks a2a run` servers do not expose the dashboard trace API. Their separate process buffers are not visible from a dashboard connected to the main HTTP server.

`--once` prints one JSON status snapshot without Textual; `--json` requires `--once`. Adding `--pipeline NAME` also includes that deployment's execution page. Stdout contains JSON only. Errors go to stderr with exit code 1; a degraded server returns its structured snapshot and exits 1. Interactive failures retain the last good snapshot with stale/age indicators. Pausing stops polling and rejects in-flight updates; resuming refreshes the view.

### mcp run

Start the MCP server:

```
# Start MCP server
hayhooks mcp run

# With custom host and port
hayhooks mcp run --host 0.0.0.0 --port 1417
```

#### Options for `mcp run`

| Option                     | Short | Description                        | Default       |
| -------------------------- | ----- | ---------------------------------- | ------------- |
| `--host`                   |       | MCP server host                    | `localhost`   |
| `--port`                   |       | MCP server port                    | `1417`        |
| `--pipelines-dir`          |       | Directory for pipeline definitions | `./pipelines` |
| `--additional-python-path` |       | Additional Python path             | `None`        |

### a2a run

Start the A2A server, exposing deployed pipelines as [A2A agents](https://deepset-ai.github.io/hayhooks/features/a2a-support/index.md):

```
# Start A2A server
hayhooks a2a run

# With custom host and port
hayhooks a2a run --host 0.0.0.0 --port 1418
```

#### Options for `a2a run`

| Option                     | Short | Description                        | Default       |
| -------------------------- | ----- | ---------------------------------- | ------------- |
| `--host`                   |       | A2A server host                    | `localhost`   |
| `--port`                   |       | A2A server port                    | `1418`        |
| `--pipelines-dir`          |       | Directory for pipeline definitions | `./pipelines` |
| `--additional-python-path` |       | Additional Python path             | `None`        |
| `--external-url`           |       | Base URL advertised in agent cards | `None`        |

## Pipeline Management Commands

The deployment commands below (`deploy-files`, `deploy-yaml`, and `undeploy`) send requests to the target server and work against a server in the default, live-deployment mode. They stay available whatever the CLI's own `HAYHOOKS_DURABLE_MODE` setting is. A server in [durable mode](https://deepset-ai.github.io/hayhooks/features/durable-execution/#durable-mode) has no deployment endpoints, so these commands fail there with `Server error: Not Found`; change its pipelines directory and restart it instead. `GET /status` reports the server's `durable_mode`.

### pipeline deploy-files

Deploy a pipeline from wrapper files:

```
# Basic deployment
hayhooks pipeline deploy-files -n my_pipeline ./path/to/pipeline

# Overwrite existing pipeline
hayhooks pipeline deploy-files -n my_pipeline --overwrite ./path/to/pipeline

# Skip saving files to server
hayhooks pipeline deploy-files -n my_pipeline --skip-saving-files ./path/to/pipeline
```

#### Options for `pipeline deploy-files`

| Option                | Short | Description                 | Default  |
| --------------------- | ----- | --------------------------- | -------- |
| `--name`              | `-n`  | Pipeline name               | Required |
| `--overwrite`         | `-o`  | Overwrite existing pipeline | `false`  |
| `--skip-saving-files` |       | Don't save files to server  | `false`  |

Use a `run_api` or `run_api_async` docstring for the description, and set the wrapper class attribute `skip_mcp = True` to omit MCP tool registration.

### pipeline deploy-yaml

Deploy a pipeline from YAML definition:

```
# Deploy from YAML file
hayhooks pipeline deploy-yaml pipelines/my_pipeline.yml

# With custom name
hayhooks pipeline deploy-yaml -n my_custom_name pipelines/my_pipeline.yml

# With description
hayhooks pipeline deploy-yaml -n my_pipeline --description "YAML pipeline" pipelines/my_pipeline.yml

# Overwrite existing
hayhooks pipeline deploy-yaml -n my_pipeline --overwrite pipelines/my_pipeline.yml

# Don't save YAML file
hayhooks pipeline deploy-yaml -n my_pipeline --no-save-file pipelines/my_pipeline.yml
```

#### Options for `pipeline deploy-yaml`

| Option           | Short | Description                 | Default        |
| ---------------- | ----- | --------------------------- | -------------- |
| `--name`         | `-n`  | Pipeline name               | YAML file stem |
| `--description`  |       | Human-readable description  | Pipeline name  |
| `--overwrite`    | `-o`  | Overwrite existing pipeline | `false`        |
| `--skip-mcp`     |       | Skip MCP tool registration  | `false`        |
| `--save-file`    |       | Save YAML to server         | `true`         |
| `--no-save-file` |       | Don't save YAML to server   | `false`        |

### pipeline undeploy

Undeploy a pipeline:

```
# Undeploy by name
hayhooks pipeline undeploy my_pipeline
```

### pipeline run

Run a deployed pipeline:

```
# Run with JSON parameters
hayhooks pipeline run my_pipeline --param 'query=What is Haystack?'

# Run with multiple parameters
hayhooks pipeline run my_pipeline --param 'query=What is Haystack?' --param 'max_results=5'

# Upload files
hayhooks pipeline run my_pipeline --file document.pdf --param 'query=Summarize this'

# Upload directory
hayhooks pipeline run my_pipeline --dir ./documents --param 'query=Analyze all documents'

# Upload multiple files
hayhooks pipeline run my_pipeline --file doc1.pdf --file doc2.txt --param 'query=Compare documents'
```

#### Options for `pipeline run`

| Option     | Short | Description                                                                       | Default |
| ---------- | ----- | --------------------------------------------------------------------------------- | ------- |
| `--file`   |       | Upload single file                                                                | None    |
| `--dir`    |       | Upload directory                                                                  | None    |
| `--param`  | `-p`  | Pass `key=value`; arrays, objects, booleans, null, and numbers are parsed as JSON | None    |
| `--stream` | `-s`  | Display a streaming `/run` response as it arrives; unavailable with file uploads  | `false` |

## Status and Monitoring Commands

### status

Check server and pipeline status:

```
# Check server status
hayhooks status
```

Note

There is no `hayhooks health` command in the CLI. Use `hayhooks status` or call HTTP endpoints directly.

CLI Limitations

- Configuration is managed via environment variables and CLI flags. See [Configuration](https://deepset-ai.github.io/hayhooks/getting-started/configuration/index.md).
- Development flows (testing, linting) are not exposed as CLI commands.
- Use your process manager or container logs to view logs; Hayhooks uses standard output.
- Advanced export/import/migrate commands are not provided by the Hayhooks CLI at this time.

## HTTP API Commands

Pipeline management and execution commands call the HTTP API. Server startup and the dashboard run locally.

Interactive API Documentation

Explore and test all HTTP API endpoints interactively:

- **Swagger UI**: `http://localhost:1416/docs`
- **ReDoc**: `http://localhost:1416/redoc`

See the [API Reference](https://deepset-ai.github.io/hayhooks/reference/api-reference/index.md) for complete documentation.

**Common HTTP API equivalents:**

### deploy-files

```
# CLI
hayhooks pipeline deploy-files -n my_pipeline ./path/to/pipeline

# HTTP API
curl -X POST http://localhost:1416/deploy_files \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "my_pipeline",
    "files": {
      "pipeline_wrapper.py": "from hayhooks import BasePipelineWrapper\nclass PipelineWrapper(BasePipelineWrapper):\n    def setup(self): pass\n    def run_api(self, query: str) -> str: return query\n"
    },
    "save_files": true,
    "overwrite": false
  }'
```

### deploy-yaml

```
# CLI
hayhooks pipeline deploy-yaml -n my_pipeline pipelines/my_pipeline.yml

# HTTP API
curl -X POST http://localhost:1416/deploy-yaml \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "my_pipeline",
    "description": "My pipeline",
    "source_code": "...",
    "overwrite": false
  }'
```

### run

```
# CLI
hayhooks pipeline run my_pipeline --param 'query=What is Haystack?'

# HTTP API
curl -X POST http://localhost:1416/my_pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is Haystack?"}'
```

## Configuration Files

### .env File

Create a `.env` file for configuration:

```
# .env
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_MCP_PORT=1417
HAYHOOKS_PIPELINES_DIR=./pipelines
LOG=INFO
```

## Error Handling

### Common Errors

1. **Server already running**

```
# Check if server is running
hayhooks status

# Kill existing process
pkill -f "hayhooks run"
```

1. **Pipeline deployment failed**

```
# Check server logs with your process manager or container runtime

# Enable debug logging
LOG=DEBUG hayhooks run
```

1. **Permission denied**

```
# Check file permissions
ls -la ./path/to/pipeline

# Grant the owner read access if needed
chmod u+r ./path/to/pipeline/pipeline_wrapper.py
```

### Debug Mode

Enable debug mode for troubleshooting:

```
# Set debug logging
export LOG=DEBUG

# Start server with debug logging
hayhooks run
```

## Examples

### Basic Workflow

```
# 1. Start server
hayhooks run --port 1416

# 2. In another terminal, deploy pipeline
hayhooks pipeline deploy-files -n chat_pipeline ./pipelines/chat

# 3. Check status
hayhooks status

# 4. Run pipeline
hayhooks pipeline run chat_pipeline --param 'query=Hello!'

# 5. Check logs
# Use your process manager or container logs
```

### Production Deployment

```
# 1. Set environment variables
export HAYHOOKS_HOST=0.0.0.0
export HAYHOOKS_PORT=1416
export LOG=INFO

# 2. Load the same pipeline definitions in every worker at startup
export HAYHOOKS_PIPELINES_DIR=./pipelines
hayhooks run --workers 4

# 3. In another terminal, monitor status
hayhooks status
```

## Next Steps

- [Configuration](https://deepset-ai.github.io/hayhooks/getting-started/configuration/index.md) - Configuration options
- [Examples](https://deepset-ai.github.io/hayhooks/examples/overview/index.md) - Working examples
