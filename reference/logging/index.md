# Logging

Hayhooks provides comprehensive logging capabilities for monitoring, debugging, and auditing pipeline execution and server operations.

## Log Levels

### Available Levels

- **TRACE**: Detailed information for debugging
- **DEBUG**: Detailed information for debugging
- **INFO**: General information about server operations
- **SUCCESS**: Success messages
- **WARNING**: Warning messages that don't stop execution
- **ERROR**: Error messages that affect functionality
- **CRITICAL**: Critical errors that may cause server failure

### Setting Log Level

```
export HAYHOOKS_LOG_LEVEL=DEBUG
hayhooks run
```

Or in your `.env` file:

```
HAYHOOKS_LOG_LEVEL=DEBUG
```

Or inline:

```
HAYHOOKS_LOG_LEVEL=DEBUG hayhooks run
```

Legacy alias

The `LOG` environment variable is still supported as a fallback. When both are set, `HAYHOOKS_LOG_LEVEL` takes precedence.

## Log Configuration

### Environment Variables

#### HAYHOOKS_LOG_LEVEL

- **Default**: `INFO`
- **Alias**: `LOG` (legacy, lower priority)
- **Description**: Minimum log level to display (consumed by Loguru)
- **Options**: `TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, `CRITICAL`

#### HAYHOOKS_LOG_FORMAT

- **Default**: `default`
- **Description**: Controls the amount of metadata shown in log lines
- **Options**:
- `default` — Timestamp, level, and message
- `verbose` — Also includes module name, function, and line number

```
# Default format
2026-03-27 10:47:00 | INFO     | Pipelines dir set to: '/app/pipelines'

# Verbose format
2026-03-27 10:47:00 | INFO     | hayhooks.server.app:deploy_pipelines:188 | Pipelines dir set to: '/app/pipelines'
```

#### HAYHOOKS_INTERCEPTED_LOGGERS

- **Default**: `["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]`
- **Description**: Stdlib loggers routed through Loguru. Only the listed loggers are intercepted; all others keep their default behaviour.

## Trace Correlation

When tracing is enabled, Hayhooks binds `trace_id` and `span_id` to log context for instrumented operations (deploy, undeploy, run, OpenAI-compatible execution, MCP tool calls).

- `request_id` is still emitted as before.
- `trace_id`/`span_id` are normalized to hexadecimal identifiers for easier cross-linking in tracing backends.

### Custom Log Format

> For fully custom formatting or additional sinks (files, external services), configure them in your host app via `log.add(...)`.

## File Logging

### Basic File Logging

> Configure file sinks in your host app using `log.add(...)`.

### Rotating File Logs

If you embed Hayhooks programmatically and want custom logging, set up logging in your host app and direct Hayhooks logs there.

## Pipeline Logging

The `log` object in Hayhooks is a [Loguru](https://loguru.readthedocs.io/) logger instance. You can use all Loguru features and capabilities in your pipeline code.

### Basic Usage

```
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        log.info("Setting up pipeline")
        # ... setup code

    def run_api(self, query: str) -> str:
        log.debug("Processing query: {query}")
        try:
            result = self.pipeline.run({"prompt": {"query": query}})
            log.info("Pipeline execution completed successfully")
            return result["llm"]["replies"][0]
        except Exception as e:
            log.error("Pipeline execution failed: {}", e)
            raise
```

### Execution Time Logging

```
import time
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, query: str) -> str:
        start_time = time.time()

        result = self.pipeline.run({"prompt": {"query": query}})

        execution_time = time.time() - start_time
        log.info("Pipeline executed in {} seconds", round(execution_time, 2))

        return result["llm"]["replies"][0]
```

For more advanced logging patterns (structured logging, custom sinks, formatting, etc.), refer to the [Loguru documentation](https://loguru.readthedocs.io/).

## Next Steps

- [API Reference](https://deepset-ai.github.io/hayhooks/reference/api-reference/index.md) - Complete API documentation
- [Environment Variables](https://deepset-ai.github.io/hayhooks/reference/environment-variables/index.md) - Configuration options
