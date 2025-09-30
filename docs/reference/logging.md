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

```bash
export LOG=debug # or LOG=DEBUG
hayhooks run
```

Or in your `.env` file:

```ini
LOG=info # or LOG=INFO
```

Or inline:

```bash
LOG=debug hayhooks run
```

## Log Configuration

### Environment Variables

#### LOG

- **Default**: `info`
- **Description**: Minimum log level to display (consumed by Loguru)
- **Options**: `debug`, `info`, `warning`, `error`

> Note: Hayhooks does not expose `HAYHOOKS_LOG_FORMAT` or `HAYHOOKS_LOG_FILE` env vars; formatting/handlers are configured internally in the code.

### Custom Log Format

> If you need custom formatting, handle it in your host app via Loguru sinks.

## File Logging

### Basic File Logging

> Configure file sinks in your host app using `log.add(...)`.

### Rotating File Logs

If you embed Hayhooks programmatically and want custom logging, set up logging in your host app and direct Hayhooks logs there.

## Pipeline Logging

### Logging in Pipeline Wrappers

```python
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        log.info("Setting up pipeline")
        # ... setup code

    def run_api(self, query: str) -> str:
        log.debug(f"Processing query: {query}")
        try:
            result = self.pipeline.run({"prompt": {"query": query}})
            log.info("Pipeline execution completed successfully")
            return result["llm"]["replies"][0]
        except Exception as e:
            log.error(f"Pipeline execution failed: {e}")
            raise
```

### Structured Logging

```python
import json
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, query: str) -> str:
        log.info({
            "event": "pipeline_start",
            "query": query,
            "timestamp": "2024-01-01T00:00:00Z"
        })

        result = self.pipeline.run({"prompt": {"query": query}})

        log.info({
            "event": "pipeline_complete",
            "query_length": len(query),
            "result_length": len(result["llm"]["replies"][0])
        })

        return result["llm"]["replies"][0]
```

### Log Aggregation

For production deployments, consider shipping stdout to your logging stack (e.g., container runtime, sidecar, or agent).

## Performance Logging

### Execution Time Logging

```python
import time
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, query: str) -> str:
        start_time = time.time()

        result = self.pipeline.run({"prompt": {"query": query}})

        execution_time = time.time() - start_time
        log.info(f"Pipeline executed in {execution_time:.2f} seconds")

        return result["llm"]["replies"][0]
```

### Request Logging

```python
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, query: str) -> str:
        log.info({
            "event": "request_start",
            "pipeline": self.__class__.__name__,
            "query_length": len(query),
            "timestamp": time.time()
        })

        # ... execute pipeline

        log.info({
            "event": "request_complete",
            "pipeline": self.__class__.__name__,
            "status": "success",
            "duration": execution_time
        })
```

## Next Steps

- [API Reference](api-reference.md) - Complete API documentation
- [Environment Variables](environment-variables.md) - Configuration options
