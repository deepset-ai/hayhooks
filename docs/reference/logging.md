# Logging

Hayhooks provides comprehensive logging capabilities for monitoring, debugging, and auditing pipeline execution and server operations.

## Log Levels

### Available Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about server operations
- **WARNING**: Warning messages that don't stop execution
- **ERROR**: Error messages that affect functionality
- **CRITICAL**: Critical errors that may cause server failure

### Setting Log Level

```bash
export LOG=DEBUG
hayhooks run
```

Or in your `.env` file:
```env
LOG=INFO
```

## Log Configuration

### Environment Variables

#### LOG
- **Default**: `INFO`
- **Description**: Minimum log level to display
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`

#### HAYHOOKS_LOG_FORMAT
- **Default**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Description**: Format string for log messages

#### HAYHOOKS_LOG_FILE
- **Default**: `None` (console output)
- **Description**: File path to write logs to

### Custom Log Format

```bash
export HAYHOOKS_LOG_FORMAT="%(levelname)s [%(name)s] %(message)s"
```

Available format variables:
- `%(asctime)s`: Timestamp
- `%(name)s`: Logger name
- `%(levelname)s`: Log level
- `%(message)s`: Log message
- `%(pathname)s`: Source file path
- `%(lineno)d`: Source line number

## File Logging

### Basic File Logging

```bash
export HAYHOOKS_LOG_FILE=/var/log/hayhooks.log
hayhooks run
```

### Rotating File Logs

For production use, consider using log rotation:

```python
# In your custom logging configuration
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'hayhooks.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

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

## Log Categories

### Server Logs

```
INFO - hayhooks.server - Starting Hayhooks server on 127.0.0.1:1416
INFO - hayhooks.server - Pipeline 'chat_pipeline' deployed successfully
ERROR - hayhooks.server - Failed to deploy pipeline 'invalid_pipeline'
```

### Pipeline Logs

```
DEBUG - hayhooks.pipeline - Running pipeline with input: {'query': 'Hello'}
INFO - hayhooks.pipeline - Pipeline execution completed in 2.3 seconds
ERROR - hayhooks.pipeline - Pipeline failed with error: ModuleNotFoundError
```

### Request Logs

```
INFO - hayhooks.request - POST /chat_pipeline/run - 200 - 0.5s
INFO - hayhooks.request - GET /status - 200 - 0.1s
WARNING - hayhooks.request - POST /invalid_pipeline/run - 404 - 0.0s
```

## MCP Server Logging

### Enable MCP Logging
```bash
export LOG=DEBUG
hayhooks mcp run
```

### MCP Log Examples

```
INFO - hayhooks.mcp - MCP server started on 127.0.0.1:1417
DEBUG - hayhooks.mcp - Received MCP request: get_all_pipeline_statuses
INFO - hayhooks.mcp - MCP tool execution completed: deploy_pipeline
```

## Custom Logging Configuration

### Python Configuration

```python
import logging
from hayhooks import configure_logging

# Custom logging configuration
configure_logging(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("hayhooks.log"),
        logging.StreamHandler()
    ]
)
```

### YAML Configuration

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: standard
    filename: hayhooks.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  hayhooks:
    level: DEBUG
    handlers: [console, file]
    propagate: false
```

## Log Monitoring

### Real-time Monitoring

```bash
# Tail log file
tail -f /var/log/hayhooks.log

# Filter by log level
tail -f /var/log/hayhooks.log | grep ERROR

# Filter by pipeline name
tail -f /var/log/hayhooks.log | grep "chat_pipeline"
```

### Log Aggregation

For production deployments, consider:

```bash
# Send logs to external service
export HAYHOOKS_LOG_FORMAT="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
hayhooks run | logger -t hayhooks
```

## Debug Logging

### Enable Debug Mode
```bash
export LOG=DEBUG
export HAYHOOKS_LOG_FILE=debug.log
hayhooks run
```

### Debug Information

Debug logs include:
- Detailed request/response information
- Pipeline execution traces
- Internal server operations
- Performance metrics

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

## Security Logging

### Audit Logging

```python
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, query: str) -> str:
        # Log access attempt
        log.info({
            "event": "pipeline_access",
            "pipeline": self.__class__.__name__,
            "user_ip": "client_ip_address",  # Get from request
            "timestamp": time.time()
        })

        result = self.pipeline.run({"prompt": {"query": query}})

        # Log successful execution
        log.info({
            "event": "pipeline_executed",
            "pipeline": self.__class__.__name__,
            "success": True,
            "timestamp": time.time()
        })

        return result["llm"]["replies"][0]
```

## Troubleshooting

### Common Logging Issues

1. **No Log Output**
   ```bash
   export LOG=DEBUG
   export HAYHOOKS_LOG_FILE=debug.log
   ```

2. **Permission Denied**
   ```bash
   export HAYHOOKS_LOG_FILE=/tmp/hayhooks.log
   ```

3. **Disk Space Issues**
   ```bash
   # Use log rotation
   export HAYHOOKS_LOG_FILE=/var/log/hayhooks.log
   ```

### Log Analysis

```bash
# Count errors in last hour
grep ERROR hayhooks.log | grep "$(date '+%Y-%m-%d %H:')" | wc -l

# Find slow pipeline executions
grep "Pipeline executed in" hayhooks.log | awk '{if ($NF > 5) print $0}'

# Monitor deployment activity
grep "deployed successfully" hayhooks.log | tail -10
```

## Best Practices

### Development

```bash
export LOG=DEBUG
export HAYHOOKS_LOG_FORMAT="%(levelname)s [%(name)s:%(lineno)d] %(message)s"
```

### Production

```bash
export LOG=INFO
export HAYHOOKS_LOG_FILE=/var/log/hayhooks.log
export HAYHOOKS_LOG_FORMAT="%(asctime)s [%(levelname)s] %(message)s"
```

### Security

```bash
export LOG=WARNING
export HAYHOOKS_LOG_FILE=/var/log/hayhooks.log
export HAYHOOKS_LOG_FORMAT="%(asctime)s [%(levelname)s] %(message)s"
```

## Next Steps

- [API Reference](api-reference.md) - Complete API documentation
- [Environment Variables](environment-variables.md) - Configuration options
- [Deployment Guidelines](../deployment/deployment-guidelines.md) - Production deployment
