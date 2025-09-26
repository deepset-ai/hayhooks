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

```bash
pip install hayhooks
```

## Global Commands

### Help

Get help for any command:

```bash
# Show main help
hayhooks --help

# Show help for specific command
hayhooks run --help
hayhooks pipeline --help
```

### Version

Check the installed version:

```bash
hayhooks --version
```

## Server Commands

### run (HTTP vs CLI example)

Start the Hayhooks server:

```bash
# Basic server start
hayhooks run

# With custom host and port
hayhooks run --host 0.0.0.0 --port 1416

# With multiple workers
hayhooks run --workers 4

# With custom pipelines directory
hayhooks run --pipelines-dir ./my_pipelines

# With additional Python path
hayhooks run --additional-python-path ./custom_code

# Reload on changes (development)
hayhooks run --reload
```

#### Options for `mcp run`

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | | Host to bind to | `127.0.0.1` |
| `--port` | | Port to listen on | `1416` |
| `--workers` | | Number of worker processes | `1` |
| `--pipelines-dir` | | Directory for pipeline definitions | `./pipelines` |
| `--additional-python-path` | | Additional Python path | `None` |
| `--root-path` | | Root path for API | `/` |

### mcp run

Start the MCP server:

```bash
# Start MCP server
hayhooks mcp run

# With custom host and port
hayhooks mcp run --host 0.0.0.0 --port 1417
```

#### Options for `pipeline deploy-files`

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | `-h` | MCP server host | `127.0.0.1` |
| `--port` | `-p` | MCP server port | `1417` |
| `--pipelines-dir` | `-d` | Directory for pipeline definitions | `./pipelines` |
| `--additional-python-path` | | Additional Python path | `None` |
| `--json-response` | `-j` | Use JSON responses instead of SSE | `false` |
| `--debug` | `-d` | Return tracebacks on errors | `false` |

## Pipeline Management Commands

### pipeline deploy-files

Deploy a pipeline from wrapper files:

```bash
# Basic deployment
hayhooks pipeline deploy-files -n my_pipeline ./path/to/pipeline

# With custom name and description
hayhooks pipeline deploy-files -n my_pipeline --description "My pipeline" ./path/to/pipeline

# Overwrite existing pipeline
hayhooks pipeline deploy-files -n my_pipeline --overwrite ./path/to/pipeline

# Skip saving files to server
hayhooks pipeline deploy-files -n my_pipeline --skip-saving-files ./path/to/pipeline

# Skip MCP tool registration
hayhooks pipeline deploy-files -n my_pipeline --skip-mcp ./path/to/pipeline
```

#### Options for `pipeline deploy-yaml`

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--name` | `-n` | Pipeline name | Required |
| `--description` | | Human-readable description | Pipeline name |
| `--overwrite` | `-o` | Overwrite existing pipeline | `false` |
| `--skip-saving-files` | | Don't save files to server | `false` |
| `--skip-mcp` | | Skip MCP tool registration | `false` |

### pipeline deploy-yaml

Deploy a pipeline from YAML definition:

```bash
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

#### Options for `pipeline undeploy`

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--name` | `-n` | Pipeline name | YAML file stem |
| `--description` | | Human-readable description | Pipeline name |
| `--overwrite` | `-o` | Overwrite existing pipeline | `false` |
| `--skip-mcp` | | Skip MCP tool registration | `false` |
| `--save-file` | | Save YAML to server | `true` |
| `--no-save-file` | | Don't save YAML to server | `false` |

### pipeline undeploy

Undeploy a pipeline:

```bash
# Undeploy by name
hayhooks pipeline undeploy my_pipeline

# Force undeploy (ignore errors)
hayhooks pipeline undeploy my_pipeline --force
```

#### Options for `pipeline run`

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--force` | | Force undeploy | `false` |

### pipeline run

Run a deployed pipeline:

```bash
# Run with JSON parameters
hayhooks pipeline run my_pipeline --param 'query="What is Haystack?"'

# Run with multiple parameters
hayhooks pipeline run my_pipeline --param 'query="What is Haystack?"' --param 'temperature=0.7'

# Upload files
hayhooks pipeline run my_pipeline --file document.pdf --param 'query="Summarize this"'

# Upload directory
hayhooks pipeline run my_pipeline --dir ./documents --param 'query="Analyze all documents"'

# Upload multiple files
hayhooks pipeline run my_pipeline --file doc1.pdf --file doc2.txt --param 'query="Compare documents"'
```

#### Options for `status`

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--file` | | Upload single file | None |
| `--dir` | | Upload directory | None |
| `--param` | | Pass parameters as JSON | None |

## Status and Monitoring Commands

### status

Check server and pipeline status:

```bash
# Check server status
hayhooks status

# Check status with verbose output
hayhooks status --verbose

# Check status with JSON output
hayhooks status --json
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--verbose` | `-v` | Show detailed information | `false` |
| `--json` | | Output in JSON format | `false` |

> Note: There is no `hayhooks health` command in the CLI. Use `hayhooks status` or call HTTP endpoints directly.

## Notes

Configuration is managed via environment variables and CLI flags. See [Configuration](../getting-started/configuration.md).

Development flows (testing, linting) are not exposed as CLI commands.

Use your process manager or container logs to view logs; Hayhooks uses standard output.

Advanced export/import/migrate commands are not provided by the Hayhooks CLI at this time.

## HTTP API Commands

All CLI commands have corresponding HTTP API endpoints:

### deploy-files

```bash
# CLI
hayhooks pipeline deploy-files -n my_pipeline ./path/to/pipeline

# HTTP API
curl -X POST http://localhost:1416/deploy_files \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "my_pipeline",
    "description": "My pipeline",
    "files": [...],
    "overwrite": false
  }'
```

### deploy-yaml

```bash
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

```bash
# CLI
hayhooks pipeline run my_pipeline --param 'query="What is Haystack?"'

# HTTP API
curl -X POST http://localhost:1416/my_pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is Haystack?"}'
```

## Configuration Files

### .env File

Create a `.env` file for configuration:

```bash
# .env
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_MCP_PORT=1417
HAYHOOKS_PIPELINES_DIR=./pipelines
LOG_LEVEL=INFO
```

### pipelines.yaml

Define pipelines in YAML:

```yaml
pipelines:
  - name: chat_pipeline
    type: files
    path: ./pipelines/chat_pipeline
    description: "Chat with AI"

  - name: rag_pipeline
    type: yaml
    path: ./pipelines/rag_pipeline.yml
    description: "RAG system"
```

## Error Handling

### Common Errors

1. **Server already running**

   ```bash
   # Check if server is running
   hayhooks status

   # Kill existing process
   pkill -f "hayhooks run"
   ```

2. **Pipeline deployment failed**

   ```bash
   # Check server logs with your process manager or container runtime

   # Enable debug logging
   hayhooks env set LOG_LEVEL DEBUG
   ```

3. **Permission denied**

   ```bash
   # Check file permissions
   ls -la ./path/to/pipeline

   # Fix permissions if needed
   chmod +x ./path/to/pipeline/pipeline_wrapper.py
   ```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Set debug logging
export LOG_LEVEL=DEBUG

# Start server with debug output
hayhooks run --verbose

# Check detailed status
hayhooks status --verbose
```

## Best Practices

### 1. Pipeline Management

- Use meaningful pipeline names
- Include descriptions for better documentation
- Use version control for pipeline definitions
- Test pipelines before deployment

### 2. Server Configuration

- Use environment variables for configuration
- Set appropriate worker counts for your workload
- Monitor server health and logs
- Use proper security settings in production

### 3. CLI Usage

- Use `--help` for command documentation
- Test commands in development environment first
- Use `--dry-run` when available for testing
- Keep CLI scripts in version control

### 4. Error Handling

- Check return codes and error messages
- Use `--verbose` flag for debugging
- Monitor logs for issues
- Implement proper error handling in scripts

## Examples

### Basic Workflow

```bash
# 1. Start server
hayhooks run --port 1416

# 2. In another terminal, deploy pipeline
hayhooks pipeline deploy-files -n chat_pipeline ./pipelines/chat

# 3. Check status
hayhooks status

# 4. Run pipeline
hayhooks pipeline run chat_pipeline --param 'query="Hello!"'

# 5. Check logs
# Use your process manager or container logs
```

### Production Deployment

```bash
# 1. Set environment variables
export HAYHOOKS_HOST=0.0.0.0
export HAYHOOKS_PORT=1416
export HAYHOOKS_WORKERS=4
export LOG_LEVEL=INFO

# 2. Start server with multiple workers
hayhooks run --workers 4

# 3. Deploy pipelines
hayhooks pipeline deploy-files -n production_pipeline ./pipelines/production

# 4. Monitor health
hayhooks health --detailed
```

## Next Steps

- [Configuration](../getting-started/configuration.md) - Configuration options
- [Deployment Guidelines](../deployment/deployment-guidelines.md) - Production deployment
- [Examples](../examples/overview.md) - Working examples
