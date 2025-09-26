# Configuration

Hayhooks can be configured through environment variables, command-line arguments, or configuration files.

## Configuration Methods

### Environment Variables

You can configure Hayhooks by setting environment variables:

```bash
# Set environment variables
export HAYHOOKS_HOST=0.0.0.0
export HAYHOOKS_PORT=1416
export HAYHOOKS_PIPELINES_DIR=./pipelines

# Start Hayhooks
hayhooks run
```

### .env File

Create a `.env` file in your project root:

```bash
# .env
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_MCP_PORT=1417
HAYHOOKS_PIPELINES_DIR=./pipelines
LOG_LEVEL=INFO
```

### Command Line Arguments

Pass configuration options to the `hayhooks run` command:

```bash
hayhooks run --host 0.0.0.0 --port 1416 --pipelines-dir ./pipelines
```

## Configuration Options

### Server Configuration

| Variable | Command Line | Description | Default |
|----------|--------------|-------------|---------|
| `HAYHOOKS_HOST` | `--host` | Host to bind to | `127.0.0.1` |
| `HAYHOOKS_PORT` | `--port` | Port to listen on | `1416` |
| `HAYHOOKS_MCP_HOST` | `--mcp-host` | MCP server host | `127.0.0.1` |
| `HAYHOOKS_MCP_PORT` | `--mcp-port` | MCP server port | `1417` |
| `HAYHOOKS_ROOT_PATH` | `--root-path` | Root path for the API | `/` |
| `HAYHOOKS_PIPELINES_DIR` | `--pipelines-dir` | Directory for pipeline definitions | `./pipelines` |
| `HAYHOOKS_ADDITIONAL_PYTHON_PATH` | `--additional-python-path` | Additional Python path | `None` |

### SSL/TLS Configuration

| Variable | Command Line | Description | Default |
|----------|--------------|-------------|---------|
| `HAYHOOKS_USE_HTTPS` | `--use-https` | Use HTTPS for CLI communication | `false` |
| `HAYHOOKS_DISABLE_SSL` | `--disable-ssl` | Disable SSL verification | `false` |

### CORS Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HAYHOOKS_CORS_ALLOW_ORIGINS` | List of allowed origins | `["*"]` |
| `HAYHOOKS_CORS_ALLOW_METHODS` | List of allowed HTTP methods | `["*"]` |
| `HAYHOOKS_CORS_ALLOW_HEADERS` | List of allowed headers | `["*"]` |
| `HAYHOOKS_CORS_ALLOW_CREDENTIALS` | Allow credentials | `false` |
| `HAYHOOKS_CORS_ALLOW_ORIGIN_REGEX` | Regex pattern for allowed origins | `null` |
| `HAYHOOKS_CORS_EXPOSE_HEADERS` | Headers to expose in response | `[]` |
| `HAYHOOKS_CORS_MAX_AGE` | Maximum age for CORS preflight responses | `600` |

### Logging Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `HAYHOOKS_SHOW_TRACEBACKS` | Show tracebacks on errors | `false` |

## Environment Variables Reference

### Core Variables

```bash
# Server Configuration
HAYHOOKS_HOST=0.0.0.0                    # Host to bind to
HAYHOOKS_PORT=1416                      # Port to listen on
HAYHOOKS_MCP_HOST=127.0.0.1             # MCP server host
HAYHOOKS_MCP_PORT=1417                  # MCP server port
HAYHOOKS_ROOT_PATH=/                    # Root path for API
HAYHOOKS_PIPELINES_DIR=./pipelines      # Pipeline definitions directory
HAYHOOKS_ADDITIONAL_PYTHON_PATH=./custom_code  # Additional Python path

# SSL/TLS
HAYHOOKS_USE_HTTPS=false                # Use HTTPS for CLI
HAYHOOKS_DISABLE_SSL=false              # Disable SSL verification

# CORS
HAYHOOKS_CORS_ALLOW_ORIGINS=["*"]        # Allowed origins
HAYHOOKS_CORS_ALLOW_METHODS=["*"]        # Allowed methods
HAYHOOKS_CORS_ALLOW_HEADERS=["*"]        # Allowed headers
HAYHOOKS_CORS_ALLOW_CREDENTIALS=false    # Allow credentials
HAYHOOKS_CORS_ALLOW_ORIGIN_REGEX=null   # Origin regex pattern
HAYHOOKS_CORS_EXPOSE_HEADERS=[]          # Exposed headers
HAYHOOKS_CORS_MAX_AGE=600               # CORS preflight max age

# Logging
LOG=INFO                               # Log level
HAYHOOKS_SHOW_TRACEBACKS=false          # Show error tracebacks
```

## Example Configurations

### Development Configuration

```bash
# .env.development
HAYHOOKS_HOST=127.0.0.1
HAYHOOKS_PORT=1416
HAYHOOKS_MCP_PORT=1417
HAYHOOKS_PIPELINES_DIR=./pipelines
HAYHOOKS_ADDITIONAL_PYTHON_PATH=./custom_code
LOG_LEVEL=DEBUG
HAYHOOKS_SHOW_TRACEBACKS=true
```

### Production Configuration

```bash
# .env.production
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_MCP_PORT=1417
HAYHOOKS_PIPELINES_DIR=/app/pipelines
HAYHOOKS_ADDITIONAL_PYTHON_PATH=/app/custom_code
LOG_LEVEL=INFO
HAYHOOKS_SHOW_TRACEBACKS=false
HAYHOOKS_USE_HTTPS=true
```

### Docker Configuration

```bash
# .env.docker
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_MCP_PORT=1417
HAYHOOKS_PIPELINES_DIR=/app/pipelines
HAYHOOKS_ADDITIONAL_PYTHON_PATH=/app/custom_code
LOG_LEVEL=INFO
```

## Validation

Hayhooks validates configuration at startup. If invalid values are provided, Hayhooks will:

1. Log the error
2. Use default values
3. Continue startup if possible
4. Exit if critical configuration is invalid

## Next Steps

After configuring Hayhooks:

- [Quick Start](quick-start.md) - Get started with basic usage
- [Pipeline Deployment](../concepts/pipeline-deployment.md) - Learn how to deploy pipelines
- [Features](../features/openai-compatibility.md) - Explore advanced features
