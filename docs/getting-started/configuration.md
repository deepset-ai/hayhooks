# Configuration

Hayhooks can be configured through environment variables, command-line arguments, or `.env` files.

## Configuration Methods

### Environment Variables

Set environment variables before starting Hayhooks:

```bash
export HAYHOOKS_HOST=0.0.0.0
export HAYHOOKS_PORT=1416
hayhooks run
```

### .env File

Create a `.env` file in your project root:

```bash
# .env
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_PIPELINES_DIR=./pipelines
LOG=INFO
```

### Command Line Arguments

Pass options directly to `hayhooks run`:

```bash
hayhooks run --host 0.0.0.0 --port 1416 --pipelines-dir ./pipelines
```

## Common Configuration Options

The most frequently used options:

- `HAYHOOKS_HOST` - Host to bind to (default: `localhost`)
- `HAYHOOKS_PORT` - Port to listen on (default: `1416`)
- `HAYHOOKS_PIPELINES_DIR` - Pipeline directory for auto-deployment (default: `./pipelines`)
- `LOG` - Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

For the complete list of all environment variables and detailed descriptions, see the [Environment Variables Reference](../reference/environment-variables.md).

## Example Configurations

### Development

```bash
# .env.development
HAYHOOKS_HOST=localhost
HAYHOOKS_PORT=1416
LOG=DEBUG
HAYHOOKS_SHOW_TRACEBACKS=true
```

### Production

```bash
# .env.production
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
LOG=INFO
HAYHOOKS_SHOW_TRACEBACKS=false
```

## Next Steps

- [Quick Start](quick-start.md) - Get started with basic usage
- [Pipeline Deployment](../concepts/pipeline-deployment.md) - Learn how to deploy pipelines
- [Environment Variables Reference](../reference/environment-variables.md) - Complete configuration reference
