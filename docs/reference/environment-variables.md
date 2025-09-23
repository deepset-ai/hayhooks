# Environment Variables

Hayhooks can be configured using environment variables for deployment flexibility and security.

## Configuration Variables

### Server Configuration

#### HAYHOOKS_HOST
- **Default**: `127.0.0.1`
- **Description**: Host address to bind the server to
- **Example**: `HAYHOOKS_HOST=0.0.0.0`

#### HAYHOOKS_PORT
- **Default**: `1416`
- **Description**: Port number for the main Hayhooks server
- **Example**: `HAYHOOKS_PORT=8080`

> Note: Worker count is configured via CLI `hayhooks run --workers N`, not via env var.

### Pipeline Configuration

#### HAYHOOKS_PIPELINES_DIR
- **Default**: `./pipelines`
- **Description**: Directory to store deployed pipeline files
- **Example**: `HAYHOOKS_PIPELINES_DIR=/app/pipelines`

> Saving/overwriting is controlled per-deploy via CLI/API flags, not global env vars.

### MCP Server Configuration

#### HAYHOOKS_MCP_HOST
- **Default**: `127.0.0.1`
- **Description**: Host address for the MCP server
- **Example**: `HAYHOOKS_MCP_HOST=0.0.0.0`

#### HAYHOOKS_MCP_PORT
- **Default**: `1417`
- **Description**: Port number for the MCP server
- **Example**: `HAYHOOKS_MCP_PORT=1418`

> Start the MCP server via `hayhooks mcp run`; no global enable flag.

### Logging Configuration

#### LOG
- **Default**: `INFO`
- **Description**: Logging level (DEBUG, INFO, WARNING, ERROR)
- **Example**: `LOG=DEBUG`

#### HAYHOOKS_LOG_FORMAT
- **Default**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Description**: Log message format
- **Example**: `HAYHOOKS_LOG_FORMAT=%(levelname)s:%(name)s:%(message)s`

#### HAYHOOKS_LOG_FILE
- **Default**: `None` (logs to console)
- **Description**: File path to write logs to
- **Example**: `HAYHOOKS_LOG_FILE=/var/log/hayhooks.log`

### Security Configuration

> CORS is always configured via specific allow/expose settings.

#### HAYHOOKS_CORS_ORIGINS
- **Default**: `*`
- **Description**: Comma-separated list of allowed CORS origins
- **Example**: `HAYHOOKS_CORS_ORIGINS=http://localhost:3000,https://yourdomain.com`

#### HAYHOOKS_API_KEY
- **Default**: `None`
- **Description**: API key for authentication (if implemented)
- **Example**: `HAYHOOKS_API_KEY=your-secret-key`

### OpenAI Configuration

#### HAYHOOKS_OPENAI_BASE_URL
- **Default**: `None`
- **Description**: Custom OpenAI API base URL
- **Example**: `HAYHOOKS_OPENAI_BASE_URL=https://api.openai.com/v1`

#### HAYHOOKS_OPENAI_API_KEY
- **Default**: `None`
- **Description**: OpenAI API key for pipelines that use it
- **Example**: `HAYHOOKS_OPENAI_API_KEY=sk-...`

## Usage Examples

### Docker Environment

```bash
docker run -d \
  -e HAYHOOKS_HOST=0.0.0.0 \
  -e HAYHOOKS_PORT=8080 \
  -e HAYHOOKS_WORKERS=4 \
  -e HAYHOOKS_LOG_LEVEL=INFO \
  -p 8080:8080 \
  deepset/hayhooks:latest
```

### Development Environment

```bash
export HAYHOOKS_HOST=127.0.0.1
export HAYHOOKS_PORT=1416
export HAYHOOKS_LOG_LEVEL=DEBUG
export HAYHOOKS_MCP_ENABLED=true

hayhooks run
```

### Production Environment

```bash
export HAYHOOKS_HOST=0.0.0.0
export HAYHOOKS_PORT=80
export HAYHOOKS_WORKERS=4
export HAYHOOKS_TIMEOUT=60
export HAYHOOKS_LOG_LEVEL=INFO
export HAYHOOKS_LOG_FILE=/var/log/hayhooks.log
export HAYHOOKS_CORS_ORIGINS=https://yourdomain.com

hayhooks run
```

### MCP Server Environment

```bash
export HAYHOOKS_MCP_ENABLED=true
export HAYHOOKS_MCP_HOST=0.0.0.0
export HAYHOOKS_MCP_PORT=1417

hayhooks mcp run
```

## Environment File (.env)

You can use a `.env` file in your project root:

```env
# Server Configuration
HAYHOOKS_HOST=127.0.0.1
HAYHOOKS_PORT=1416
HAYHOOKS_WORKERS=1
HAYHOOKS_TIMEOUT=30

# Pipeline Configuration
HAYHOOKS_PIPELINES_DIR=./pipelines
HAYHOOKS_SAVE_PIPELINES=true
HAYHOOKS_OVERWRITE_PIPELINES=false

# MCP Configuration
HAYHOOKS_MCP_ENABLED=false
HAYHOOKS_MCP_HOST=127.0.0.1
HAYHOOKS_MCP_PORT=1417

# Logging
HAYHOOKS_LOG_LEVEL=INFO
HAYHOOKS_LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# CORS
HAYHOOKS_CORS_ENABLED=true
HAYHOOKS_CORS_ORIGINS=*
```

## Configuration Priority

Environment variables are loaded in this order:

1. Default values
2. `.env` file (if present)
3. System environment variables

## Security Considerations

### Sensitive Variables

- Never commit API keys or secrets to version control
- Use secret management tools in production
- Consider using `.env.example` for template configuration

### Production Security

```bash
# Secure production configuration
export HAYHOOKS_CORS_ORIGINS=https://yourdomain.com
export HAYHOOKS_API_KEY=${SECURE_API_KEY}
export HAYHOOKS_LOG_LEVEL=WARNING
export HAYHOOKS_TIMEOUT=30
```

## Debugging

### Check Current Configuration

```bash
# Start with debug logging
export HAYHOOKS_LOG_LEVEL=DEBUG
hayhooks run

# Check environment variables
env | grep HAYHOOKS
```

### Common Issues

1. **Port Already in Use**
   ```bash
   export HAYHOOKS_PORT=1417
   ```

2. **Permission Denied**
   ```bash
   export HAYHOOKS_PIPELINES_DIR=/tmp/pipelines
   ```

3. **CORS Issues**
   ```bash
   export HAYHOOKS_CORS_ORIGINS=http://localhost:3000
   ```

## Next Steps

- [API Reference](api-reference.md) - Complete API documentation
- [Logging](logging.md) - Logging configuration and usage
- [Deployment Guidelines](../deployment/deployment-guidelines.md) - Production deployment tips
