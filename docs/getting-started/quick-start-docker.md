# Quick Start with Docker Compose

To quickly get started with Hayhooks, we provide a ready-to-use Docker Compose 🐳 setup with pre-configured integration with [open-webui](https://openwebui.com/).

It's available in the [Hayhooks + Open WebUI Docker Compose repository](https://github.com/deepset-ai/hayhooks-open-webui-docker-compose).

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/deepset-ai/hayhooks-open-webui-docker-compose.git
cd hayhooks-open-webui-docker-compose
```

### 2. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit the `.env` file to configure your settings:

```bash
# Hayhooks Configuration
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_MCP_PORT=1417
HAYHOOKS_PIPELINES_DIR=/app/pipelines

# OpenWebUI Configuration
OPENWEBUI_PORT=3000
```

### 3. Start the Services

```bash
docker-compose up -d
```

This will start:
- Hayhooks server on port 1416
- Hayhooks MCP server on port 1417
- OpenWebUI on port 3000

### 4. Access the Services

- **Hayhooks API**: http://localhost:1416
- **OpenWebUI**: http://localhost:3000
- **Hayhooks API Documentation**: http://localhost:1416/docs

### 5. Deploy Example Pipelines

The Docker Compose setup includes example pipelines that can be deployed automatically:

```bash
# Deploy a sample pipeline
docker-compose exec hayhooks hayhooks pipeline deploy-files -n chat_with_website /app/examples/pipeline_wrappers/chat_with_website_streaming
```

## Configuration Options

### Environment Variables

The following environment variables can be configured in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `HAYHOOKS_HOST` | Host to bind to | `0.0.0.0` |
| `HAYHOOKS_PORT` | Port for Hayhooks API | `1416` |
| `HAYHOOKS_MCP_PORT` | Port for Hayhooks MCP server | `1417` |
| `HAYHOOKS_PIPELINES_DIR` | Directory for pipeline definitions | `/app/pipelines` |
| `OPENWEBUI_PORT` | Port for OpenWebUI | `3000` |

### Volume Mounts

The Docker Compose setup includes the following volume mounts:

- **Pipeline Directory**: `/app/pipelines` - For storing pipeline definitions
- **Custom Code**: `/app/custom_code` - For shared Python modules
- **Data**: `/app/data` - For persistent data storage

## Integrating with OpenWebUI

The Docker Compose setup comes pre-configured to integrate Hayhooks with OpenWebUI:

### 1. Configure OpenWebUI

1. Access OpenWebUI at http://localhost:3000
2. Go to **Settings → Connections**
3. Add a new connection with:
   - **API Base URL**: `http://hayhooks:1416/v1`
   - **API Key**: `any-value` (not used by Hayhooks)

### 2. Deploy a Pipeline

Deploy a pipeline that supports chat completion:

```bash
docker-compose exec hayhooks hayhooks pipeline deploy-files -n chat_agent /app/examples/pipeline_wrappers/open_webui_agent_events
```

### 3. Test the Integration

1. In OpenWebUI, select the Hayhooks backend
2. Start a conversation with your deployed pipeline
3. The pipeline will respond through the OpenWebUI interface

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Ensure ports 1416, 1417, and 3000 are available
2. **Permission Issues**: Make sure Docker has proper permissions
3. **Network Issues**: Check that containers can communicate with each other

### Logs

Check logs for troubleshooting:

```bash
# Hayhooks logs
docker-compose logs -f hayhooks

# OpenWebUI logs
docker-compose logs -f openwebui
```

### Cleanup

To stop and remove all containers:

```bash
docker-compose down -v
```

## Next Steps

- [Configuration](../getting-started/configuration.md) - Learn about advanced configuration
- [Examples](../examples/overview.md) - Explore more examples
- [OpenWebUI Integration](../features/openwebui-integration.md) - Deep dive into OpenWebUI integration