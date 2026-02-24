# Quick Start with Docker Compose

To quickly get started with Hayhooks, we provide a ready-to-use Docker Compose 🐳 setup with pre-configured integration with [Open WebUI](https://openwebui.com/).

It's available in the [Hayhooks + Open WebUI Docker Compose repository](https://github.com/deepset-ai/hayhooks-open-webui-docker-compose).

## Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/deepset-ai/hayhooks-open-webui-docker-compose.git
cd hayhooks-open-webui-docker-compose
```

### 2. Start the Services

```
docker-compose up -d
```

This will start:

- Hayhooks server on port 1416
- Hayhooks MCP server on port 1417
- Open WebUI on port 3000

### 3. Access the Services

- **Hayhooks API**: <http://localhost:1416>
- **Open WebUI**: <http://localhost:3000>
- **Hayhooks API Documentation**: <http://localhost:1416/docs>

### 4. Deploy Example Pipelines

Install Hayhooks locally to use the CLI:

```
pip install hayhooks
```

Then deploy example pipelines:

```
# Deploy a sample pipeline
hayhooks pipeline deploy-files -n chat_with_website pipelines/chat_with_website_streaming
```

Alternative: Deploy via API

You can also deploy pipelines using the HTTP API endpoints: `POST /deploy_files` (PipelineWrapper files) or `POST /deploy-yaml` (YAML pipeline definition). See the [API Reference](https://deepset-ai.github.io/hayhooks/reference/api-reference/#pipeline-management) for details.

## Configuration Options

### Environment Variables

The following environment variables can be configured in `.env`:

| Variable           | Description              | Default |
| ------------------ | ------------------------ | ------- |
| `OPEN_WEBUI_PORT`  | Port for Open WebUI      | `3000`  |
| `WEBUI_DOCKER_TAG` | Tag for Open WebUI image | `main`  |

### Volume Mounts

The Docker Compose setup includes the following volume mounts:

- **Pipeline Directory**: `/pipelines` – Directory mounted inside the Hayhooks container where your pipeline wrappers or YAML files live. Hayhooks auto-deploys anything it finds here at startup.

## Integrating with Open WebUI

The Docker Compose setup comes pre-configured to integrate Hayhooks with Open WebUI:

### 1. Configure Open WebUI

1. Access Open WebUI at <http://localhost:3000>
1. Go to **Settings → Connections**
1. Add a new connection with:
1. **API Base URL**: `http://hayhooks:1416/v1`
1. **API Key**: `any-value` (not used by Hayhooks)

### 2. Deploy a Pipeline

Deploy a pipeline that supports chat completion:

```
hayhooks pipeline deploy-files -n chat_with_website pipelines/chat_with_website_streaming
```

Alternatively, use the [API endpoints](https://deepset-ai.github.io/hayhooks/reference/api-reference/#pipeline-management) (`POST /deploy_files` or `POST /deploy-yaml`).

### 3. Test the Integration

1. In Open WebUI, select the Hayhooks backend
1. Start a conversation with your deployed pipeline
1. The pipeline will respond through the Open WebUI interface

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Ensure ports 1416, 1417, and 3000 are available
1. **Permission Issues**: Make sure Docker has proper permissions
1. **Network Issues**: Check that containers can communicate with each other

### Logs

Check logs for troubleshooting:

```
# Hayhooks logs
docker-compose logs -f hayhooks

# Open WebUI logs
docker-compose logs -f openwebui
```

### Cleanup

To stop and remove all containers:

```
docker-compose down -v
```

## Next Steps

- [Configuration](https://deepset-ai.github.io/hayhooks/getting-started/configuration/index.md) - Learn about advanced configuration
- [Examples](https://deepset-ai.github.io/hayhooks/examples/overview/index.md) - Explore more examples
- [Open WebUI Integration](https://deepset-ai.github.io/hayhooks/features/openwebui-integration/index.md) - Deep dive into Open WebUI integration
