# API Reference

Hayhooks provides a comprehensive REST API for managing and executing Haystack pipelines and agents.

## Base URL

```
http://localhost:1416
```

## Authentication

Currently, Hayhooks does not include built-in authentication. Consider implementing:
- Reverse proxy authentication
- Network-level security
- Custom middleware

## Endpoints

### Pipeline Management

#### Deploy Pipeline (files)
```http
POST /deploy_files
```

**Request Body:**
```json
{
  "name": "pipeline_name",
  "files": {
    "pipeline_wrapper.py": "...file content...",
    "other.py": "..."
  },
  "save_files": true,
  "overwrite": false
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Pipeline deployed successfully"
}
```

#### Undeploy Pipeline
```http
POST /undeploy/{pipeline_name}
```

Remove a deployed pipeline.

**Response:**
```json
{
  "status": "success",
  "message": "Pipeline undeployed successfully"
}
```

#### Get Pipeline Status
```http
GET /pipelines/{pipeline_name}/status
```

Check the status of a specific pipeline.

**Response:**
```json
{
  "name": "pipeline_name",
  "status": "running",
  "version": "1.0.0",
  "uptime": 3600
}
```

#### Get All Pipeline Statuses
```http
GET /status
```

Get status of all deployed pipelines.

**Response:**
```json
{
  "pipelines": [
    {
      "name": "pipeline1",
      "status": "running",
      "version": "1.0.0"
    },
    {
      "name": "pipeline2",
      "status": "error",
      "version": "1.0.0",
      "error": "Import error"
    }
  ]
}
```

### Pipeline Execution

#### Run Pipeline
```http
POST /{pipeline_name}/run
```

Execute a deployed pipeline.

**Request Body:**
```json
{
  "query": "What is the capital of France?",
  "params": {
    "temperature": 0.7
  }
}
```

**Response:**
```json
{
  "result": "The capital of France is Paris."
}
```

### OpenAI Compatibility

#### Chat Completion
```http
POST /chat/completions
POST /v1/chat/completions
```

OpenAI-compatible chat completion endpoint.

**Request Body:**
```json
{
  "model": "pipeline_name",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "stream": false,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "chat-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "pipeline_name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 20,
    "total_tokens": 32
  }
}
```

#### Streaming Chat Completion
Use the same endpoints with `"stream": true`. Hayhooks streams chunks in OpenAI-compatible format.

### MCP Server

#### MCP Health Check
```http
GET /mcp/health
```

Check MCP server status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

#### List MCP Tools
```http
GET /mcp/tools
```

Get available MCP tools.

**Response:**
```json
{
  "tools": [
    {
      "name": "get_all_pipeline_statuses",
      "description": "Get status of all deployed pipelines"
    },
    {
      "name": "deploy_pipeline",
      "description": "Deploy a new pipeline"
    }
  ]
}
```

### OpenAPI Schema

#### Get OpenAPI Schema
```http
GET /openapi.json
GET /openapi.yaml
```

Get the complete OpenAPI specification.

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

### Common Error Codes

- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Pipeline or endpoint not found
- **500 Internal Server Error**: Server-side error

## Rate Limiting

Currently, Hayhooks does not include built-in rate limiting. Consider implementing:
- Reverse proxy rate limiting
- Custom middleware
- Request throttling

<!-- WebSocket section removed: not supported by Hayhooks server at this time -->

## Webhook Support

Configure webhooks for pipeline events:

```json
{
  "webhooks": {
    "on_deploy": "https://your-webhook.com/deploy",
    "on_error": "https://your-webhook.com/error"
  }
}
```

## Examples

### Deploy via cURL

```bash
curl -X POST http://localhost:1416/pipelines/deploy \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "chat_pipeline",
    "files": [
      {
        "name": "pipeline_wrapper.py",
        "content": "'$(base64 -w 0 pipeline_wrapper.py)'"
      }
    ]
  }'
```

### Run Pipeline via cURL

```bash
curl -X POST http://localhost:1416/chat_pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"query": "Hello!"}'
```

### OpenAI-Compatible Request

```bash
curl -X POST http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chat_pipeline",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Next Steps

- [Environment Variables](environment-variables.md) - Configuration options
- [Logging](logging.md) - Logging configuration
- [Deployment Guidelines](../deployment/deployment-guidelines.md) - Production deployment
