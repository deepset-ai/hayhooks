# API Reference

Hayhooks provides a comprehensive REST API for managing and executing Haystack pipelines and agents.

## Base URL

```text
http://localhost:1416
```

## Authentication

Currently, Hayhooks does not include built-in authentication. Consider implementing:

- Reverse proxy authentication
- Network-level security
- Custom middleware

## Endpoints

### Pipeline Management

The deploy and undeploy endpoints exist only in the default, live-deployment mode. A server in
[durable mode](../features/durable-execution.md#durable-mode) does not register them, so they return `404`.

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
GET /status/{pipeline_name}
```

Check the status of a specific pipeline.

**Response:**

```json
{
  "status": "Up!",
  "pipeline": "pipeline_name"
}
```

#### Get All Pipeline Statuses

```http
GET /status
```

Get status of all deployed pipelines, the server mode, and durable deployment health.

**Response:**

```json
{
  "status": "Up!",
  "pipelines": [
    "pipeline1",
    "pipeline2"
  ],
  "durable_mode": false,
  "durable": {"healthy": true, "deployments": {}}
}
```

`durable_mode` is `true` when the pipeline set is fixed at startup. `durable` reports each durable deployment in
[durable mode](../features/durable-execution.md#durable-mode); `status` is `Degraded` when one is unhealthy.

### Pipeline Execution

#### Run Pipeline

```http
POST /{pipeline_name}/run
```

Execute a deployed pipeline.

**Request Body:**

```json
{
  "query": "What is the capital of France?"
}
```

**Response:**

```json
{
  "result": "The capital of France is Paris."
}
```

### Durable Execution

`create_durable_router(deployment, ...)` adds these typed routes under the
prefix the host mounts it at. In durable mode the Hayhooks server mounts one per
durable wrapper, at `/{pipeline_name}`:

| Method | Route | Result |
|---|---|---|
| `POST` | `/{prefix}/run-durable` | Submit and return `202`, `Location`, execution ID, and links |
| `GET` | `/{prefix}/executions/{execution_id}` | Inspect the authoritative execution projection |
| `POST` | `/{prefix}/executions/{execution_id}/cancel` | Request cooperative cancellation |
| `POST` | `/{prefix}/executions/{execution_id}/resume` | Validate resume input and requeue waiting work |
| `GET` | `/{prefix}/executions/{execution_id}/stream` | Reattachable SSE chunks and terminal event |

The submit and resume request schemas come from the deployment's Pydantic
models and appear in OpenAPI. The execution projection keeps `result` as
JSON so results written by an older immutable revision remain readable; the
active revision still validates new results before committing them. A
projection includes status, attempt, sequence, progress, public wait data,
result or sanitized error, timestamps, and links. A new submission has
`attempt: 0`; claims start at 1, and every resume, retry, handoff, and crash
recovery claim increments it. It never exposes input, checkpoints, application
state, lease/fence data, ownership, or idempotency material.

Status codes:

- `200`: inspection, terminal replay, or terminal cancellation result;
- `202`: accepted submission, cancellation request, or resume;
- `404`: missing execution or owner mismatch;
- `409`: an idempotency key was reused within the same deployment and owner
  with different explicitly sent request fields, or a revision or resume-state
  conflict occurred;
- `422`: request, resume, header, cursor, or payload validation failure;
- `503`: admission closed or durable store unavailable.

SSE accepts `Last-Event-ID`. Every `chunk` carries `attempt`. A higher attempt
means the run restarted from its last checkpoint, so clients discard text from
lower attempts. Events are `chunk`, optional `gap`, and one terminal
`completed`, `failed`, or `canceled` event. See
[Durable Execution](../features/durable-execution.md) for semantics and
ownership modes.

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
  "stream": false
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

> MCP runs in a separate Starlette app when invoked via `hayhooks mcp run`. Use the configured Streamable HTTP endpoint `/mcp` or SSE `/sse` depending on your client. See the MCP feature page for details.

### Interactive API Documentation

Hayhooks provides interactive API documentation for exploring and testing endpoints:

- **Swagger UI**: `http://localhost:1416/docs` - Interactive API explorer with built-in request testing
- **ReDoc**: `http://localhost:1416/redoc` - Clean, responsive API documentation

### OpenAPI Schema

#### Get OpenAPI Schema

```http
GET /openapi.json
GET /openapi.yaml
```

Get the complete OpenAPI specification for programmatic access or tooling integration.

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

## Examples

### Running a Pipeline

=== "cURL"

    ```bash
    curl -X POST http://localhost:1416/chat_pipeline/run \
      -H 'Content-Type: application/json' \
      -d '{"query": "Hello!"}'
    ```

=== "Python"

    ```python
    import requests

    response = requests.post(
        "http://localhost:1416/chat_pipeline/run",
        json={"query": "Hello!"}
    )
    print(response.json())
    ```

=== "Hayhooks CLI"

    ```bash
    hayhooks pipeline run chat_pipeline --param 'query="Hello!"'
    ```

### OpenAI-Compatible Chat Completion

=== "cURL"

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

=== "Python"

    ```python
    import requests

    response = requests.post(
        "http://localhost:1416/v1/chat/completions",
        json={
            "model": "chat_pipeline",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }
    )
    print(response.json())
    ```

=== "OpenAI Python SDK"

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:1416/v1",
        api_key="not-needed"  # Hayhooks doesn't require auth by default
    )

    response = client.chat.completions.create(
        model="chat_pipeline",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print(response.choices[0].message.content)
    ```

## Next Steps

- [Environment Variables](environment-variables.md) - Configuration options
- [Logging](logging.md) - Logging configuration
