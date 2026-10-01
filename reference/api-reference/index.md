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

The deploy and undeploy endpoints exist only in the default, live-deployment mode. A server in [durable mode](https://deepset-ai.github.io/hayhooks/features/durable-execution/#durable-mode) does not register them, so they return `404`.

#### Deploy Pipeline (files)

```
POST /deploy_files
```

**Request Body:**

```
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

```
{
  "status": "success",
  "message": "Pipeline deployed successfully"
}
```

#### Undeploy Pipeline

```
POST /undeploy/{pipeline_name}
```

Remove a deployed pipeline.

**Response:**

```
{
  "status": "success",
  "message": "Pipeline undeployed successfully"
}
```

#### Get Pipeline Status

```
GET /status/{pipeline_name}
```

Check the status of a specific pipeline.

**Response:**

```
{
  "status": "Up!",
  "pipeline": "pipeline_name"
}
```

#### Get All Pipeline Statuses

```
GET /status
```

Get status of all deployed pipelines, the server mode, and durable deployment health.

**Response:**

```
{
  "status": "Up!",
  "pipelines": [
    "jobs"
  ],
  "durable_mode": true,
  "durable": {
    "healthy": true,
    "deployments": {
      "jobs": {
        "healthy": true,
        "configured_slots": 1,
        "running_slots": 1,
        "draining_slots": 0,
        "draining_runs": 0,
        "active_executions": 0,
        "maintenance_running": true,
        "accepting": true,
        "store_error_streak": 0,
        "counts": {
          "nonterminal": 0,
          "revision_nonterminal": 0,
          "revision_runnable": 0,
          "lease_expiry": 0
        }
      }
    }
  }
}
```

`durable_mode` is `true` when the pipeline set is fixed at startup. `durable` reports each durable deployment in [durable mode](https://deepset-ai.github.io/hayhooks/features/durable-execution/#durable-mode); `status` is `Degraded` when one is unhealthy. The endpoint always returns HTTP 200, so it is safe for liveness. Use `status` or `durable.healthy` from the JSON for readiness and alerts.

Durable health is at most one second old, and concurrent probes share one read. If that read exceeds one second, the deployment reports `"healthy": false` and `"operational_error": "TimeoutError"`, and the top-level status is `Degraded`.

### Pipeline Execution

#### Run Pipeline

```
POST /{pipeline_name}/run
```

Execute a deployed pipeline.

**Request Body:**

```
{
  "query": "What is the capital of France?"
}
```

**Response:**

```
{
  "result": "The capital of France is Paris."
}
```

### Durable Execution

`create_durable_router(deployment, ...)` adds these typed routes under the prefix the host mounts it at. In durable mode the Hayhooks server mounts one per durable wrapper, at `/{pipeline_name}`:

| Method | Route                                        | Result                                                       |
| ------ | -------------------------------------------- | ------------------------------------------------------------ |
| `POST` | `/{prefix}/run-durable`                      | Submit and return `202`, `Location`, execution ID, and links |
| `GET`  | `/{prefix}/executions/{execution_id}`        | Inspect the authoritative execution projection               |
| `POST` | `/{prefix}/executions/{execution_id}/cancel` | Request cooperative cancellation                             |
| `POST` | `/{prefix}/executions/{execution_id}/resume` | Validate resume input and requeue waiting work               |
| `GET`  | `/{prefix}/executions/{execution_id}/stream` | Reattachable SSE chunks and terminal event                   |

The submit and resume request schemas come from the deployment's Pydantic models and appear in OpenAPI. The execution projection keeps `result` as JSON so results written by an older immutable revision remain readable; the active revision still validates new results before committing them. A projection includes status, attempt, sequence, progress, public wait data, result or sanitized error, timestamps, and links. A new submission has `attempt: 0`; claims start at 1, and every resume, retry, handoff, and crash recovery claim increments it. It never exposes input, checkpoints, application state, lease/fence data, ownership, or idempotency material.

Status codes:

- `200`: inspection, terminal replay, or terminal cancellation result;
- `202`: accepted submission, cancellation request, or resume;
- `401` or `403`: rejected by the wrapper's `durable_owner_id` dependency in durable mode;
- `404`: missing execution or owner mismatch;
- `409`: an idempotency key was reused within the same deployment and owner with different explicitly sent request fields, or a revision or resume-state conflict occurred;
- `422`: request, resume, header, cursor, or payload validation failure;
- `500`: stored execution state is invalid, with detail `Durable execution state is invalid`; this response is not retryable;
- `503`: `Durable execution store is unavailable`, `Durable execution service is unavailable`, or an admission failure. Admission failures include `Retry-After: 1` when the nonterminal limit is reached or the deployment is shutting down.

SSE accepts `Last-Event-ID`. Every `chunk` carries `attempt`. A higher attempt means the run restarted from its last checkpoint, so clients discard text from lower attempts. Events are `chunk`, optional `gap`, one terminal `completed`, `failed`, or `canceled` event, and `error`. An interrupted stream sends `event: error` with data `{"detail":"Execution stream interrupted"}` and no `id`, then ends; reconnect with the last `Last-Event-ID`. The first frame and every idle 15 seconds are a `: heartbeat` comment. Undecodable entries are logged and skipped without an event.

A finished execution replays retained history and sends its terminal event immediately. When history has expired, a fresh stream gets only the terminal event; a resumed cursor gets `gap` followed by the terminal event. See [Durable Execution](https://deepset-ai.github.io/hayhooks/features/durable-execution/index.md) for semantics and ownership modes.

### OpenAI Compatibility

#### Chat Completion

```
POST /chat/completions
POST /v1/chat/completions
```

OpenAI-compatible chat completion endpoint.

**Request Body:**

```
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

```
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

```
GET /openapi.json
GET /openapi.yaml
```

Get the complete OpenAPI specification for programmatic access or tooling integration.

## Error Handling

### Error Response Format

```
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

```
curl -X POST http://localhost:1416/chat_pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"query": "Hello!"}'
```

```
import requests

response = requests.post(
    "http://localhost:1416/chat_pipeline/run",
    json={"query": "Hello!"}
)
print(response.json())
```

```
hayhooks pipeline run chat_pipeline --param 'query="Hello!"'
```

### OpenAI-Compatible Chat Completion

```
curl -X POST http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chat_pipeline",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

```
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

```
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

- [Environment Variables](https://deepset-ai.github.io/hayhooks/reference/environment-variables/index.md) - Configuration options
- [Logging](https://deepset-ai.github.io/hayhooks/reference/logging/index.md) - Logging configuration
