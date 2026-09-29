# Standalone FastAPI integration

This example embeds the Hayhooks durable engine in a FastAPI app that is not
the Hayhooks server. It runs a real Haystack Pipeline, stores execution state in
Redis, validates requests and resume input with Pydantic, scopes executions to
an authenticated owner, and manages workers and Redis through the app lifespan.

From the repository root:

```bash
pip install -e ".[durable]"
docker compose -f examples/durable-compose.yaml up -d
export APP_API_KEY="$(openssl rand -hex 32)"
uvicorn examples.durable_fastapi.app:app --port 8000
```

Submit a document. Reusing the same `Idempotency-Key` safely returns the same
execution instead of creating duplicate work.

```bash
curl -i http://localhost:8000/jobs/document-analysis/run-durable \
  -H "Authorization: Bearer $APP_API_KEY" \
  -H "Idempotency-Key: document-42-v1" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "document-42",
    "text": "Haystack pipelines can continue after a process restart.",
    "require_approval": true,
    "processing_delay_seconds": 10
  }'
```

The response is `202 Accepted` and contains links. Use the returned execution ID
to approve and follow the run:

```bash
curl -X POST http://localhost:8000/jobs/document-analysis/executions/EXECUTION_ID/resume \
  -H "Authorization: Bearer $APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'

curl -N http://localhost:8000/jobs/document-analysis/executions/EXECUTION_ID/stream \
  -H "Authorization: Bearer $APP_API_KEY"
```

Restart Uvicorn during the ten-second processing delay to verify that Redis
preserves the checkpoint. The `extract` component is not repeated: recovery
resumes at `analyze`. The delay only makes recovery easy to observe; replace the
example analysis with the real Pipeline or Agent work in your application.

The API-key dependency is deliberately small so the example runs by itself. In
an existing application, replace it with your normal authentication dependency
and return a stable user or tenant ID. External writes are still at least once;
make them idempotent with a unique key such as
`f"{context.execution_id}:publish"`.
