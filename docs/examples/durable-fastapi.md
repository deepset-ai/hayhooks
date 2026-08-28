# Embed Durable Execution in FastAPI

The durable engine is portable: a FastAPI application can host it without
running the Hayhooks server. The maintained
[standalone example](https://github.com/deepset-ai/hayhooks/tree/main/examples/durable_fastapi)
runs a Haystack Pipeline with Redis-backed recovery, typed approval, owner
isolation, idempotent submission, SSE, and health reporting.

## Integration shape

The application creates one Redis client, store, Haystack adapter, deployment,
and runtime. FastAPI's lifespan starts workers only after Redis initialization
succeeds and closes both workers and the client during shutdown.

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from hayhooks.durable import (
    DurableDeployment,
    DurableRuntime,
    create_durable_router,
)
from hayhooks.durable.haystack import HaystackDurableAdapter
from hayhooks.durable.redis import RedisExecutionStore

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=False)
store = RedisExecutionStore(redis, "document-analysis")
adapter = HaystackDurableAdapter(pipeline)
deployment = DurableDeployment(
    "document-analysis",
    "document-analysis-v1",
    store,
    DocumentRequest,
    run_document_analysis,
    result_model=AnalysisResult,
    resume_model=Approval,
    kind=adapter.kind,
    adapter=adapter,
)
runtime = DurableRuntime()
runtime.add(deployment)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await runtime.start()
    try:
        yield
    finally:
        await runtime.close()
        await redis.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(
    create_durable_router(deployment, owner_id_dependency=owner_id),
    prefix="/jobs/document-analysis",
)
```

The full [`app.py`](https://github.com/deepset-ai/hayhooks/blob/main/examples/durable_fastapi/app.py)
includes the Pydantic models, Pipeline, approval flow, authentication dependency,
and `/health` endpoint.

## Run it

From the repository root:

```bash
pip install -e ".[durable]"
docker compose -f examples/durable-compose.yaml up -d
export APP_API_KEY="$(openssl rand -hex 32)"
uvicorn examples.durable_fastapi.app:app --port 8000
```

Submit work with an idempotency key. Repeating the same request with the same
key returns the existing execution; changing the payload produces `409`.

```bash
curl -i http://localhost:8000/jobs/document-analysis/run-durable \
  -H "Authorization: Bearer $APP_API_KEY" \
  -H "Idempotency-Key: document-42-v1" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "document-42",
    "text": "Haystack pipelines can continue after a restart.",
    "require_approval": true,
    "processing_delay_seconds": 10
  }'
```

The response contains `self`, `resume`, `cancel`, and `stream` links. Approve
the waiting execution, then connect to its stream:

```bash
curl -X POST http://localhost:8000/jobs/document-analysis/executions/EXECUTION_ID/resume \
  -H "Authorization: Bearer $APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'

curl -N http://localhost:8000/jobs/document-analysis/executions/EXECUTION_ID/stream \
  -H "Authorization: Bearer $APP_API_KEY"
```

Restart Uvicorn during the ten-second processing delay to exercise recovery.
The Pipeline checkpoint is before `analyze`, so the completed `extract`
component is restored rather than run again. The delay only makes recovery easy
to observe; replace the example analysis with the real Pipeline or Agent work
in your application.

## Adapt it to an existing app

- Replace the example API key with the application's existing authentication
  dependency and return its stable user or tenant ID.
- Keep `decode_responses=False`; the Redis store validates and persists binary
  payloads.
- Keep the revision immutable while work is live. Change it only when deploying
  incompatible runner or Pipeline behavior.
- Make external writes idempotent with a unique key such as
  `f"{context.execution_id}:publish"`.
- Expose `runtime.health()` through existing health checks and follow
  [Durable Operations](../deployment/durable-operations.md) for Redis persistence,
  capacity, leases, and monitoring.
