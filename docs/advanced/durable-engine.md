# Durable engine

Hayhooks durable execution provides detached, checkpointed work for **Haystack
3** Pipelines and Agents.

It accepts validated work, runs it outside the request, resumes from Haystack
checkpoints, and recovers after a worker or process disappears. External
writes use application idempotency keys derived from the execution ID and
logical step, keeping replay safe when a process exits between an effect and
its next checkpoint.

## Supported capabilities

- Detached, typed Pipeline and Agent execution through durable REST endpoints.
- Pipeline snapshots and Agent state checkpoints with bounded retries,
  progress, cancellation, and typed wait/resume.
- Redis-backed fenced claims and lease recovery across process restarts.
- Idempotent submission, optional owner-isolated REST access, and managed A2A
  task projection.
- Native Redis TTL for terminal records, plus an equivalent volatile
  in-memory backend for local development and tests.
- Explicit revision checks and safe deploy/undeploy handling.

## Supported scope and tradeoffs

- validated durable input before submission succeeds;
- at-least-once execution with one fenced worker owner at a time;
- checkpoints, progress, retries, cancellation, wait/resume, and terminal
  results; and
- a public result that excludes private input, checkpoint, state, owner, and
  fence details.

The pure reducer in `hayhooks.durable.engine` is the only lifecycle
decision-maker. Storage atomically persists its plan and derives indexes from
the old and new control records.

```text
queued ── claim ──> running ── complete/fail/cancel ──> terminal
  │                    │
  │                    ├── checkpoint / heartbeat
  │                    ├── retry ──> queued (due later)
  │                    └── suspend ──> waiting ── resume ──> queued
  └── cancel ──> terminal
```

## Embedding the runtime

Applications can import the runtime, providers, deployment contracts, public
exceptions, and FastAPI adapter directly from `hayhooks.durable`. A standalone
runtime starts only deployments attached to that runtime; it never inspects the
Hayhooks pipeline registry.

This complete `app.py` adds an authenticated durable API to an existing FastAPI
application. Authentication middleware is expected to set a stable principal
on `request.state` before the owner dependency runs:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from haystack import Pipeline
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext
from hayhooks.durable import DurableRuntime, DurableSettings, create_durable_router


class JobRequest(BaseModel):
    document_id: str


class JobResult(BaseModel):
    indexed: bool


class JobWrapper(BasePipelineWrapper):
    durable_revision = "job-v1"

    def setup(self) -> None:
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: JobRequest) -> JobResult:
        # Replace this example body with checkpointed Pipeline work.
        return JobResult(indexed=bool(request.document_id))


runtime = DurableRuntime(
    durable_settings=DurableSettings(
        durable_store="memory",  # Development only; use Redis in production.
        durable_poll_interval=0.05,
    )
)

wrapper = JobWrapper()
wrapper.setup()
deployment = runtime.deployment("jobs", wrapper)


def current_owner_id(request: Request) -> str:
    principal = request.state.principal
    return f"{principal.tenant_id}:{principal.subject_id}"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        await runtime.start()
        yield
    finally:
        await runtime.close()


app = FastAPI(lifespan=lifespan)
app.include_router(
    create_durable_router(deployment, owner_id_dependency=current_owner_id),
    prefix="/jobs",
)
```

The adapter exposes typed submit, inspect, cancel, and resume routes. It does
not start workers or own the runtime. Host middleware and dependencies retain
control of authentication and authorization; the durable layer persists only
the stable owner ID returned by the dependency. Durable wrapper code can read
that value through `context.owner_id` after process recovery.

Passing `owner_id_dependency=None` is an explicit unscoped security choice:

```python
app.include_router(
    create_durable_router(deployment, owner_id_dependency=None),
    prefix="/internal-jobs",
)
```

In this mode, possession of the unguessable execution ID grants access. Use it
only behind one application-wide authorization boundary or for local
development.

For production Redis, give each application/environment an isolated prefix:

```python
from hayhooks.durable import DurableRuntime, RedisExecutionStoreProvider

provider = RedisExecutionStoreProvider(
    redis_url="redis://localhost:6379/0",
    key_prefix="myapp:production:durable",
)
runtime = DurableRuntime(provider)
```

When the host owns an existing binary Redis client, pass `redis=client` and
`close_redis=False`, then close the durable runtime before closing the client.
Do not use `decode_responses=True`.

Every Uvicorn worker owns a runtime, Redis pool, and worker tasks. Redis leases
and fences coordinate them, so effective concurrency is `processes ×
durable_execution_concurrency` per deployment. Keep wrapper revisions identical
across replicas and start with one to three processes and conservative
concurrency.

Built-in providers snapshot `DurableSettings`, and the runtime adopts that
snapshot when a provider is supplied. Conflicting runtime and provider settings
are rejected before deployment creation. The selected provider is fixed for the
runtime's lifetime; create a new runtime to change storage backends.

## Redis layout

Each deployment has an isolated namespace with controls and opaque input,
checkpoint, result, error, wait, and progress payload keys. It has exactly two
sorted-set indexes:

| Key | Purpose |
|---|---|
| `runnable` | All queued work, scored by its retry deadline or immediate transition time. |
| `lease-expiry` | Running fences, scored by their Redis-server lease deadline. |

The namespace also contains a `capacity` hash with only `nonterminal` and one
idempotency binding per execution. Terminal execution and idempotency keys use
native Redis TTL. The in-memory backend schedules equivalent cleanup. The store
assigns progress sequence numbers when it commits each transition, preserving
checkpoint and cancellation progress when they race.

Workers poll `runnable` every configured poll interval, one second by default.
They use Redis `TIME` and read one due member without removing it. Multiple
replicas can observe it; the watched control and fence let exactly one
transition to `running`. Lease maintenance uses the same interval and recovers
at most 100 expired entries per pass. The default averages about 500 ms claim
latency and can add up to one second to claim or lease recovery.

The controlled beta deployment profile uses one logical deployment with one to
three replicas and low-to-moderate load. Its two indexes, native TTL, and
single reducer keep the worker model observable during normal operation and
recovery.

## Revisions and rollout

Every durable wrapper, including a managed A2A Agent, declares a non-empty
`durable_revision`. Use an image digest or Git SHA in production and update it
with checkpoint-relevant code, prompts, configuration, or dependencies. Claims
and resumes verify that persisted work matches the active revision.

## Operations

`DurableExecutionManager.health_snapshot()` reports `nonterminal`, `runnable`,
`lease_expiry`, and the current worker store-error streak. Repeated claim or
transition failures make readiness unhealthy until a worker completes a store
operation successfully. Alert on sustained runnable growth, repeated lease
recovery, worker/store health failures, and runs that exceed their expected
duration.

See [Durable execution operations](durable-execution-operations.md) for
deployment, retention, and incident guidance.
