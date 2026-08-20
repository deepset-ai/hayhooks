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
- Live SSE chunk streaming that clients can detach from and reattach to.
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

This durable integration fragment assumes the host FastAPI application already
has authentication middleware that sets a stable principal on `request.state`
before the owner dependency runs. The authentication middleware itself is
application-specific and intentionally omitted:

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

The adapter exposes typed submit, inspect, stream, cancel, and resume routes.
It does not start workers or own the runtime. Host middleware and dependencies
retain control of authentication and authorization; the durable layer persists
only the stable owner ID returned by the dependency, which must be a non-empty
string of at most 512 characters (a misconfigured dependency fails closed with
500). Durable wrapper code can read that value through `context.owner_id` after
process recovery. The same deployment is also drivable without HTTP through
`deployment.submit(...)` and the store's resume/cancel operations; see
[Streaming chunks](#streaming-chunks) for the reattachable SSE endpoint.

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
and fences coordinate them, so the setting is a per-process, per-deployment
ceiling and effective concurrency is `processes ×
durable_execution_concurrency`. Keep wrapper revisions identical across replicas
and start with one to three processes and conservative concurrency.

Built-in providers snapshot `DurableSettings`, and the runtime adopts that
snapshot when a provider is supplied. Conflicting runtime and provider settings
are rejected before deployment creation. The selected provider is fixed for the
runtime's lifetime; create a new runtime to change storage backends.

## Redis layout

Each deployment has an isolated namespace with controls and opaque input,
checkpoint, result, error, wait, and progress payload keys, plus one `chunks`
stream per execution. It has exactly two sorted-set indexes:

| Key | Purpose |
|---|---|
| `runnable` | All queued work, scored by its retry deadline or immediate transition time. |
| `lease-expiry` | Running fences, scored by their Redis-server lease deadline. |
| `exec:<id>:chunks` | Bounded append-only display chunks, read by the execution SSE stream. |

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

## Streaming chunks

`GET /{pipeline}/executions/{execution_id}/stream` is a Server-Sent Events
stream of one execution's display chunks, followed by a terminal `completed`,
`failed`, or `canceled` event carrying the same public projection the inspect
route returns. The submit response advertises it as the `stream` link.

Chunks are best-effort display data, deliberately outside the durable fence. A
chunk append uses one non-transactional Redis pipeline and never writes the
control record, so token-rate streaming cannot invalidate a heartbeat. Nothing
about a chunk can fail a run either: an invalid or oversized payload (chunks are
capped at 64 KB) or a backend blip drops that chunk and logs it, because
replaying a pipeline to recover a display token is never the right trade.
Progress events remain the coarse durable audit trail.

```python
class StreamingWrapper(BasePipelineWrapper):
    durable_revision = "streaming-v1"

    async def run_durable_async(self, context: DurableContext, request: Question) -> Answer:
        result = await context.run_agent_async(
            messages=[ChatMessage.from_user(request.query)],
            streaming_callback=context.stream_chunk,
        )
        return Answer(reply=result["messages"][-1].text)
```

Bind the callback to the component when the work is a Pipeline rather than an
Agent. `async_streaming_generator` passes it per run in `pipeline_run_args`, and
that does not survive here: `run_pipeline_async` data is serialized into the
`PipelineSnapshot`, Haystack drops the callable it cannot serialize, and
`Pipeline.run` rebuilds its `data` from the snapshot when resuming, so a callback
passed as run data disappears at the first checkpoint.

Hayhooks provides the synchronous callback that a Pipeline component needs.
Binding it to a shared component is safe for the same reason
`async_streaming_generator` can hand the same module-level
`_async_streaming_callback` to every concurrent run: the callback carries no
per-run state and resolves its destination on each call from a `ContextVar`.
Hayhooks routes on `_ASYNC_STREAMING_QUEUE`; the durable path routes on the
execution context, which the engine sets per execution task and `asyncio.to_thread`
copies into the Pipeline's worker thread:

```python
from hayhooks import durable_streaming_callback


class StreamingPipelineWrapper(BasePipelineWrapper):
    durable_revision = "streaming-pipeline-v1"

    def setup(self) -> None:
        self.pipeline = Pipeline.loads(...)
        self.pipeline.get_component("llm").streaming_callback = durable_streaming_callback
```

`run_pipeline_async` drives the Pipeline on a worker thread, which is why that
callback uses the synchronous bridge. The Agent path is the other way round:
`run_agent_async` awaits `Agent.run_async` on the server loop, where the bridge
cannot work, so an Agent takes `context.stream_chunk` as shown above. What
`pipeline_run_args` injection actually buys `async_streaming_generator` is
control over *which* components stream without permanently mutating a shared
Pipeline; here the callback simply does nothing outside a durable execution. A
run-time `streaming_callback` still takes precedence over a bound one, so an
ordinary streaming endpoint on the same wrapper is unaffected. See
`examples/durable_chat_with_website` for the whole wrapper. Each SSE event
carries the entry ID as `id:` and the producing `attempt` in its payload; a
client resets its buffer when `attempt` increases, because a retried attempt
re-streams from its checkpoint. The server ignores chunks from an older attempt
once a newer one is known. Reconnecting clients resend `Last-Event-ID`
automatically and resume from that cursor. If the bounded log no longer contains
that cursor, the stream emits a `gap` event, replays the retained tail, and then
continues; reset or mark the client buffer as partial when that happens. A
browser `EventSource` reconnects whenever the server closes the connection,
including after the terminal event, so call `close()` once that event arrives.

The stream follows one execution for its whole life, not just one run of it: an
execution that suspends into `waiting` keeps its stream open and heartbeating
until it is resumed and reaches a terminal state, which for an approval-gated
workflow can be a long time. Detach and reattach with `Last-Event-ID` if a
connection parked that long is not what you want.

`durable_max_stream_chunks` bounds the log per execution (10 000 by default);
`0` disables chunk production entirely while leaving the endpoint working.
`durable_max_stream_chunk_bytes` caps a single chunk at 64 KB by default; an
oversized chunk is dropped, never failed, and it also sets how many entries one
read returns, so that a single read stays under 4 MB whatever the cap is. The
two bounds multiply: size Redis for
`durable_max_stream_chunks * durable_max_stream_chunk_bytes` per streaming
execution, times the executions running at once. The log expires with its
execution under `durable_terminal_ttl_seconds`. A new viewer that starts after
the log has overflowed receives only the retained tail; the terminal result
remains authoritative.

A stream that breaks after its headers were sent has no status code left to
report with, so it ends in an `error` event instead. Treat it the way a client
treats a dropped connection: reattach with `Last-Event-ID`, or read the
execution's terminal state from the inspect route.

## Revisions and rollout

Every durable wrapper, including a managed A2A Agent, declares a non-empty
`durable_revision`. Use an image digest or Git SHA in production and update it
with checkpoint-relevant code, prompts, configuration, or dependencies. Claims
and resumes verify that persisted work matches the active revision.

## Operations

`DurableExecutionManager.health_snapshot()` reports `nonterminal`, `runnable`,
`lease_expiry`, and the current worker store-error streak. Repeated claim or
transition failures make the deployment health snapshot unhealthy until a
worker completes a store operation successfully; `/status/{pipeline_name}`
then returns `503`. Alert on sustained runnable growth, repeated lease
recovery, worker/store health failures, and runs that exceed their expected
duration.

See [Durable execution operations](durable-execution-operations.md) for
deployment, retention, and incident guidance.
