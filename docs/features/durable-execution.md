# Durable Execution

Durable execution runs a typed Pipeline or Agent after the submit request has
returned. Checkpoints, progress, retries, cancellation, waits, results, and
stream cursors survive a Hayhooks restart when Redis is used.

Install the durable dependencies and start Redis:

```bash
pip install "hayhooks[durable]"
docker compose -f examples/durable-compose.yaml up -d
```

## Lifecycle

```mermaid
stateDiagram-v2
    [*] --> queued: submit
    queued --> running: fenced claim
    queued --> canceled: cancel
    running --> queued: retry or expired lease
    running --> waiting: suspend
    running --> completed: complete
    running --> failed: fail or attempt budget
    running --> canceled: requested cancellation wins
    waiting --> queued: resume
    waiting --> canceled: cancel
```

The pure reducer is the only lifecycle authority. Memory and Redis stores apply
its transition plans atomically. Redis supplies authoritative time, runnable and
lease indexes, fencing, retention, and recovery across processes.

## Author a durable wrapper

A wrapper declares one immutable revision and implements exactly one durable
method. The first bound parameter is `DurableContext`; the second is any
Pydantic request model. An optional Pydantic resume model validates input sent
to the resume endpoint.

```python
from haystack import Pipeline
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext


class Request(BaseModel):
    value: int


class Approval(BaseModel):
    approved: bool


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "job-v1"
    durable_resume_model = Approval

    def setup(self) -> None:
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, payload: Request) -> dict:
        if not context.state.get("approved"):
            resume = context.resume_input
            if resume is None:
                await context.suspend({"kind": "approval", "message": "Continue?"})
            if not Approval.model_validate(resume).approved:
                raise ValueError("execution was rejected")
            context.state["approved"] = True
        await context.report_progress("Running")
        return {"value": payload.value}
```

Use `context.run_pipeline[_async](..., checkpoint_at="component")` for one
Pipeline component boundary. Agent wrappers use `context.run_agent[_async]`;
the adapter restores Agent state and checkpoints continuing tool/LLM loops.

## REST and SSE

Each durable wrapper adds typed routes under `/{pipeline_name}`. Submission
returns `202`, a random execution ID, a `Location` header, and links for inspect,
cancel, resume, and stream. See the [API reference](../reference/api-reference.md#durable-execution)
for the route and status-code contract.

SSE streams are reattachable with `Last-Event-ID`. Chunks are bounded,
attempt-tagged display data: they may be dropped without failing work. A `gap`
event means the requested cursor expired and the retained tail follows. The
terminal `completed`, `failed`, or `canceled` event contains the authoritative
execution projection.

## Ownership and embedding

Hayhooks uses unscoped bearer-ID access: possession of the random execution ID
grants access. Caller-provided idempotency keys support safe submission retries
and act as bearer secrets in this mode, so generate them with the same entropy
and confidentiality as execution IDs. A standalone
FastAPI host can pass `owner_id_dependency` to `create_durable_router`. The host
authenticates the request and returns a stable, non-empty owner ID; the router
then scopes inspect, cancel, resume, stream, and idempotency to that owner while
hiding mismatches as `404`.

The portable `hayhooks.durable` package accepts an explicit store, deployment,
and runner and does not import the Hayhooks server. Hosts own runtime startup,
shutdown, authentication, and Redis client lifetime.

## Polling and recovery

Runnable work and lease expiry use independent polling intervals. An empty
Redis scheduling index is checked with one sorted-set command and does not
request Redis time. Redis time remains authoritative whenever an index contains
a deadline that must be evaluated.

The worker interval bounds ordinary pickup latency while workers are idle. The
maintenance interval bounds the additional delay between lease expiry and
recovery. Unexpected local worker-task exits are supervised immediately and do
not wait for lease maintenance or issue a Redis command. See
[Durable Operations](../deployment/durable-operations.md#polling-tradeoffs) for
configuration profiles and command-rate estimates.

Fixed polling is intentionally the correctness and compatibility baseline, not
a claim that it is the most efficient wake-up mechanism at every scale. The
empty-index optimization and independent intervals keep that baseline
inexpensive and predictable. If measured deployment or replica scale justifies
the added coordination, advisory wake-up notifications can be layered over the
same sorted-set source of truth. Bounded safety polling must remain so a lost
notification affects latency, not correctness.

## Delivery and revisions

Execution is at least once. Fencing blocks stale workers from committing, but a
process can fail after an external effect and before its checkpoint. Make every
external write idempotent with a key derived from the execution ID and logical
step.

Queued, running, and waiting executions are pinned to their immutable wrapper
revision, and workers claim only their own revision. Hayhooks rejects dynamic
overwrite or undeploy with `409` while such work exists. It does not
force-delete live durable state.

Long-running A2A execution, A2A task persistence, push delivery, and A2A resume
are not supported in this release. Existing A2A execution remains request-bound.

See the [durable execution](https://github.com/deepset-ai/hayhooks/tree/main/examples/durable_execution)
and [website stream](https://github.com/deepset-ai/hayhooks/tree/main/examples/durable_chat_with_website)
examples for retry, approval, checkpoints, cancellation, restart, and reconnect
flows.
