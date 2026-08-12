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

- durable input before submission succeeds;
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

Applications can import `DurableRuntime`, `ExecutionStore`,
`ExecutionStoreProvider`, `InMemoryExecutionStoreProvider`, and
`RedisExecutionStoreProvider` directly from `hayhooks.durable`. A standalone
runtime starts only deployments attached to that runtime; it does not inspect
Hayhooks' process-global pipeline registry.

This complete `app.py` embeds an in-memory durable worker in FastAPI. Its tool
simulates an eight-second upstream call so detached execution is easy to see:

```python
import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, status
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext
from hayhooks.durable import DurableRuntime, ExecutionResult, InMemoryExecutionStoreProvider
from hayhooks.settings import AppSettings


class AgentRequest(BaseModel):
    question: str


@tool
async def check_order(order_id: Annotated[str, "The customer's order ID"]) -> str:
    """Return the current shipping status for an order."""
    # Intentional demo delay: replace it with a real upstream API call.
    await asyncio.sleep(8)
    # Read-only tools are replay-safe; make mutating tools idempotent.
    return f"Order {order_id} shipped and arrives Friday."


class SupportAgentWrapper(BasePipelineWrapper):
    # Bump this when checkpoint-relevant code or prompts change.
    durable_revision = "support-agent-v1"

    def setup(self) -> None:
        self.pipeline = Agent(
            chat_generator=OpenAIChatGenerator(),
            system_prompt="Help customers with their orders. Use the order tool when needed.",
            tools=[check_order],
        )

    async def run_durable_async(self, context: DurableContext, request: AgentRequest) -> dict:
        return await context.run_agent_async(messages=[ChatMessage.from_user(request.question)])


durable_settings = AppSettings(durable_store="memory", durable_poll_interval=0.05)
provider = InMemoryExecutionStoreProvider(app_settings=durable_settings)
runtime = DurableRuntime(provider)

wrapper = SupportAgentWrapper()
wrapper.setup()
deployment = runtime.deployment("support-agent", wrapper)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        await runtime.start()
        yield
    finally:
        await runtime.close()


app = FastAPI(lifespan=lifespan)


@app.post("/agent-runs", response_model=ExecutionResult, status_code=status.HTTP_202_ACCEPTED)
async def submit_agent_run(request: AgentRequest) -> ExecutionResult:
    _, record = await deployment.submit(request.model_dump(mode="json"))
    return ExecutionResult.model_validate(record.safe_view(links={"self": f"/agent-runs/{record.execution_id}"}))


@app.get("/agent-runs/{execution_id}", response_model=ExecutionResult)
async def get_agent_run(execution_id: str) -> ExecutionResult:
    try:
        record = await deployment.get(execution_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Agent run not found") from error
    return ExecutionResult.model_validate(record.safe_view(links={"self": f"/agent-runs/{record.execution_id}"}))
```

Run it and submit work:

```bash
pip install "hayhooks[durable]"
export OPENAI_API_KEY="your-api-key"
uvicorn app:app

execution_id="$(
  curl --fail --silent -X POST http://127.0.0.1:8000/agent-runs \
    -H 'content-type: application/json' \
    -d '{"question":"Where is order A-123?"}' | jq -r '.execution_id'
)"

# The first poll should show `queued` or `running`; repeat until `completed`.
curl --fail --silent "http://127.0.0.1:8000/agent-runs/${execution_id}" | jq
```

The runtime owns provider shutdown. Built-in providers snapshot their settings,
and the runtime adopts that snapshot when a provider is supplied. Pass custom
settings once—either to a built-in provider as above, or to `DurableRuntime`
when it selects the default provider. Conflicting runtime and provider settings
are rejected before a deployment is created.

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
native Redis TTL. The in-memory backend schedules equivalent cleanup.

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
and `lease_expiry`. Alert on sustained runnable growth, repeated lease recovery,
worker/store health failures, and runs that exceed their expected duration.

See [Durable execution operations](durable-execution-operations.md) for
deployment, retention, and incident guidance.
