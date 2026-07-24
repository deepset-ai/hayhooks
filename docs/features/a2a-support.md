# A2A Support

Hayhooks supports the [A2A protocol](https://a2a-protocol.org) (Agent2Agent) and can act as an A2A server, exposing deployed pipelines and agents as A2A agents that other agents can discover and delegate tasks to.

A2A complements [MCP support](mcp-support.md): MCP exposes pipelines as **tools** for an agent to call (agent→tool), while A2A exposes them as **agents** that other agents talk to (agent→agent).

## Overview

The Hayhooks A2A Server:

- Exposes deployed chat, durable Agent, and native A2A wrappers as A2A agents
- Serves a per-agent [Agent Card](#agent-cards) for discovery, auto-generated from the pipeline and customizable from the wrapper
- Implements the JSON-RPC protocol binding of the [A2A specification](https://a2a-protocol.org/latest/specification/) (v1.0), including SSE streaming
- Streams pipeline output incrementally as task artifact updates
- Supports detached long-running task execution with polling, subscription, and cooperative async cancellation

## Requirements

- Install with `pip install hayhooks[a2a]` (uses the official [a2a-sdk](https://github.com/a2aproject/a2a-python)
  and [redis-py](https://redis.io/docs/latest/develop/clients/redis-py/) clients)

## Getting Started

### Install with A2A Support

```bash
pip install hayhooks[a2a]
```

### Start the A2A Server

```bash
hayhooks a2a run
```

This starts the A2A server on `HAYHOOKS_A2A_HOST:HAYHOOKS_A2A_PORT` (default: `localhost:1418`), deploying pipelines from `HAYHOOKS_PIPELINES_DIR` (or `--pipelines-dir`).

### Configuration

Environment variables for the A2A server:

```bash
HAYHOOKS_A2A_HOST=localhost         # A2A server host
HAYHOOKS_A2A_PORT=1418              # A2A server port
HAYHOOKS_A2A_EXTERNAL_URL=          # Base URL advertised in agent cards
                                    # (set when behind a reverse proxy)
HAYHOOKS_A2A_V0_3_COMPAT=true       # Also accept A2A spec 0.3 requests
                                    # (used by older clients and tools)
HAYHOOKS_A2A_TASK_STORE=auto        # auto, memory, or redis
HAYHOOKS_A2A_REDIS_URL=redis://localhost:6379/0
HAYHOOKS_A2A_REDIS_KEY_PREFIX=hayhooks:a2a
HAYHOOKS_A2A_TASK_STORE_PROVIDER=   # Optional module:ClassName for a custom
                                    # TaskStoreProvider
HAYHOOKS_DURABLE_STORE=redis        # Redis by default; memory is volatile
HAYHOOKS_DURABLE_REDIS_URL=redis://localhost:6379/0
HAYHOOKS_DURABLE_REDIS_KEY_PREFIX=hayhooks:durable
HAYHOOKS_DURABLE_EXECUTION_CONCURRENCY=1
                                    # Workers per deployed durable Agent
```

## Which pipelines are exposed

A deployed pipeline is exposed as an A2A agent when it uses either authoring mode:

- **Chat compatibility**: implement `run_chat_completion` or `run_chat_completion_async`, the same methods used by the [OpenAI-compatible chat endpoints](openai-compatibility.md).
- **Native A2A**: inherit from `hayhooks.a2a.A2APipelineWrapper` and return an SDK `AgentExecutor` from `create_a2a_agent_executor()`.
- **Durable Agent**: inherit from `hayhooks.a2a.A2APipelineWrapper`, set `durable = True`, and assign a Haystack 3 `Agent` to `self.pipeline`. Hayhooks supplies the detached executor, checkpoints, progress projection, and durable store.

To exclude a chat-capable pipeline from A2A, set `skip_a2a` on the wrapper:

```python
class PipelineWrapper(BasePipelineWrapper):
    skip_a2a = True
```

If a wrapper implements both modes, A2A uses the native executor. The OpenAI-compatible endpoints continue to use the chat methods.

## Endpoints

Each exposed pipeline is mounted under its own path prefix:

| Endpoint | Description |
|----------|-------------|
| `GET /{pipeline_name}/.well-known/agent-card.json` | The pipeline's agent card |
| `POST /{pipeline_name}/` | JSON-RPC binding (`SendMessage`, `SendStreamingMessage`, `GetTask`, ...) |
| `GET /status` | Server status and the list of exposed agents |

For example, with a deployed `weather_agent` pipeline:

```bash
curl http://localhost:1418/weather_agent/.well-known/agent-card.json
```

## Agent Cards

Agent cards are generated automatically: the card name is the pipeline name, the description comes from the pipeline's registry metadata, and a single default skill is created. Override any of it with the `a2a_card` class attribute:

```python
class PipelineWrapper(BasePipelineWrapper):
    a2a_card = {
        "name": "weather_agent",
        "description": "Answers questions about the current weather in any city.",
        "version": "2.0.0",
        "skills": [
            {
                "id": "get_current_weather",
                "name": "Get current weather",
                "description": "Report current conditions for a city.",
                "tags": ["weather"],
                "examples": ["What's the weather in Berlin right now?"],
            }
        ],
    }
```

## Calling an agent

With the [a2a-sdk](https://github.com/a2aproject/a2a-python) client:

```python
import asyncio

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest


async def main():
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:1418/weather_agent")
        card = await resolver.get_agent_card()

        client = await create_client(agent=card, client_config=ClientConfig(streaming=True, httpx_client=httpx_client))
        try:
            request = SendMessageRequest(message=new_text_message("Weather in Berlin?", role=Role.ROLE_USER))
            async for response in client.send_message(request):
                if response.HasField("artifact_update"):
                    print(get_stream_response_text(response), end="", flush=True)
        finally:
            await client.close()


asyncio.run(main())
```

Or with plain JSON-RPC over HTTP:

```bash
curl -s http://localhost:1418/weather_agent/ \
  -H "Content-Type: application/json" -H "A2A-Version: 1.0" \
  -d '{"jsonrpc": "2.0", "id": "1", "method": "SendMessage",
       "params": {"message": {"messageId": "m1", "role": "ROLE_USER",
                              "parts": [{"text": "Weather in Berlin?"}]}}}'
```

## Task lifecycle and streaming

Each request is handled as an A2A task:

1. A `Task` is created from the incoming message.
2. The task transitions to `working` and the pipeline's chat completion method runs.
3. Pipeline output is emitted as a single `response` artifact. Streaming results (generators returned by `streaming_generator` / `async_streaming_generator`) are emitted incrementally as artifact chunk updates, so `SendStreamingMessage` clients receive text as it is produced.
4. The task ends in `completed`, `failed`, or `canceled`. Enable `HAYHOOKS_SHOW_TRACEBACKS` to include tracebacks in failure messages.

By default, non-streaming `SendMessage` remains blocking for backward compatibility: the response is returned after the task reaches a terminal or interrupted state.

For detached execution, set `configuration.returnImmediately`:

```bash
curl -s http://localhost:1418/weather_agent/ \
  -H "Content-Type: application/json" -H "A2A-Version: 1.0" \
  -d '{"jsonrpc": "2.0", "id": "1", "method": "SendMessage",
       "params": {"configuration": {"returnImmediately": true},
                  "message": {"messageId": "m1", "role": "ROLE_USER",
                              "parts": [{"text": "Start the long task"}]}}}'
```

The response contains a non-terminal task. Poll it with `GetTask`:

```bash
curl -s http://localhost:1418/weather_agent/ \
  -H "Content-Type: application/json" -H "A2A-Version: 1.0" \
  -d '{"jsonrpc": "2.0", "id": "2", "method": "GetTask",
       "params": {"id": "<task-id>"}}'
```

Or subscribe to an active task with `SubscribeToTask` to receive the latest task snapshot followed by task updates over SSE:

```bash
curl -N http://localhost:1418/weather_agent/ \
  -H "Content-Type: application/json" -H "A2A-Version: 1.0" \
  -d '{"jsonrpc": "2.0", "id": "3", "method": "SubscribeToTask",
       "params": {"id": "<task-id>"}}'
```

When `HAYHOOKS_A2A_V0_3_COMPAT=true`, A2A 0.3 clients can request the same detached behavior with `configuration.blocking=false`.

## Task storage

With `HAYHOOKS_A2A_TASK_STORE=auto`, Hayhooks uses Redis for Redis-backed durable Agents and otherwise gives each exposed agent its own A2A SDK `InMemoryTaskStore`. An explicit `memory` choice is never overridden.

Task storage is server infrastructure rather than pipeline configuration. Hayhooks includes independent in-memory and Redis-backed providers. Select the built-in Redis provider with `HAYHOOKS_A2A_TASK_STORE=redis` or `hayhooks a2a run --task-store redis`; configure its URL and key prefix with `HAYHOOKS_A2A_REDIS_URL` and `HAYHOOKS_A2A_REDIS_KEY_PREFIX`.

The A2A extra includes the official Redis client. Redis task records are protobuf payloads scoped by agent and resolved owner. A bounded reconciler uses renewable per-task leases and compare-and-set projection versions, so additional replicas do not create one polling coroutine per task and a stale projector cannot overwrite a newer state. Reconciliation checks execution sequences in batches and loads full execution records only when they changed. Recovery task payloads are also loaded by owner batches. Persistent task records do not replay historical live event queues.

Terminal tasks use `HAYHOOKS_A2A_TERMINAL_TASK_TTL_SECONDS`. Runtime maintenance performs cleanup even when no later A2A request arrives and removes the protobuf payload plus owner, active, retention, and version indexes. Execution-record retention remains independent.

For custom backends, implement `TaskStoreProvider` in an importable Python module. Hayhooks calls `create_task_store()` once for each agent at startup, passing its deployed pipeline name, and calls `close()` when the server shuts down. A custom provider is selected with `HAYHOOKS_A2A_TASK_STORE_PROVIDER` or `--task-store-provider`; it cannot be combined with the built-in Redis selection.

Applications constructing the server directly can use the built-in provider without any import-path configuration:

```python
from hayhooks.a2a import RedisTaskStoreProvider
from hayhooks.server.a2a import A2ARuntime, create_a2a_app

runtime = A2ARuntime(
    task_store_provider=RedisTaskStoreProvider(
        redis_url="redis://localhost:6379/0",
        key_prefix="my-app:a2a",
    )
)
app = create_a2a_app(runtime=runtime)
```

For example, the following user-owned `task_store_provider.py` uses the SDK's optional SQLAlchemy store. The owner resolver includes the agent name so agents sharing the table cannot list or retrieve one another's tasks:

```python
import os
from functools import partial

from a2a.server.context import ServerCallContext
from a2a.server.owner_resolver import resolve_user_scope
from a2a.server.tasks import DatabaseTaskStore, TaskStore
from hayhooks.a2a import TaskStoreProvider
from sqlalchemy.ext.asyncio import create_async_engine


def resolve_agent_owner(agent_name: str, context: ServerCallContext) -> str:
    return f"{agent_name}:{resolve_user_scope(context)}"


class ProjectTaskStoreProvider(TaskStoreProvider):
    def __init__(self) -> None:
        self.engine = create_async_engine(os.environ["A2A_DATABASE_URL"])
        self.stores: dict[str, TaskStore] = {}

    def create_task_store(self, agent_name: str) -> TaskStore:
        if agent_name not in self.stores:
            self.stores[agent_name] = DatabaseTaskStore(
                engine=self.engine,
                owner_resolver=partial(resolve_agent_owner, agent_name),
            )
        return self.stores[agent_name]

    async def close(self) -> None:
        await self.engine.dispose()
```

Install the backend dependencies in the application environment and select the provider by its `module:ClassName` import path:

```bash
pip install "a2a-sdk[postgresql]"
export A2A_DATABASE_URL="postgresql+asyncpg://user:password@localhost/a2a"
export HAYHOOKS_A2A_TASK_STORE_PROVIDER="task_store_provider:ProjectTaskStoreProvider"
hayhooks a2a run
```

The class loaded by the CLI must have a no-argument constructor. It can read its own settings from environment variables, as above. If its module is outside the current Python path, use `--additional-python-path`. The equivalent CLI option is:

```bash
hayhooks a2a run \
  --additional-python-path /path/to/project \
  --task-store-provider task_store_provider:ProjectTaskStoreProvider
```

Applications constructing the server directly can pass an already configured provider without using an import path:

```python
from hayhooks.server.a2a import A2ARuntime, create_a2a_app

runtime = A2ARuntime(task_store_provider=ProjectTaskStoreProvider())
app = create_a2a_app(runtime=runtime)
```

Persisting task records alone does not make execution recoverable. For durable Agents, use Redis durable execution
(the default) and configure the A2A task store for the task history retention your clients need.

## Native A2A wrappers

Use the native mode when an agent needs the full A2A lifecycle, such as custom artifacts, progress states, `input-required`, or specialized cancellation behavior.

```python
from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor
from a2a.server.tasks import TaskUpdater

from hayhooks.a2a import A2APipelineWrapper


class RepositoryAgentExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        result = "Repository inspected"
        await updater.add_artifact([new_text_part(result)], name="response", last_chunk=True)
        await updater.complete()

    async def cancel(self, context, event_queue):
        task = context.current_task
        if task is not None:
            await TaskUpdater(event_queue, task.id, task.context_id).cancel()


class PipelineWrapper(A2APipelineWrapper):
    def setup(self):
        self.pipeline = object()

    def create_a2a_agent_executor(self):
        return RepositoryAgentExecutor()
```

The `object()` in this minimal lifecycle snippet is deliberate: native A2A execution lives in the `AgentExecutor`, and
a Haystack pipeline is optional. For the managed alternative, the
[long-running example](https://github.com/deepset-ai/hayhooks/tree/main/examples/a2a_long_running) declares a durable
Haystack Agent and lets Hayhooks supply the executor, recovery, and progress projection.

Native executors receive the official SDK `RequestContext` and `EventQueue`. They are responsible for emitting the initial `Task` for a new task before status or artifact updates. To pause for more user input, call `TaskUpdater.requires_input()` and return; the SDK will invoke the same executor instance again when the client sends another message for the same task. A follow-up may provide only `taskId`: Hayhooks infers `contextId` from the stored task and rejects a mismatching pair.

A native executor that owns background execution resources may also implement asynchronous `start()` and `close()` methods. Hayhooks detects this optional lifecycle structurally: it calls `start()` during A2A application startup and `close()` before closing the configured task-store provider. This allows an executor to run a managed in-process scheduler or Redis Stream consumer without a separate service. Ordinary `AgentExecutor` implementations need no changes.

### Durable Haystack Agents

For the managed mode, set `durable = True` on an `A2APipelineWrapper` with a Haystack 3 `Agent`. Hayhooks creates the
execution record using the A2A task ID, captures public Agent state at model/tool boundaries, and projects safe
progress, waiting, completion, failure, and cancellation states back to the A2A task.

The execution record is authoritative; the A2A Task is its persisted client-facing projection. This is why the two
records have separate retention settings, and why a persistent A2A task store alone cannot recover interrupted work.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator

from hayhooks import A2APipelineWrapper


class PipelineWrapper(A2APipelineWrapper):
    durable = True

    def setup(self) -> None:
        self.pipeline = Agent(chat_generator=OpenAIChatGenerator(), tools=[])
```

The wrapper does not create an executor, worker, record, queue, or Redis client. A native
`create_a2a_agent_executor()` still takes precedence when custom protocol handling is needed. Durable execution uses
Redis by default; set `HAYHOOKS_DURABLE_STORE=memory` only for non-recoverable local development. Concurrent
execution is controlled by `HAYHOOKS_DURABLE_EXECUTION_CONCURRENCY` and requires the Agent, tools, and their shared
dependencies to be concurrency-safe.

Snapshots, validated messages, and internal tool state remain server-side. A restarted Hayhooks process reclaims
incomplete Redis work from its last safe checkpoint. Tool effects before a checkpoint may be replayed, so tools should
be idempotent. See the [durable A2A example](https://github.com/deepset-ai/hayhooks/tree/main/examples/a2a_long_running).

## Inspecting agents with a2a-inspector

The official [a2a-inspector](https://github.com/a2aproject/a2a-inspector) is a web UI to connect to, inspect, and validate A2A agents — fetch the agent card, chat with the agent, and watch the raw protocol events. Point it at an agent's base URL, e.g. `http://localhost:1418/weather_agent`.

## Multi-agent example

See [examples/a2a_multi_agent](https://github.com/deepset-ai/hayhooks/tree/main/examples/a2a_multi_agent) for a complete demo with two agents — each with its own MCP tools — where one agent delegates to the other over A2A.

## Current limitations

- **Process-owned execution**: execution pauses while the A2A server is offline. Ordinary executors lose active work on restart; durable Agents and lifecycle-aware native executors can persist checkpoints and reclaim incomplete work after Hayhooks starts again.
- **Automatic task-store selection**: `auto` selects Redis only for Redis-backed durable Agents. Explicit `memory` stays process-local. A persistent task store preserves the protocol projection, but does not recover interrupted execution by itself.
- **Push notifications**: push notification delivery is not enabled yet, and agent cards do not advertise it.
- **Static agents list**: A2A routes are built from the registry at startup. Pipelines deployed or undeployed at runtime require restarting `hayhooks a2a run`.
- **Path-prefixed agent cards**: one server hosts many agents, so cards live under `/{pipeline_name}/.well-known/agent-card.json` instead of the domain root. If a consumer requires strict root-level discovery, run one A2A server instance per agent (separate `--pipelines-dir` and `--port`).
- **Cancellation is cooperative**: async chat wrappers, durable Agents, and native executors can observe cancellation. A durable A2A task reports cancellation requested first and becomes A2A canceled only after the execution record is terminal canceled. Synchronous work cannot be forcibly interrupted and retains its fenced claim until it returns.
