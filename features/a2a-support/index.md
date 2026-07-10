# A2A Support

Hayhooks supports the [A2A protocol](https://a2a-protocol.org) (Agent2Agent) and can act as an A2A server, exposing deployed pipelines and agents as A2A agents that other agents can discover and delegate tasks to.

A2A complements [MCP support](https://deepset-ai.github.io/hayhooks/features/mcp-support/index.md): MCP exposes pipelines as **tools** for an agent to call (agent→tool), while A2A exposes them as **agents** that other agents talk to (agent→agent).

## Overview

The Hayhooks A2A Server:

- Exposes every deployed pipeline that implements `run_chat_completion` or `run_chat_completion_async` as an A2A agent
- Serves a per-agent [Agent Card](#agent-cards) for discovery, auto-generated from the pipeline and customizable from the wrapper
- Implements the JSON-RPC protocol binding of the [A2A specification](https://a2a-protocol.org/latest/specification/) (v1.0), including SSE streaming
- Streams pipeline output incrementally as task artifact updates

## Requirements

- Install with `pip install hayhooks[a2a]` (uses the official [a2a-sdk](https://github.com/a2aproject/a2a-python))

## Getting Started

### Install with A2A Support

```
pip install hayhooks[a2a]
```

### Start the A2A Server

```
hayhooks a2a run
```

This starts the A2A server on `HAYHOOKS_A2A_HOST:HAYHOOKS_A2A_PORT` (default: `localhost:1418`), deploying pipelines from `HAYHOOKS_PIPELINES_DIR` (or `--pipelines-dir`).

### Configuration

Environment variables for the A2A server:

```
HAYHOOKS_A2A_HOST=localhost         # A2A server host
HAYHOOKS_A2A_PORT=1418              # A2A server port
HAYHOOKS_A2A_EXTERNAL_URL=          # Base URL advertised in agent cards
                                    # (set when behind a reverse proxy)
HAYHOOKS_A2A_V0_3_COMPAT=true       # Also accept A2A spec 0.3 requests
                                    # (used by older clients and tools)
```

## Which pipelines are exposed

A deployed pipeline is exposed as an A2A agent when it implements `run_chat_completion` or `run_chat_completion_async` — the same methods used by the [OpenAI-compatible chat endpoints](https://deepset-ai.github.io/hayhooks/features/openai-compatibility/index.md). No extra method is needed.

To exclude a chat-capable pipeline from A2A, set `skip_a2a` on the wrapper:

```
class PipelineWrapper(BasePipelineWrapper):
    skip_a2a = True
```

## Endpoints

Each exposed pipeline is mounted under its own path prefix:

| Endpoint                                           | Description                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| `GET /{pipeline_name}/.well-known/agent-card.json` | The pipeline's agent card                                                |
| `POST /{pipeline_name}/`                           | JSON-RPC binding (`SendMessage`, `SendStreamingMessage`, `GetTask`, ...) |
| `GET /status`                                      | Server status and the list of exposed agents                             |

For example, with a deployed `weather_agent` pipeline:

```
curl http://localhost:1418/weather_agent/.well-known/agent-card.json
```

## Agent Cards

Agent cards are generated automatically: the card name is the pipeline name, the description comes from the pipeline's registry metadata, and a single default skill is created. Override any of it with the `a2a_card` class attribute:

```
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

```
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

```
curl -s http://localhost:1418/weather_agent/ \
  -H "Content-Type: application/json" -H "A2A-Version: 1.0" \
  -d '{"jsonrpc": "2.0", "id": "1", "method": "SendMessage",
       "params": {"message": {"messageId": "m1", "role": "ROLE_USER",
                              "parts": [{"text": "Weather in Berlin?"}]}}}'
```

## Task lifecycle and streaming

Each request is handled as an A2A task:

1. A `Task` is created from the incoming message.
1. The task transitions to `working` and the pipeline's chat completion method runs.
1. Pipeline output is emitted as a single `response` artifact. Streaming results (generators returned by `streaming_generator` / `async_streaming_generator`) are emitted incrementally as artifact chunk updates, so `SendStreamingMessage` clients receive text as it is produced.
1. The task ends in `completed` (or `failed`, with the error in the status message — enable `HAYHOOKS_SHOW_TRACEBACKS` to include tracebacks).

## Inspecting agents with a2a-inspector

The official [a2a-inspector](https://github.com/a2aproject/a2a-inspector) is a web UI to connect to, inspect, and validate A2A agents — fetch the agent card, chat with the agent, and watch the raw protocol events. Point it at an agent's base URL, e.g. `http://localhost:1418/weather_agent`.

## Multi-agent example

See [examples/a2a_multi_agent](https://github.com/deepset-ai/hayhooks/tree/main/examples/a2a_multi_agent) for a complete demo with two agents — each with its own MCP tools — where one agent delegates to the other over A2A.

## Current limitations

- **Request-bound task execution**: Hayhooks currently treats A2A as a chat-shaped bridge. Each task runs inside the request handler by calling `run_chat_completion` / `run_chat_completion_async`, so non-streaming `SendMessage` returns after the task has completed or failed. This means detached task execution via [`returnImmediately`](https://a2a-protocol.org/latest/specification/#322-sendmessageconfiguration), [`input-required`](https://a2a-protocol.org/latest/specification/#63-multi-turn-interaction) pauses, and [push notification delivery](https://a2a-protocol.org/latest/specification/#353-push-notification-delivery) are not supported yet.
- **Static agents list**: A2A routes are built from the registry at startup. Pipelines deployed or undeployed at runtime require restarting `hayhooks a2a run`.
- **In-memory task store**: task state is kept in memory and lost on restart.
- **Path-prefixed agent cards**: one server hosts many agents, so cards live under `/{pipeline_name}/.well-known/agent-card.json` instead of the domain root. If a consumer requires strict root-level discovery, run one A2A server instance per agent (separate `--pipelines-dir` and `--port`).
- **Cancellation is best-effort**: cancelling a task marks it canceled but does not interrupt a running pipeline.
