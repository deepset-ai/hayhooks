# Hayhooks

**Serve [Haystack](https://haystack.deepset.ai/) Pipelines and Agents as production APIs: REST, OpenAI-compatible chat, MCP, A2A, and durable executions that survive restarts.**

Write a small Python wrapper around your Pipeline or Agent, and Hayhooks turns it into typed endpoints, streaming chat backends, and agent-to-agent tools. When the work outlives a request, a restart, or a deploy, run it as a **durable execution**: checkpointed in Redis, resumable on any replica, and able to wait for a human.

[Get started](https://deepset-ai.github.io/hayhooks/getting-started/quick-start/index.md) [Durable execution](#durable-execution)

- **REST APIs from Python**

  ______________________________________________________________________

  Every wrapper or YAML pipeline becomes typed, validated endpoints with OpenAPI docs, [file uploads](https://deepset-ai.github.io/hayhooks/features/file-upload-support/index.md), and custom routes.

- **Durable executions**

  ______________________________________________________________________

  Checkpoints, retries, human approval, cancellation, and restart recovery for long-running Pipelines and Agents.

  [Durable execution](https://deepset-ai.github.io/hayhooks/features/durable-execution/index.md)

- **OpenAI-compatible chat**

  ______________________________________________________________________

  Streaming [chat completion](https://deepset-ai.github.io/hayhooks/features/openai-compatibility/index.md) backends for [Open WebUI](https://deepset-ai.github.io/hayhooks/features/openwebui-integration/index.md), or an embedded [Chainlit](https://deepset-ai.github.io/hayhooks/features/chainlit-integration/index.md) UI with `--with-chainlit`.

- **MCP server**

  ______________________________________________________________________

  Each pipeline becomes an [MCP tool](https://deepset-ai.github.io/hayhooks/features/mcp-support/index.md) in [Cursor](https://cursor.com), [Claude Desktop](https://claude.ai/download), or any MCP client.

- **A2A protocol**

  ______________________________________________________________________

  Publish pipelines as [A2A agents](https://deepset-ai.github.io/hayhooks/features/a2a-support/index.md) with auto-generated agent cards, so other agents can delegate tasks to them.

- **Tracing and dashboard**

  ______________________________________________________________________

  OpenTelemetry spans (`hayhooks[tracing]`) for deploy, run, and durable attempts, with a live `/dashboard` via `--with-tracing-dashboard`.

## Quick Start

### 1. Install Hayhooks

```
# Install Hayhooks
pip install hayhooks
```

### 2. Start Hayhooks

```
hayhooks run
```

### 3. Create a simple agent

Create a minimal agent wrapper with streaming chat support and a simple HTTP POST API:

```
from typing import AsyncGenerator
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.components.generators.chat import OpenAIChatGenerator
from hayhooks import BasePipelineWrapper, async_streaming_generator


# Define a Haystack Tool that provides weather information for a given location.
def weather_function(location):
    return f"The weather in {location} is sunny."

weather_tool = Tool(
    name="weather_tool",
    description="Provides weather information for a given location.",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
    function=weather_function,
)

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt="You're a helpful agent",
            tools=[weather_tool],
        )

    # This will create a POST /my_agent/run endpoint
    # `question` will be the input argument and will be auto-validated by a Pydantic model
    async def run_api_async(self, question: str) -> str:
        result = await self.agent.run_async(messages=[ChatMessage.from_user(question)])
        return result["last_message"].text

    # This will create an OpenAI-compatible /chat/completions endpoint
    async def run_chat_completion_async(
        self, model: str, messages: list[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        chat_messages = [
            ChatMessage.from_openai_dict_format(message) for message in messages
        ]

        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={
                "messages": chat_messages,
            },
        )
```

Save as `my_agent_dir/pipeline_wrapper.py`.

### 4. Deploy it

```
hayhooks pipeline deploy-files -n my_agent ./my_agent_dir
```

### 5. Run it

Call the HTTP POST API (`/my_agent/run`):

```
curl -X POST http://localhost:1416/my_agent/run \
  -H 'Content-Type: application/json' \
  -d '{"question": "What can you do?"}'
```

Call the OpenAI-compatible chat completion API (streaming enabled):

```
curl -X POST http://localhost:1416/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my_agent",
    "messages": [{"role": "user", "content": "What can you do?"}]
  }'
```

Or chat with it in the [embedded Chainlit UI](https://deepset-ai.github.io/hayhooks/features/chainlit-integration/index.md) (`hayhooks run --with-chainlit`) or [integrate it with Open WebUI](https://deepset-ai.github.io/hayhooks/features/openwebui-integration/index.md)!

## Durable execution

Some work does not fit in a request: an Agent that researches for ten minutes, a Pipeline that must wait for a reviewer, a job that must not start over because you deployed. Durable execution runs a Pipeline or Agent detached from the request, checkpoints it in Redis, and resumes it from the last checkpoint on any replica.

```
stateDiagram-v2
    direction LR
    [*] --> queued: submit (202)
    queued --> running: worker claims
    running --> waiting: suspend for a human
    waiting --> queued: resume
    running --> queued: retry or crash recovery
    running --> completed
    running --> failed
    running --> canceled
```

| Capability                         | What you get                                                                             |
| ---------------------------------- | ---------------------------------------------------------------------------------------- |
| **Detached runs**                  | Submit returns `202 Accepted` with links to inspect, resume, cancel, and stream          |
| **Restart recovery**               | A crashed or redeployed worker's run is reclaimed and continues from its last checkpoint |
| **Pipeline and Agent checkpoints** | Completed components are not run again; Agent loops continue after their last tool call  |
| **Human in the loop**              | Suspend with a public wait reason and continue with typed, validated resume input        |
| **Retries and cancellation**       | Separate budgets for crashes and application retries; cooperative cancellation           |
| **Live output**                    | Reattachable SSE streams (`Last-Event-ID`) that any replica can serve                    |
| **Multi-replica safety**           | Leases and fencing stop a stale worker from committing after another took over           |

A durable wrapper sets a revision and implements `run_durable_async`. This Agent waits for approval, then researches with its tools:

```
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext


@tool
def search(query: str) -> str:
    """Search the knowledge base."""
    return f"Hayhooks serves Haystack pipelines and agents; results for {query!r}."


class Task(BaseModel):
    question: str


class Approval(BaseModel):
    approved: bool


class Report(BaseModel):
    answer: str


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "research-agent-v1"  # in-flight work stays pinned to this revision
    durable_resume_model = Approval  # typed input for human-in-the-loop waits

    def setup(self) -> None:
        generator = OpenAIResponsesChatGenerator(model="gpt-6-luna", generation_kwargs={"reasoning": {"effort": "high"}})
        self.pipeline = Agent(chat_generator=generator, tools=[search])

    async def run_durable_async(self, context: DurableContext, task: Task) -> Report:
        # Wait for a human. The wait lives in Redis, so it survives restarts and deploys.
        if not context.state.get("approved"):
            decision = context.resume_input  # consumed on first read
            if decision is None:
                await context.suspend({"kind": "approval", "message": f"Research {task.question!r}?"})
            if not Approval.model_validate(decision).approved:
                raise ValueError("research was rejected")
            context.state["approved"] = True

        # Agent state is checkpointed after tool calls: recovery continues the loop.
        result = await context.run_agent_async(messages=[ChatMessage.from_user(task.question)])
        return Report(answer=result["last_message"].text)
```

```
pip install "hayhooks[durable]"
docker run -d -p 6379:6379 redis
# Save the wrapper as ./pipelines/research_agent/pipeline_wrapper.py
HAYHOOKS_DURABLE_MODE=true hayhooks run --pipelines-dir ./pipelines
```

The durable extra installs `redis>=5.0.1,<9`.

```
curl -i http://localhost:1416/research_agent/run-durable \
  -H 'Content-Type: application/json' -d '{"question": "What does Hayhooks do?"}'
```

```
curl -X POST http://localhost:1416/research_agent/executions/EXECUTION_ID/resume \
  -H 'Content-Type: application/json' -d '{"approved": true}'

curl -N http://localhost:1416/research_agent/executions/EXECUTION_ID/stream
```

Try a restart

Stop the server mid-run and start it again: the run continues from its last checkpoint. Not using the Hayhooks server? The same engine [embeds in any FastAPI app](https://deepset-ai.github.io/hayhooks/examples/durable-fastapi/index.md). See the [durable execution guide](https://deepset-ai.github.io/hayhooks/features/durable-execution/index.md) and the [operations guide](https://deepset-ai.github.io/hayhooks/deployment/durable-operations/index.md) for the full contract.

## Next Steps

- [Quick Start Guide](https://deepset-ai.github.io/hayhooks/getting-started/quick-start/index.md) - Get started with Hayhooks
- [Installation](https://deepset-ai.github.io/hayhooks/getting-started/installation/index.md) - Install Hayhooks and dependencies
- [Configuration](https://deepset-ai.github.io/hayhooks/getting-started/configuration/index.md) - Configure Hayhooks for your needs
- [Durable Execution](https://deepset-ai.github.io/hayhooks/features/durable-execution/index.md) - Run Pipelines and Agents beyond the request
- [Examples](https://deepset-ai.github.io/hayhooks/examples/overview/index.md) - Explore example implementations

## Community & Support

- **GitHub**: [deepset-ai/hayhooks](https://github.com/deepset-ai/hayhooks)
- **Issues**: [GitHub Issues](https://github.com/deepset-ai/hayhooks/issues)
- **Documentation**: [Full Documentation](https://deepset-ai.github.io/hayhooks/)

Hayhooks is actively maintained by the [deepset](https://deepset.ai/) team.
