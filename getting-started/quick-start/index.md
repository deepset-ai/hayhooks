# Quick Start

Serve a Haystack agent through a typed HTTP endpoint, then inspect its execution in the terminal dashboard.

## Prerequisites

- Python 3.10 through 3.14
- An OpenAI API key for this example

The shell commands below use Bash or zsh. In PowerShell, set the key with `$env:OPENAI_API_KEY = "your-api-key"`.

## Installation

In your Python environment, install Hayhooks:

```
pip install hayhooks
```

No repository checkout is needed. See [Installation](https://deepset-ai.github.io/hayhooks/getting-started/installation/index.md) for other installation options.

## Basic Usage

### 1. Start Hayhooks

Set the API key in the server's terminal, then start Hayhooks:

```
export OPENAI_API_KEY="your-api-key"
hayhooks run
```

The server listens on `http://localhost:1416`. Leave it running and use a second terminal in the same Python environment for the following commands.

### 2. Create and deploy an agent

Create a directory named `my_agent_dir` and save this as `my_agent_dir/pipeline_wrapper.py`:

```
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        )

    async def run_api_async(self, question: str) -> str:
        result = await self.pipeline.run_async(messages=[ChatMessage.from_user(question)])
        return result["last_message"].text
```

Deploy the wrapper from the directory that contains `my_agent_dir`:

```
hayhooks pipeline deploy-files -n my_agent ./my_agent_dir
```

Hayhooks creates `POST /my_agent/run` and validates its `question` input from the type hint.

### 3. Check Status

```
hayhooks status
```

The output includes the `my_agent` deployment. Open [the API docs](http://localhost:1416/docs) to inspect and try its endpoint.

### 4. Run Your Pipeline

Call the agent using one of these clients:

```
curl http://localhost:1416/my_agent/run \
  -H 'Content-Type: application/json' \
  -d '{"question": "What can you do?"}'
```

```
import requests

response = requests.post(
    "http://localhost:1416/my_agent/run",
    json={"question": "What can you do?"},
    timeout=60,
)
response.raise_for_status()
print(response.json())
```

```
hayhooks pipeline run my_agent --param 'question=What can you do?'
```

The response contains the agent's answer. Each request in this example starts a new conversation.

### 5. Inspect the run

On macOS or Linux, open the dashboard in the second terminal under the same OS account as the server:

```
pip install "hayhooks[tui]"
hayhooks dashboard
```

Press `e` for executions or `v` for traces, then Enter for details. Local capture is enabled by default, so the run remains visible until it is evicted from the recent buffer or the server restarts.

For Windows or remote access, follow the [dashboard connection guide](https://deepset-ai.github.io/hayhooks/features/terminal-dashboard/#connect-to-a-remote-server).

## Quick Start with Docker Compose

For a container setup with Open WebUI, see [Quick Start with Docker Compose](https://deepset-ai.github.io/hayhooks/getting-started/quick-start-docker/index.md).

## Next Steps

- [OpenAI Compatibility](https://deepset-ai.github.io/hayhooks/features/openai-compatibility/index.md) - Add streaming chat or Responses endpoints
- [MCP Support](https://deepset-ai.github.io/hayhooks/features/mcp-support/index.md) - Expose the agent as a tool for MCP clients
- [A2A Support](https://deepset-ai.github.io/hayhooks/features/a2a-support/index.md) - Add chat support and expose the agent to other agents
- [Terminal Dashboard](https://deepset-ai.github.io/hayhooks/features/terminal-dashboard/index.md) - Inspect deployments, executions, and component traces
- [Examples](https://deepset-ai.github.io/hayhooks/examples/overview/index.md) - Explore agents with tools and complete pipelines
