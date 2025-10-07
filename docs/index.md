# Hayhooks

**Hayhooks** makes it easy to deploy and serve [Haystack](https://haystack.deepset.ai/) [Pipelines](https://docs.haystack.deepset.ai/docs/pipelines) and [Agents](https://docs.haystack.deepset.ai/docs/agents).

With Hayhooks, you can:

- 📦 **Deploy your Haystack pipelines and agents as REST APIs** with maximum flexibility and minimal boilerplate code.
- 🛠️ **Expose your Haystack pipelines and agents over the MCP protocol**, making them available as tools in AI dev environments like [Cursor](https://cursor.com) or [Claude Desktop](https://claude.ai/download). Under the hood, Hayhooks runs as an [MCP Server](https://modelcontextprotocol.io/docs/concepts/architecture), exposing each pipeline and agent as an [MCP Tool](https://modelcontextprotocol.io/docs/concepts/tools).
- 💬 **Integrate your Haystack pipelines and agents with [Open WebUI](https://openwebui.com)** as OpenAI-compatible chat completion backends with streaming support.
- 🕹️ **Control Hayhooks core API endpoints through chat** - deploy, undeploy, list, or run Haystack pipelines and agents by chatting with [Claude Desktop](https://claude.ai/download), [Cursor](https://cursor.com), or any other MCP client.

[![PyPI - Version](https://img.shields.io/pypi/v/hayhooks.svg)](https://pypi.org/project/hayhooks)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hayhooks.svg)](https://pypi.org/project/hayhooks)
[![Docker image release](https://github.com/deepset-ai/hayhooks/actions/workflows/docker.yml/badge.svg)](https://github.com/deepset-ai/hayhooks/actions/workflows/docker.yml)
[![Tests](https://github.com/deepset-ai/hayhooks/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/hayhooks/actions/workflows/tests.yml)

## Quick Start

### 1. Install Hayhooks

```bash
# Install Hayhooks
pip install hayhooks
```

### 2. Start Hayhooks

```bash
hayhooks run
```

### 3. Create a simple agent

Create a minimal agent wrapper with streaming chat support and a simple HTTP POST API:

```python
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
        result = await self.agent.run_async({"messages": [ChatMessage.from_user(question)]})
        return result["replies"][0].text

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

```bash
hayhooks pipeline deploy-files -n my_agent ./my_agent_dir
```

### 5. Run it

Call the HTTP POST API (`/my_agent/run`):

```bash
curl -X POST http://localhost:1416/my_agent/run \
  -H 'Content-Type: application/json' \
  -d '{"question": "What can you do?"}'
```

Call the OpenAI-compatible chat completion API (streaming enabled):

```bash
curl -X POST http://localhost:1416/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my_agent",
    "messages": [{"role": "user", "content": "What can you do?"}]
  }'
```

Or [integrate it with Open WebUI](features/openai-compatibility.md#open-webui-integration) and start chatting with it!

## Key Features

### 🚀 Easy Deployment

- Deploy Haystack pipelines and agents as REST APIs with minimal setup
- Support for both YAML-based and wrapper-based pipeline deployment
- Automatic OpenAI-compatible endpoint generation

### 🌐 Multiple Integration Options

- **MCP Protocol**: Expose pipelines as MCP tools for use in AI development environments
- **Open WebUI Integration**: Use Hayhooks as a backend for Open WebUI with streaming support
- **OpenAI Compatibility**: Seamless integration with OpenAI-compatible tools and frameworks

### 🔧 Developer Friendly

- CLI for easy pipeline management
- Flexible configuration options
- Comprehensive logging and debugging support
- Custom route and middleware support

### 📁 File Upload Support

- Built-in support for handling file uploads in pipelines
- Perfect for RAG systems and document processing

## Next Steps

- [Quick Start Guide](getting-started/quick-start.md) - Get started with Hayhooks
- [Installation](getting-started/installation.md) - Install Hayhooks and dependencies
- [Configuration](getting-started/configuration.md) - Configure Hayhooks for your needs
- [Examples](examples/overview.md) - Explore example implementations

## Community & Support

- **GitHub**: [deepset-ai/hayhooks](https://github.com/deepset-ai/hayhooks)
- **Issues**: [GitHub Issues](https://github.com/deepset-ai/hayhooks/issues)
- **Documentation**: [Full Documentation](https://deepset-ai.github.io/hayhooks/)

Hayhooks is actively maintained by the [deepset](https://deepset.ai/) team.
