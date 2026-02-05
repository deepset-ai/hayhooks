# Chainlit Integration

Hayhooks provides optional integration with [Chainlit](https://chainlit.io/), allowing you to embed a chat UI directly within your Hayhooks server. This provides a zero-configuration frontend for interacting with your deployed Haystack pipelines.

## Overview

The Chainlit integration offers:

- **Single Deployment**: Run both backend and frontend in one process/container
- **Zero Configuration**: Works out-of-the-box with Hayhooks' OpenAI-compatible endpoints
- **Streaming Support**: Real-time streaming responses in the chat interface
- **Pipeline Selection**: Automatically discovers and lists deployed pipelines

## Installation

Install Hayhooks with the `ui` extra:

```bash
pip install "hayhooks[ui]"
```

## Quick Start

### Using CLI

The simplest way to enable the Chainlit UI is via the `--with-ui` flag:

```bash
hayhooks run --with-ui
```

This starts Hayhooks with the embedded Chainlit UI available at `http://localhost:1416/ui`.

### Custom UI Path

You can customize the URL path where the UI is mounted:

```bash
hayhooks run --with-ui --ui-path /chat
```

Now the UI will be available at `http://localhost:1416/chat`.

### Using Environment Variables

You can also configure the UI via environment variables:

```bash
export HAYHOOKS_UI_ENABLED=true
export HAYHOOKS_UI_PATH=/ui

hayhooks run
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HAYHOOKS_UI_ENABLED` | Enable/disable the Chainlit UI | `false` |
| `HAYHOOKS_UI_PATH` | URL path where UI is mounted | `/ui` |
| `HAYHOOKS_UI_APP` | Custom Chainlit app file path | (uses default) |

### Chainlit App Configuration

The default Chainlit app also supports these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HAYHOOKS_BASE_URL` | Base URL for Hayhooks API calls | `http://localhost:1416` |
| `HAYHOOKS_DEFAULT_MODEL` | Default pipeline to use | (auto-select) |

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Hayhooks Server                       │
│                                                          │
│  ┌──────────────────┐    ┌─────────────────────────┐    │
│  │   Chainlit UI    │───▶│  /v1/chat/completions   │    │
│  │   (mounted at    │    │  (OpenAI-compatible)    │    │
│  │    /ui)          │    └─────────────────────────┘    │
│  └──────────────────┘              │                     │
│                                    ▼                     │
│                         ┌─────────────────────────┐     │
│                         │   Haystack Pipelines    │     │
│                         │   (with chat support)   │     │
│                         └─────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

The Chainlit UI is mounted as a FastAPI sub-application and communicates with your pipelines through Hayhooks' OpenAI-compatible endpoints. This means:

1. Your pipelines must implement `run_chat_completion` or `run_chat_completion_async`
2. The UI automatically discovers available pipelines via `/v1/models`
3. Streaming is supported out-of-the-box

## Example: Complete Setup

### 1. Create a Pipeline Wrapper

```python
# pipelines/my_chat/pipeline_wrapper.py
from typing import Generator

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from hayhooks import BasePipelineWrapper, get_last_user_message, streaming_generator


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        template = [ChatMessage.from_user("Answer this question: {{query}}")]
        chat_prompt_builder = ChatPromptBuilder(template=template)
        llm = OpenAIChatGenerator(model="gpt-4o-mini")

        self.pipeline = Pipeline()
        self.pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("chat_prompt_builder.prompt", "llm.messages")

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> Generator:
        question = get_last_user_message(messages)
        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"chat_prompt_builder": {"query": question}},
        )
```

### 2. Run Hayhooks with UI

```bash
hayhooks run --with-ui --pipelines-dir ./pipelines
```

### 3. Open the UI

Navigate to `http://localhost:1416/ui` in your browser. You'll see your deployed pipeline and can start chatting!

## Custom Chainlit App

You can provide your own Chainlit app for more customization:

```bash
hayhooks run --with-ui
```

With environment variable:

```bash
export HAYHOOKS_UI_APP=/path/to/my_chainlit_app.py
hayhooks run --with-ui
```

### Example Custom App

```python
# my_chainlit_app.py
import os
import chainlit as cl
import httpx

HAYHOOKS_URL = os.getenv("HAYHOOKS_BASE_URL", "http://localhost:1416")

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome! How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    response_msg = cl.Message(content="")
    await response_msg.send()

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{HAYHOOKS_URL}/v1/chat/completions",
            json={
                "model": "my_pipeline",  # Your pipeline name
                "messages": [{"role": "user", "content": message.content}],
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    import json
                    chunk = json.loads(line[6:])
                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        await response_msg.stream_token(content)

    await response_msg.update()
```

## Comparison with Other Frontends

| Feature | Chainlit (Embedded) | Open WebUI | LibreChat |
|---------|---------------------|------------|-----------|
| Deployment | Single process | Separate container | Multiple containers |
| Setup complexity | Low | Medium | High |
| Dependencies | `pip install hayhooks[ui]` | Docker | Docker + MongoDB + Redis |
| Multi-user auth | Basic | Yes | Yes |
| Conversation history | Session-based | Persistent | Persistent |
| Best for | Quick demos, simple deployments | Production chat UI | Enterprise features |

## Troubleshooting

### UI Not Loading

1. Ensure Chainlit is installed: `pip install "hayhooks[ui]"`
2. Check that `--with-ui` flag is set or `HAYHOOKS_UI_ENABLED=true`
3. Verify the UI path in logs (default: `/ui`)

### No Pipelines Available

The UI requires at least one deployed pipeline with chat completion support:

1. Ensure your pipeline implements `run_chat_completion` or `run_chat_completion_async`
2. Check that pipelines are deployed: `curl http://localhost:1416/v1/models`

### Streaming Not Working

1. Ensure your pipeline returns a `Generator` or `AsyncGenerator`
2. Use `streaming_generator` or `async_streaming_generator` helpers
3. Check browser console for WebSocket errors

## Limitations

- **Session-based history**: Conversation history is stored in the browser session, not persisted
- **Single process**: The UI runs in the same process as Hayhooks, which may not be ideal for high-traffic scenarios
- **Basic authentication**: For production use with authentication, consider Open WebUI or LibreChat

## Next Steps

- [OpenAI Compatibility](openai-compatibility.md) - Learn about chat completion implementation
- [Open WebUI Integration](openwebui-integration.md) - For a more feature-rich frontend
- [Examples](../examples/overview.md) - Working pipeline examples
