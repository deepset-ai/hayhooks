# OpenAI Compatibility

Hayhooks provides OpenAI-compatible endpoints for Haystack pipelines and agents, enabling integration with OpenAI-compatible tools and frameworks.

!!! tip "Open WebUI Integration"
    Looking to integrate with Open WebUI? Check out the complete [Open WebUI Integration](openwebui-integration.md) guide for detailed setup instructions, event handling, and advanced features.

## Overview

Hayhooks can automatically generate OpenAI-compatible endpoints if you implement the appropriate methods in your pipeline wrapper. This makes Hayhooks compatible with any OpenAI-compatible client or tool, including chat interfaces, agent frameworks, and custom applications.

Hayhooks supports two OpenAI API surfaces:

- **[Chat Completions API](https://platform.openai.com/docs/api-reference/chat)** (`/v1/chat/completions`) -- implement `run_chat_completion` or `run_chat_completion_async`
- **[Responses API](https://platform.openai.com/docs/api-reference/responses)** (`/v1/responses`) -- implement `run_response` or `run_response_async`

Both APIs are available simultaneously. A pipeline wrapper can implement one or both.

## Key Features

- **Automatic Endpoint Generation**: OpenAI-compatible endpoints are created automatically
- **Streaming Support**: Real-time streaming responses for chat interfaces
- **Async Support**: High-performance async chat completion and responses
- **Responses API**: Full support for the OpenAI Responses API with streaming named SSE events
- **Files API**: Upload files via `/v1/files` for use with the Responses API
- **Multiple Integration Options**: Works with various OpenAI-compatible clients
- **Open WebUI Ready**: Full support for [Open WebUI](openwebui-integration.md) with events and tool call interception

## Implementation

### Basic Chat Completion

```python
from pathlib import Path
from typing import Union, Generator
from haystack import Pipeline
from hayhooks import get_last_user_message, BasePipelineWrapper, log

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Initialize your pipeline
        pipeline_yaml = (Path(__file__).parent / "pipeline.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str | Generator:
        log.trace("Running pipeline with model: {}, messages: {}, body: {}", model, messages, body)

        question = get_last_user_message(messages)
        log.trace("Question: {}", question)

        # Pipeline run, returns a string
        result = self.pipeline.run({"prompt": {"query": question}})
        return result["llm"]["replies"][0]
```

### Async Chat Completion with Streaming

```python
from collections.abc import AsyncGenerator

from hayhooks import async_streaming_generator, get_last_user_message, log

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Initialize async pipeline
        pipeline_yaml = (Path(__file__).parent / "pipeline.yml").read_text()
        self.pipeline = AsyncPipeline.loads(pipeline_yaml)

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        log.trace("Running pipeline with model: {}, messages: {}, body: {}", model, messages, body)

        question = get_last_user_message(messages)
        log.trace("Question: {}", question)

        # Async streaming pipeline run
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt": {"query": question}},
        )
```

## Method Signatures

### run_chat_completion(...)

```python
def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str | Generator:
    """
    Run the pipeline for OpenAI-compatible chat completion.

    Args:
        model: The pipeline name
        messages: List of messages in OpenAI format
        body: Full request body with additional parameters

    Returns:
        str: Non-streaming response
        Generator: Streaming response generator
    """
```

### run_chat_completion_async(...)

```python
async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> str | AsyncGenerator:
    """
    Async version of run_chat_completion.

    Args:
        model: The pipeline name
        messages: List of messages in OpenAI format
        body: Full request body with additional parameters

    Returns:
        str: Non-streaming response
        AsyncGenerator: Streaming response generator
    """
```

## Generated Endpoints

Hayhooks automatically creates the following OpenAI-compatible endpoints:

### Models

- `/v1/models` - List all deployed pipelines
- `/models` - Alias for `/v1/models`

```bash
curl http://localhost:1416/v1/models
```

### Chat Completions

- `/v1/chat/completions` - [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)
- `/chat/completions` - Alias for `/v1/chat/completions`

```json
{
  "model": "pipeline_name",
  "messages": [
    {"role": "user", "content": "Your message"}
  ],
  "stream": false
}
```

### Responses API

- `/v1/responses` - [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses/create)
- `/responses` - Alias for `/v1/responses`

```json
{
  "model": "pipeline_name",
  "input": [
    {"role": "user", "type": "message", "content": [
      {"type": "input_text", "text": "Your message"}
    ]}
  ],
  "stream": false
}
```

### Files API

- `/v1/files` - [OpenAI Files API](https://platform.openai.com/docs/api-reference/files/create)
- `/files` - Alias for `/v1/files`

```bash
curl http://localhost:1416/v1/files \
  -F "file=@document.pdf" \
  -F "purpose=user_data"
```

!!! note
    By default, the Files API returns file metadata (id, filename, size) but does not persist file bytes. See the [file store example](../examples/overview.md) for how to implement custom file storage.

## Streaming Responses

### Streaming Generator

```python
from hayhooks import streaming_generator

def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> Generator:
    question = get_last_user_message(messages)

    return streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt": {"query": question}},
    )
```

### Async Streaming Generator

```python
from hayhooks import async_streaming_generator

async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
    question = get_last_user_message(messages)

    return async_streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt": {"query": question}},
    )
```

## Responses API

The [Responses API](https://platform.openai.com/docs/api-reference/responses) is an alternative to Chat Completions that uses named SSE events for streaming and supports a richer input format. Implement `run_response` or `run_response_async` in your pipeline wrapper to enable it.

### Basic Response

```python
from hayhooks import BasePipelineWrapper, get_last_user_input_text, log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline.loads(...)

    def run_response(self, model: str, input_items: list[dict], body: dict) -> str:
        question = get_last_user_input_text(input_items)
        result = self.pipeline.run({"prompt": {"query": question}})
        return result["llm"]["replies"][0]
```

### Sync Streaming Response

```python
from collections.abc import Generator

from hayhooks import BasePipelineWrapper, get_last_user_input_text, streaming_generator


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline.loads(...)

    def run_response(self, model: str, input_items: list[dict], body: dict) -> Generator:
        question = get_last_user_input_text(input_items)
        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt": {"query": question}},
        )
```

### Async Streaming Response

```python
from collections.abc import AsyncGenerator

from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_input_text


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = AsyncPipeline.loads(...)

    async def run_response_async(self, model: str, input_items: list[dict], body: dict) -> AsyncGenerator:
        question = get_last_user_input_text(input_items)
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt": {"query": question}},
        )
```

### Agent with Tools

Haystack Agents work with the Responses API too. The agent handles tool calling internally — the client just sends a question and gets an answer back. This is useful for clients like Codex CLI that don't support the `/v1/files` upload flow.

```python
from collections.abc import AsyncGenerator
from pathlib import Path

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.tools import Tool

from hayhooks import BasePipelineWrapper, async_streaming_generator


def read_file(path: str) -> str:
    return Path(path).expanduser().resolve().read_text()

read_file_tool = Tool(
    name="read_file",
    description="Read a text file from disk given its path.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    function=read_file,
)


def _input_items_to_chat_messages(input_items: list[dict]) -> list[ChatMessage]:
    """Convert Responses API input items to Haystack ChatMessage objects.

    Note: ``ChatMessage.from_openai_dict_format`` does not work with Responses
    API input items — use this helper instead.
    """
    messages: list[ChatMessage] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        text = content if isinstance(content, str) else ""
        if isinstance(content, list):
            text = "\n".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("text")
            )
        if not text:
            continue
        if role == "user":
            messages.append(ChatMessage.from_user(text))
        elif role in ("system", "developer"):
            messages.append(ChatMessage.from_system(text))
        elif role == "assistant":
            messages.append(ChatMessage.from_assistant(text))
    return messages


async def _strip_tool_calls(gen: AsyncGenerator) -> AsyncGenerator:
    """Filter internal Agent tool calls from the stream.

    Without this, fastapi-openai-compat translates StreamingChunk.tool_calls
    into SSE function-call events. Agentic clients (e.g. Codex CLI) would
    interpret those as client-side calls and loop forever.
    """
    async for chunk in gen:
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            if hasattr(chunk, "content") and chunk.content:
                yield StreamingChunk(content=chunk.content)
        else:
            yield chunk


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt="You are a helpful assistant that can read files.",
            tools=[read_file_tool],
        )

    async def run_response_async(self, model: str, input_items: list[dict], body: dict) -> AsyncGenerator:
        messages = _input_items_to_chat_messages(input_items)
        gen = async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={"messages": messages},
        )
        return _strip_tool_calls(gen)
```

!!! warning "Server-side tool calls and agentic clients"
    When the Agent handles tools internally, you **must** filter `tool_calls` from the streaming chunks with `_strip_tool_calls` (or equivalent). Otherwise, `fastapi-openai-compat` emits SSE function-call events that agentic clients like Codex CLI interpret as client-side calls — causing an infinite request loop.

See the [responses_with_file_upload](../examples/overview.md) example for a complete implementation with file reading and CWD detection.

### Responses API Method Signatures

#### run_response(...)

```python
def run_response(self, model: str, input_items: list[dict], body: dict) -> str | Generator:
    """
    Handle an OpenAI Responses API request.

    Args:
        model: The pipeline name
        input_items: Normalized input items in OpenAI Responses API format
        body: Full request body with additional parameters (temperature, tools, instructions, etc.)

    Returns:
        str: Non-streaming response
        Generator: Streaming response generator
    """
```

#### run_response_async(...)

```python
async def run_response_async(self, model: str, input_items: list[dict], body: dict) -> str | AsyncGenerator:
    """
    Async version of run_response.

    Args:
        model: The pipeline name
        input_items: Normalized input items in OpenAI Responses API format
        body: Full request body with additional parameters

    Returns:
        str: Non-streaming response
        AsyncGenerator: Streaming response generator
    """
```

### Input Items

The `input_items` parameter contains normalized input items. The library converts string shorthand to a message item and `None` to an empty list. Common input item shapes:

```python
# User text message
{"type": "message", "role": "user", "content": [
    {"type": "input_text", "text": "What is Haystack?"}
]}

# Function call output (tool result)
{"type": "function_call_output", "call_id": "call_abc", "output": "72°F"}
```

Hayhooks provides helpers for working with input items:

- `get_last_user_input_text(input_items)` — extract the last user text (similar to `get_last_user_message(messages)` for chat completions)
- `get_input_files(input_items)` — extract all `input_file` content parts as a list of dicts, each containing at least `file_id`

## Using Hayhooks with Haystack's OpenAIChatGenerator

Hayhooks' OpenAI-compatible endpoints can be used as a backend for Haystack's `OpenAIChatGenerator`, enabling you to create pipelines that consume other Hayhooks-deployed pipelines:

```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage

# Connect to a Hayhooks-deployed pipeline
client = OpenAIChatGenerator(
    model="chat_with_website",  # Your deployed pipeline name
    api_key=Secret.from_token("not-used"),  # Hayhooks doesn't require authentication
    api_base_url="http://localhost:1416/v1/",
    streaming_callback=lambda chunk: print(chunk.content, end="")
)

# Use it like any OpenAI client
result = client.run([ChatMessage.from_user("What is Haystack?")])
print(result["replies"][0].content)
```

This enables powerful use cases:

- **Pipeline Composition**: Chain multiple Hayhooks pipelines together
- **Testing**: Test your pipelines using Haystack's testing tools
- **Hybrid Deployments**: Mix local and remote pipeline execution

!!! warning "Limitations"
    If you customize your Pipeline wrapper to emit [Open WebUI Events](../features/openwebui-integration.md#open-webui-events), it may break out-of-the-box compatibility with Haystack's `OpenAIChatGenerator`.

## Examples

### Sync Chat Pipeline (Non-Streaming)

```python
class SyncChatWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        template = [ChatMessage.from_user("Answer: {{query}}")]
        chat_prompt_builder = ChatPromptBuilder(template=template)
        llm = OpenAIChatGenerator(model="gpt-4o-mini")

        self.pipeline = Pipeline()
        self.pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("chat_prompt_builder.prompt", "llm.messages")

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str:
        question = get_last_user_message(messages)
        result = self.pipeline.run({"chat_prompt_builder": {"query": question}})
        return result["llm"]["replies"][0].content
```

### Async Streaming Pipeline

```python
class AsyncStreamingWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack import AsyncPipeline
        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        template = [ChatMessage.from_user("Answer: {{query}}")]
        chat_prompt_builder = ChatPromptBuilder(template=template)
        llm = OpenAIChatGenerator(model="gpt-4o")

        self.pipeline = AsyncPipeline()
        self.pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("chat_prompt_builder.prompt", "llm.messages")

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        question = get_last_user_message(messages)
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"chat_prompt_builder": {"query": question}},
        )
```

## Request Parameters

The OpenAI-compatible endpoints support standard parameters from the `body` argument:

```python
def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str:
    # Access additional parameters
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 150)
    stream = body.get("stream", False)

    # Use them in your pipeline
    result = self.pipeline.run({
        "llm": {
            "generation_kwargs": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
    })
    return result["llm"]["replies"][0].content
```

**Common parameters include:**

- `temperature`: Controls randomness (0.0 to 2.0)
- `max_tokens`: Maximum number of tokens to generate
- `stream`: Enable streaming responses
- `stop`: Stop sequences
- `top_p`: Nucleus sampling parameter

See the OpenAI API reference for [Chat Completions](https://platform.openai.com/docs/api-reference/chat/create) and [Responses](https://platform.openai.com/docs/api-reference/responses/create) for the complete list of parameters.

## Next Steps

- [Open WebUI Integration](openwebui-integration.md) - Use Hayhooks with Open WebUI chat interface
- [Examples](../examples/overview.md) - Working examples and use cases
- [File Upload Support](file-upload-support.md) - Handle file uploads in pipelines
