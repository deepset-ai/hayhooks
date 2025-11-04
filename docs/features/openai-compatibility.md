# OpenAI Compatibility

Hayhooks provides OpenAI-compatible endpoints for Haystack pipelines and agents, enabling integration with OpenAI-compatible tools and frameworks.

!!! tip "Open WebUI Integration"
    Looking to integrate with Open WebUI? Check out the complete [Open WebUI Integration](openwebui-integration.md) guide for detailed setup instructions, event handling, and advanced features.

## Overview

Hayhooks can automatically generate OpenAI-compatible endpoints if you implement the `run_chat_completion` or `run_chat_completion_async` method in your pipeline wrapper. This makes Hayhooks compatible with any OpenAI-compatible client or tool, including chat interfaces, agent frameworks, and custom applications.

## Key Features

- **Automatic Endpoint Generation**: OpenAI-compatible endpoints are created automatically
- **Streaming Support**: Real-time streaming responses for chat interfaces
- **Async Support**: High-performance async chat completion
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

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> Union[str, Generator]:
        log.trace("Running pipeline with model: {}, messages: {}, body: {}", model, messages, body)

        question = get_last_user_message(messages)
        log.trace("Question: {}", question)

        # Pipeline run, returns a string
        result = self.pipeline.run({"prompt": {"query": question}})
        return result["llm"]["replies"][0]
```

### Async Chat Completion with Streaming

```python
from typing import AsyncGenerator
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
def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> Union[str, Generator]:
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
async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> Union[str, AsyncGenerator]:
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

When you implement chat completion methods, Hayhooks automatically creates:

### Chat Endpoints

- `/{pipeline_name}/chat` - Direct chat endpoint for a specific pipeline
- `/chat/completions` - OpenAI-compatible endpoint (routes to the model specified in request)
- `/v1/chat/completions` - OpenAI API v1 compatible endpoint

All endpoints support the standard OpenAI chat completion request format:

```json
{
  "model": "pipeline_name",
  "messages": [
    {"role": "user", "content": "Your message"}
  ],
  "stream": false
}
```

### Available Models

Use the `/v1/models` endpoint to list all deployed pipelines that support chat completion:

```bash
curl http://localhost:1416/v1/models
```

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
        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        template = [ChatMessage.from_user("Answer: {{query}}")]
        chat_prompt_builder = ChatPromptBuilder(template=template)
        llm = OpenAIChatGenerator(model="gpt-4o")

        self.pipeline = Pipeline()
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

See the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create) for the complete list of parameters.

## Next Steps

- [Open WebUI Integration](openwebui-integration.md) - Use Hayhooks with Open WebUI chat interface
- [Examples](../examples/overview.md) - Working examples and use cases
- [File Upload Support](file-upload-support.md) - Handle file uploads in pipelines
