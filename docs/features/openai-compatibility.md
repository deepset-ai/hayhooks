# OpenAI Compatibility

Hayhooks provides seamless OpenAI-compatible endpoints for Haystack pipelines and agents, enabling integration with OpenAI-compatible tools and frameworks.

## Overview

Hayhooks can automatically generate OpenAI-compatible endpoints if you implement the `run_chat_completion` method in your pipeline wrapper. This makes Hayhooks compatible with fully-featured chat interfaces like [Open WebUI](https://openwebui.com/), so you can use it as a backend for your chat interface.

## Key Features

- **Automatic Endpoint Generation**: OpenAI-compatible endpoints are created automatically
- **Streaming Support**: Real-time streaming responses for chat interfaces
- **Async Support**: High-performance async chat completion
- **Multiple Integration Options**: Works with various OpenAI-compatible clients

## Implementation

### Basic Chat Completion

```python
from typing import List, Union, Generator
from haystack import Pipeline
from hayhooks import get_last_user_message, BasePipelineWrapper, log

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Initialize your pipeline
        pipeline_yaml = (Path(__file__).parent / "pipeline.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

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

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        log.trace(f"Running pipeline with model: {model}, messages: {messages}, body: {body}")

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

        # Async streaming pipeline run
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt": {"query": question}},
        )
```

## Method Signatures

### run_chat_completion(...)

```python
def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
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
async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> Union[str, AsyncGenerator]:
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

- `/{pipeline_name}/chat` - Direct chat endpoint
- `/chat/completions` - OpenAI-compatible endpoint
- `/v1/chat/completions` - OpenAI API compatible endpoint

### OpenAPI Schema

The endpoints are automatically documented in the OpenAPI schema:

```yaml
/chat/completions:
  post:
    summary: OpenAI-compatible chat completion
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              model:
                type: string
              messages:
                type: array
                items:
                  $ref: '#/components/schemas/ChatMessage'
              stream:
                type: boolean
                default: false
    responses:
      '200':
        description: Successful response
```

## Open WebUI Integration

For detailed Open WebUI setup and integration, see the [Open WebUI Integration](openwebui-integration.md) guide.

## Streaming Responses

### Streaming Generator

```python
from hayhooks import streaming_generator

def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Generator:
    question = get_last_user_message(messages)

    return streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt": {"query": question}},
    )
```

### Async Streaming Generator

```python
from hayhooks import async_streaming_generator

async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
    question = get_last_user_message(messages)

    return async_streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt": {"query": question}},
    )
```

![chat-completion-streaming-example](../assets/chat-completion-streaming.gif)

## OpenAIChatGenerator Integration

Hayhooks works seamlessly with Haystack's OpenAIChatGenerator:

```python
from haystack.components.generators.chat import OpenAIChatGenerator

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.llm = OpenAIChatGenerator(
            model="gpt-4o-mini",
            streaming_callback=lambda chunk: print(chunk)
        )
        # ... rest of pipeline setup

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"messages": messages},
        )
```

## Examples

### Simple Chat Pipeline

```python
class SimpleChatWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components.builders import PromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator

        prompt_builder = PromptBuilder(template="Answer: {{query}}")
        llm = OpenAIChatGenerator(model="gpt-4o-mini")

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("prompt_builder", "llm")

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> str:
        question = get_last_user_message(messages)
        result = self.pipeline.run({"prompt_builder": {"query": question}})
        return result["llm"]["replies"][0].content
```

### Advanced Streaming Pipeline

```python
class AdvancedStreamingWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components.builders import PromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator

        prompt_builder = PromptBuilder(template="Answer: {{query}}")
        llm = OpenAIChatGenerator(
            model="gpt-4o",
            streaming_callback=lambda chunk: None
        )

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("prompt_builder", "llm")

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        question = get_last_user_message(messages)
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt_builder": {"query": question}},
        )
```

## Next Steps

- [Open WebUI Integration](openwebui-integration.md) - Complete Open WebUI setup and advanced features
- [Examples](../examples/overview.md) - See working examples
