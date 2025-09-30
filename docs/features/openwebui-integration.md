# Open WebUI Integration

Hayhooks provides seamless integration with [Open WebUI](https://openwebui.com/), enabling you to use Haystack pipelines and agents as chat completion backends with full feature support.

## Overview

Open WebUI integration allows you to:

- Use Haystack pipelines as OpenAI-compatible chat backends
- Support streaming responses in real-time
- Send status events to enhance user experience
- Intercept tool calls for better feedback
- Use file uploads with your pipelines

## Getting Started

### Prerequisites

- Open WebUI instance running
- Hayhooks server running
- Pipeline with chat completion support

### Configuration (OpenAPI Tool Server)

#### 1. Install Open WebUI

```bash
# Using Docker (recommended)
docker run -d -p 3000:8080 --name open-webui --restart always ghcr.io/open-webui/open-webui:main

# Or install locally
pip install open-webui
open-webui serve
```

#### 2. Configure Open WebUI

Step 1: Disable Auto-generated Content

Go to **Admin Settings → Interface** and turn off:

- Tags generation
- Title generation
- Follow-up message generation

![open-webui-settings](../assets/open-webui-settings.png)

This prevents unnecessary calls to your pipelines.

Step 2: Add Hayhooks Connection

Go to **Settings → Connections** (or **Admin Settings → Connections** for admin-level configuration) and add a new connection:

- **API Base URL**: `http://localhost:1416/v1`
- **API Key**: `any-value` (not used by Hayhooks)

![open-webui-settings-connections](../assets/open-webui-settings-connections.png)


## Pipeline Implementation

### Basic Chat Pipeline

```python
from typing import List, Union, Generator
from haystack import Pipeline
from hayhooks import get_last_user_message, BasePipelineWrapper, streaming_generator

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components import PromptBuilder, OpenAIChatGenerator

        prompt_builder = PromptBuilder(template="Answer: {{query}}")
        llm = OpenAIChatGenerator(model="gpt-4o-mini")

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("prompt_builder", "llm")

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        question = get_last_user_message(messages)
        result = self.pipeline.run({"prompt_builder": {"query": question}})
        return result["llm"]["replies"][0].content
```

### Streaming Chat Pipeline

```python
from typing import AsyncGenerator
from hayhooks import async_streaming_generator, get_last_user_message

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components import PromptBuilder, OpenAIChatGenerator

        prompt_builder = PromptBuilder(template="Answer: {{query}}")
        llm = OpenAIChatGenerator(
            model="gpt-4o",
            streaming_callback=lambda x: None
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

## Open WebUI Events

Hayhooks supports sending events to Open WebUI for enhanced user experience:

### Available Events

- **loading_start**: Show loading spinner
- **loading_end**: Hide loading spinner
- **message_update**: Update chat messages
- **toast_notification**: Show toast notifications

### Event Implementation

```python
from typing import AsyncGenerator, List
from hayhooks import async_streaming_generator, get_last_user_message, BasePipelineWrapper
from hayhooks.open_webui import create_status_event, create_message_event, OpenWebUIEvent

class PipelineWrapper(BasePipelineWrapper):
    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator[str | OpenWebUIEvent, None]:
        # Indicate loading
        yield create_status_event("Processing your request...", done=False)

        question = get_last_user_message(messages)

        try:
            # Stream model output alongside events
            result = async_streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args={"prompt_builder": {"query": question}},
            )

            # Optional UI hint
            yield create_message_event("✍️ Generating response...")

            async for chunk in result:
                yield chunk

            yield create_status_event("Request completed successfully", done=True)
        except Exception as e:
            yield create_status_event("Request failed", done=True)
            yield create_message_event(f"Error: {str(e)}")
            raise
```

## Tool Call Interception

For agent pipelines, you can intercept tool calls to provide real-time feedback:

```python
def on_tool_call_start(tool_name: str, arguments: dict, tool_id: str):
    """Called when a tool call starts"""
    print(f"Tool call started: {tool_name}")


def on_tool_call_end(tool_name: str, arguments: dict, result: dict, error: bool):
    """Called when a tool call ends"""
    print(f"Tool call ended: {tool_name}, Error: {error}")


class PipelineWrapper(BasePipelineWrapper):
    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Generator:
        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"messages": messages},
            on_tool_call_start=on_tool_call_start,
            on_tool_call_end=on_tool_call_end,
        )
```

## File Upload Support

OpenWebUI can send files to your pipelines:

```python
from fastapi import UploadFile
from typing import Optional, List

class PipelineWrapper(BasePipelineWrapper):
    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> str:
        # Access uploaded files through the body
        files = body.get("files", [])

        if files:
            # Process uploaded files
            filenames = [f.get("filename", "unknown") for f in files]
            return f"Received {len(files)} files: {', '.join(filenames)}"

        question = get_last_user_message(messages)
        result = self.pipeline.run({"prompt_builder": {"query": question}})
        return result["llm"]["replies"][0].content
```

## OpenAPI Tool Server

Hayhooks can also serve as an OpenAPI Tool Server for Open WebUI:

### Configuration

1. Go to **Settings → Tools**
2. Add OpenAPI Tool Server:
   - **Name**: Hayhooks
   - **URL**: `http://localhost:1416/openapi.json`

![open-webui-settings](../assets/open-webui-openapi-tools.png)

### Example: Deploy a Haystack pipeline from `open-webui` chat interface

Here's a video example of how to deploy a Haystack pipeline from the `open-webui` chat interface:

![open-webui-deploy-pipeline-from-chat-example](../assets/open-webui-deploy-pipeline-from-chat.gif)

### Available Tools

- **Deploy Pipeline**: Deploy new pipelines
- **Undeploy Pipeline**: Remove existing pipelines
- **Run Pipeline**: Execute deployed pipelines
- **Get Status**: Check pipeline status

## Example: Chat with Website

Here's a complete example for a website chat pipeline:

```python
from typing import AsyncGenerator, List
from haystack import Pipeline
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIChatGenerator
from hayhooks import async_streaming_generator, get_last_user_message

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        fetcher = LinkContentFetcher()
        converter = HTMLToDocument()
        prompt_builder = PromptBuilder(
            template="Based on this content: {{documents}}\nAnswer: {{query}}"
        )
        llm = OpenAIChatGenerator(
            model="gpt-4o",
            streaming_callback=lambda x: None
        )

        self.pipeline = Pipeline()
        self.pipeline.add_component("fetcher", fetcher)
        self.pipeline.add_component("converter", converter)
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("fetcher.content", "converter")
        self.pipeline.connect("converter.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        question = get_last_user_message(messages)

        # Extract URLs from messages or use defaults
        urls = ["https://haystack.deepset.ai"]  # Default URL

        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "fetcher": {"urls": urls},
                "prompt_builder": {"query": question}
            },
        )
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify Hayhooks server is running
   - Check API URL in Open WebUI settings
   - Ensure correct port (1416 by default)

2. **No Response**
   - Check if pipeline implements `run_chat_completion`
   - Verify pipeline is deployed
   - Check server logs for errors

3. **Streaming Not Working**
   - Ensure `streaming_callback` is set on generator
   - Check if `run_chat_completion_async` is implemented
   - Verify Open WebUI streaming is enabled

### Debug Commands

```bash
# Check Hayhooks status
hayhooks status

# Check deployed pipelines
curl http://localhost:1416/status

# Test pipeline directly
curl -X POST http://localhost:1416/my_pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"query": "test"}'
```

## Best Practices

### 1. Pipeline Design

- Implement both sync and async methods for compatibility
- Use proper error handling
- Add logging for debugging
- Consider streaming for better UX

### 2. Open WebUI Configuration

- Disable auto-generated content for simple pipelines
- Use appropriate models for your use case
- Configure timeouts appropriately

### 3. Performance

- Use async pipelines for better performance
- Implement proper error handling
- Monitor resource usage

## Next Steps

- [OpenAI Compatibility](openai-compatibility.md) - OpenAI integration
- [MCP Support](mcp-support.md) - MCP server integration
- [Examples](../examples/overview.md) - See working examples
