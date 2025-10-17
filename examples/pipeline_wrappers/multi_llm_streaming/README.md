# Multi-LLM Streaming Example

This example demonstrates hayhooks' automatic multi-component streaming support.

## Overview

The pipeline contains **two LLM components in sequence**:

1. **LLM 1** (`gpt-5-nano` with `reasoning_effort: low`): Provides a short, concise initial answer to the user's question
2. **LLM 2** (`gpt-5-nano` with `reasoning_effort: medium`): Refines and expands the answer into a detailed, professional response

Both LLMs automatically stream their responses - no special configuration needed!

![Multi-LLM Streaming Example](./multi_stream.gif)

## How It Works

Hayhooks automatically enables streaming for **all** streaming-capable components. Both LLMs stream their responses serially (one after another) without any special configuration.

The pipeline connects LLM 1's replies directly to the second prompt builder. Using Jinja2 template syntax, the second prompt builder can access the `ChatMessage` attributes directly: `{{previous_response[0].text}}`. This approach is simple and doesn't require any custom extraction components.

This example also demonstrates injecting a visual separator (`**[LLM 2 - Refining the response]**`) between the two LLM outputs using `StreamingChunk.component_info` to detect component transitions.

## Usage

### Deploy with Hayhooks

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Deploy the pipeline
hayhooks deploy examples/pipeline_wrappers/multi_llm_streaming

# Test it via OpenAI-compatible API
curl -X POST http://localhost:1416/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "multi_llm_streaming",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "stream": true
  }'
```

### Use Directly in Code

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from hayhooks import streaming_generator

# Create your pipeline with multiple streaming components
pipeline = Pipeline()
# ... add LLM 1 and prompt_builder_1 ...

# Add second prompt builder that accesses ChatMessage attributes via Jinja2
pipeline.add_component(
    "prompt_builder_2",
    ChatPromptBuilder(
        template=[
            ChatMessage.from_system("You are a helpful assistant."),
            ChatMessage.from_user("Previous: {{previous_response[0].text}}\n\nRefine this.")
        ]
    )
)
# ... add LLM 2 ...

# Connect: LLM 1 replies directly to prompt_builder_2
pipeline.connect("llm_1.replies", "prompt_builder_2.previous_response")

# streaming_generator automatically streams from ALL components
for chunk in streaming_generator(
    pipeline=pipeline,
    pipeline_run_args={"prompt_builder_1": {"template_variables": {"query": "Your question"}}}
):
    print(chunk.content, end="", flush=True)
```

## Integration with OpenWebUI

This pipeline works seamlessly with OpenWebUI:

1. Configure OpenWebUI to connect to hayhooks (see [OpenWebUI Integration docs](../../../docs/features/openwebui-integration.md))
2. Deploy this pipeline
3. Select it as a model in OpenWebUI
4. Watch both LLMs stream their responses in real-time

## Technical Details

- **Pipeline Flow**: `LLM 1 → Prompt Builder 2 → LLM 2`
- **Jinja2 Templates**: `ChatPromptBuilder` uses Jinja2, allowing direct access to `ChatMessage` attributes in templates
- **Template Variables**: LLM 1's `List[ChatMessage]` replies are passed directly as `previous_response` to the second prompt builder
- **Accessing ChatMessage Content**: Use `{{previous_response[0].text}}` in templates to access the text content
- **Streaming**: Serial execution with automatic callback management for all components
- **Transition Detection**: Uses `StreamingChunk.component_info.name` to detect when LLM 2 starts
- **Visual Separator**: Injects a `StreamingChunk` between LLM outputs
- **Error Handling**: Stream terminates gracefully if any component fails
