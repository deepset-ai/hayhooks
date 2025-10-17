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
from hayhooks import streaming_generator

# Create your pipeline with multiple streaming components
pipeline = Pipeline()
# ... add components ...

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

- **Streaming**: Serial execution with automatic callback management for all components
- **Transition Detection**: Uses `StreamingChunk.component_info.name` to detect when LLM 2 starts
- **Visual Separator**: Injects a `StreamingChunk` between LLM outputs
- **Error Handling**: Stream terminates gracefully if any component fails
