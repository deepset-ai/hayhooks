# Async Hybrid Streaming Example

This example demonstrates using `allow_sync_streaming_callbacks=True` to enable hybrid streaming mode with AsyncPipeline and legacy sync-only components.

## Overview

This example shows how to use an **AsyncPipeline** with **OpenAIGenerator** (a legacy component that only supports synchronous streaming callbacks) by enabling hybrid mode with `allow_sync_streaming_callbacks=True`.

## The Problem

Some Haystack components like `OpenAIGenerator` only support **synchronous** streaming callbacks and don't have `run_async()` support. When you try to use them with `async_streaming_generator` in an AsyncPipeline, you'll get an error:

```text
ValueError: Component 'llm' of type 'OpenAIGenerator' seems to not support async streaming callbacks
```

## The Solution

Set `allow_sync_streaming_callbacks=True` to enable **hybrid mode**:

```python
async_streaming_generator(
    pipeline=self.pipeline,
    pipeline_run_args={...},
    allow_sync_streaming_callbacks=True  # ✅ Enables hybrid mode
)
```

### What Hybrid Mode Does

When `allow_sync_streaming_callbacks=True`, the system automatically detects components with sync-only streaming callbacks (e.g., `OpenAIGenerator`) and enables hybrid mode to bridge them to work in async context. If all components support async, no bridging is applied (pure async mode).

## When to Use This

**Use `allow_sync_streaming_callbacks=True` when:**

- Working with **legacy components** like `OpenAIGenerator` that don't have async equivalents
- Deploying **YAML pipelines** where you don't control which components are used
- **Migrating** from sync to async pipelines gradually
- Using **third-party components** without async support

**For new code, prefer:**

- Using async-compatible components (e.g., `OpenAIChatGenerator` instead of `OpenAIGenerator`)
- Default strict mode (`allow_sync_streaming_callbacks=False`) to ensure proper async components

## Usage

### Deploy with Hayhooks

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Deploy the pipeline
hayhooks deploy examples/pipeline_wrappers/async_hybrid_streaming

# Test it via OpenAI-compatible API
curl -X POST http://localhost:1416/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "async_hybrid_streaming",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "stream": true
  }'
```

## Performance

Hybrid mode might have a minimal overhead (~1-2 microseconds per streaming chunk for sync components). This is negligible compared to network latency and LLM generation time.

## Related Documentation

- [Hybrid Streaming Concept](https://deepset-ai.github.io/hayhooks/concepts/pipeline-wrapper/#hybrid-streaming-mixing-async-and-sync-components)
- [Async Operations](https://deepset-ai.github.io/hayhooks/examples/async-operations/)
- [Pipeline Wrapper Guide](https://deepset-ai.github.io/hayhooks/concepts/pipeline-wrapper/)
