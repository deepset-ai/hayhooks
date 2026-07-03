# Async Hybrid Streaming Example

This example demonstrates using `allow_sync_streaming_callbacks=True` to enable hybrid streaming mode with a `Pipeline` that contains a sync-only component.

## Overview

Haystack v3 merged `AsyncPipeline` into `Pipeline` (which now natively supports `run_async`).
Some components, however, still only support **synchronous** streaming callbacks and don't
implement `run_async()` — for example custom or third-party components. This example uses a
small custom `SyncOnlyGenerator` to stand in for such a component and shows how to stream from
it inside an async pipeline by enabling hybrid mode with `allow_sync_streaming_callbacks=True`.

## The Problem

A component that only supports **synchronous** streaming callbacks (no `run_async()`) raises an
error when used with `async_streaming_generator` in strict mode:

```text
ValueError: Component 'llm' of type 'SyncOnlyGenerator' seems to not support async streaming callbacks
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

When `allow_sync_streaming_callbacks=True`, the system automatically detects components with
sync-only streaming callbacks (components without `run_async`) and bridges them to work in the
async context. If all components support async, no bridging is applied (pure async mode).

## When to Use This

**Use `allow_sync_streaming_callbacks=True` when:**

- Working with **sync-only components** that don't implement `run_async()`
- Deploying **YAML pipelines** where you don't control which components are used
- **Migrating** from sync to async pipelines gradually
- Using **third-party components** without async support

**For new code, prefer:**

- Using async-compatible components (e.g., `OpenAIChatGenerator`)
- Default strict mode (`allow_sync_streaming_callbacks=False`) to ensure proper async components

## Usage

### Deploy with Hayhooks

```bash
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
