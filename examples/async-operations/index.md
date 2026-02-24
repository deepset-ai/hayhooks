# Async Operations Example

Patterns for async pipelines in Hayhooks: streaming responses, concurrency, and background work. Use these patterns when you need high throughput or to avoid blocking.

## Where is the code?

- Async wrapper: [async_question_answer](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/async_question_answer)
- See main docs for async `run_api_async` and `run_chat_completion_async`

## Deploy (example)

```
hayhooks pipeline deploy-files -n async-question-answer examples/pipeline_wrappers/async_question_answer
```

## Run

- OpenAI-compatible chat (async streaming):

```
curl -X POST http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "async-question-answer",
    "messages": [{"role": "user", "content": "Tell me a joke about programming"}]
  }'
```

Best Practices

- Prefer `run_chat_completion_async` for streaming and concurrency
- Use async-compatible components (e.g., `OpenAIChatGenerator`) for best performance
- For legacy pipelines with sync-only components (like `OpenAIGenerator`), use `allow_sync_streaming_callbacks=True` to enable hybrid mode
- See [Hybrid Streaming](https://deepset-ai.github.io/hayhooks/concepts/pipeline-wrapper/#hybrid-streaming-mixing-async-and-sync-components) for handling legacy components

## Related

- General guide: [Main docs](https://deepset-ai.github.io/hayhooks/index.md)
- Examples index: [Examples Overview](https://deepset-ai.github.io/hayhooks/examples/overview/index.md)
