# Async Operations Example

Patterns for async pipelines in Hayhooks: streaming responses, concurrency, and background work. Use these patterns when you need high throughput or to avoid blocking.

## Where is the code?

- Async wrappers: [async_question_answer](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/async_question_answer), [chat_with_website_streaming](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/chat_with_website_streaming)
- See main docs for async `run_api_async` and `run_chat_completion_async`

## Deploy (example)

```bash
hayhooks pipeline deploy-files -n async-question-answer examples/pipeline_wrappers/async_question_answer
```

## Run

- OpenAI-compatible chat (async streaming):

```bash
curl -X POST http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "async-question-answer",
    "messages": [{"role": "user", "content": "Tell me a joke about programming"}]
  }'
```

!!! tip "Best Practices"
    - Prefer `run_chat_completion_async` for streaming and concurrency
    - Ensure components support async streaming callbacks; otherwise use the sync `streaming_generator`

## Related

- General guide: [Main docs](../index.md)
- Examples index: [Examples Overview](overview.md)
