# Chat with Website Example

Build a pipeline that answers questions about one or more websites. Uses fetching, cleaning and an LLM to generate answers, and supports streaming via OpenAI-compatible chat endpoints when implemented in the wrapper.

## Where is the code?

- Wrapper example directory: [examples/pipeline_wrappers/chat_with_website_streaming](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/chat_with_website_streaming)
- See the main docs for `PipelineWrapper` basics and OpenAI compatibility

## Deploy

```bash
hayhooks pipeline deploy-files -n chat_with_website examples/pipeline_wrappers/chat_with_website_streaming
```

## Run

- API mode:

```bash
curl -X POST http://localhost:1416/chat_with_website/run \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is this website about?", "urls": ["https://python.org"]}'
```

- Chat (OpenAI-compatible), when `run_chat_completion`/`_async` is implemented:

```bash
curl -X POST http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chat_with_website",
    "messages": [{"role": "user", "content": "Tell me about https://github.com"}]
  }'
```

!!! tip "Development Tips"
    - For development, use `--overwrite` to redeploy a changed wrapper: `hayhooks pipeline deploy-files -n chat_with_website --overwrite <dir>`
    - Some examples may require extra Python packages (e.g., `trafilatura`). Install as needed.

## Durable execution

The [durable variant](https://github.com/deepset-ai/hayhooks/tree/main/examples/durable_chat_with_website) keeps the
same fetcher → converter → prompt → OpenAI generator graph and adds a typed `run_durable_async()` method. Hayhooks
persists a Pipeline snapshot after the network fetch and before conversion, allowing the fetched page to survive a
restart without duplicating converted documents and rendered prompts in later snapshots.

```bash
curl -i -X POST http://localhost:1416/chat_with_website/run-durable \
  -H 'content-type: application/json' \
  -H 'Idempotency-Key: python-generators-v2' \
  -d '{
    "urls": ["https://docs.python.org/3/howto/functional.html"],
    "question": "What is a generator and why would I use one?"
  }'
```

See the example README for Redis startup, result inspection, restart testing, cancellation, and the at-least-once
limitations around LLM requests.

## Related

- General guide: [Main docs](../index.md)
- Examples index: [Examples Overview](overview.md)
