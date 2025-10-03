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

## Related

- General guide: [Main docs](../index.md)
- Examples index: [Examples Overview](overview.md)
