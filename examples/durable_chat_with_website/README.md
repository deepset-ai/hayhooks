# Durable chat with a website

This opt-in network example fetches up to three public pages with Haystack,
checkpoints before answer generation, and emits bounded display chunks. It uses
a deterministic excerpt answer, so no paid model or API key is required.

```bash
docker compose -f examples/durable-compose.yaml up -d
HAYHOOKS_DURABLE_STORE=redis \
  hayhooks run --pipelines-dir examples/durable_chat_with_website/pipelines
```

```bash
curl -i http://localhost:1416/durable_chat_with_website/run-durable \
  -H 'content-type: application/json' \
  -d '{"urls":["https://example.com"],"question":"What is this domain used for?"}'
curl -N http://localhost:1416/durable_chat_with_website/executions/EXECUTION_ID/stream
```

Reconnect with the last received SSE cursor:

```bash
curl -N http://localhost:1416/durable_chat_with_website/executions/EXECUTION_ID/stream \
  -H 'Last-Event-ID: CURSOR'
```

Display chunks are bounded and may be dropped; the terminal result remains
durable. Restarting Hayhooks with the same Redis store resumes after the fetch
checkpoint instead of fetching completed pages again. Network fetching is for
this local showcase; production URL ingestion should enforce the host's SSRF
policy.
