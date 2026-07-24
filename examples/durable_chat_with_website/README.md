# Durable chat with website

This example takes the familiar Hayhooks `chat_with_website` Pipeline and runs
it through the durable REST API. It retains the usual fetcher → converter →
prompt → OpenAI generator graph while adding typed submission, checkpoints,
restart recovery, bounded retries, progress, inspection, and cancellation.

Start Redis, install the durable extra, and deploy the wrapper:

```bash
docker compose -f examples/durable_chat_with_website/compose.yaml up -d
pip install "hayhooks[durable]"
export OPENAI_API_KEY=...
export HAYHOOKS_DURABLE_REDIS_URL=redis://localhost:6379/0
hayhooks run --pipelines-dir examples/durable_chat_with_website/pipelines
```

Submit the same URLs and question used by the ordinary API, with a stable
idempotency key:

```bash
curl -i -X POST http://localhost:1416/chat_with_website/run-durable \
  -H 'content-type: application/json' \
  -H 'Idempotency-Key: python-generators-v2' \
  -d '{
    "urls": ["https://docs.python.org/3/howto/functional.html"],
    "question": "What is a generator and why would I use one?"
  }'
```

The key is also the execution ID, so the result can be inspected directly:

```bash
curl -s http://localhost:1416/chat_with_website/executions/python-generators-v2 | jq
```

The wrapper checkpoints before `converter`, after the network fetch has
completed. If Hayhooks stops after that checkpoint, restart it with the same
command: the persisted Pipeline snapshot resumes without fetching the website
again. Conversion, prompt construction, and the OpenAI call may repeat. Keeping
those downstream values out of additional snapshots also avoids duplicating a
large page until the execution record exceeds its configured size limit.

Cancellation is persisted:

```bash
curl -i -X POST http://localhost:1416/chat_with_website/executions/python-generators-v2/cancel
```

The execution model remains at least once. In particular, an OpenAI request may
repeat if the process dies after the provider accepts it but before Hayhooks
persists completion. Large pages also produce large Pipeline snapshots; keep
the URL set bounded and configure the durable record-size limit deliberately.
