# Durable document-preparation Pipeline

This is the canonical REST durable-execution example. The wrapper owns typed
application behavior; Hayhooks owns records, Redis Streams, fenced workers,
checkpoints, retry delay, waiting/resume, cancellation, and retention.

Start production-shaped local Redis and Hayhooks:

```bash
docker compose -f examples/durable_execution/compose.yaml up -d
pip install "hayhooks[durable]"
export HAYHOOKS_DURABLE_REDIS_URL=redis://localhost:6379/0
hayhooks run --pipelines-dir examples/durable_execution/pipelines
```

Submit with an idempotency key:

```bash
curl -i -X POST http://localhost:1416/durable_job/run-durable \
  -H 'content-type: application/json' \
  -H 'Idempotency-Key: prepare-hayhooks-guide-v1' \
  -d '{
    "documents": [{
      "document_id": "hayhooks-guide",
      "content": "Hayhooks deploys Haystack pipelines. Durable execution checkpoints safe component boundaries and can resume after a restart."
    }],
    "fail_first_attempt": true,
    "require_approval": true
  }'
```

Save the `Location` header and poll it:

```bash
EXECUTION_URL=http://localhost:1416/durable_job/executions/prepare-hayhooks-guide-v1
watch -n 1 "curl -s $EXECUTION_URL | python -m json.tool"
```

The first attempt schedules one delayed retry. The second reaches `waiting`;
resume it through the typed approval contract:

```bash
curl -i -X POST "$EXECUTION_URL/resume" \
  -H 'content-type: application/json' \
  -d '{"approved": true}'
```

To test recovery, stop Hayhooks after a `clean` or `split` checkpoint while
leaving Redis running, then start the same command again. Completed upstream
Pipeline components are restored from `PipelineSnapshot`; an interrupted
component repeats from its checkpoint boundary.

To test cancellation:

```bash
curl -i -X POST "$EXECUTION_URL/cancel"
```

An accepted request first exposes `cancellation_requested_at`; the execution
becomes terminal `canceled` at a cooperative or fenced terminal-write boundary.

Durability is at least once, not exactly once. Pipeline computation can repeat
after process death, and external effects must use their own idempotency key,
normally derived from the execution ID and logical step.
