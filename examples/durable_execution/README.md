# Durable execution

This local example demonstrates detached execution, a Pipeline checkpoint,
one bounded retry, typed approval/resume, cooperative cancellation, durable
progress, and SSE output without a paid API.

> [!NOTE]
> Hayhooks-managed hosting of durable wrappers follows in a later release. In
> this release, `hayhooks run` rejects this wrapper at startup, so the steps
> below apply once durable hosting ships. To run durable work today, embed the
> engine as in the [standalone FastAPI example](../durable_fastapi/).

Start Redis and Hayhooks from the repository root:

```bash
docker compose -f examples/durable-compose.yaml up -d
HAYHOOKS_DURABLE_STORE=redis \
  hayhooks run --pipelines-dir examples/durable_execution/pipelines
```

Submit work:

```bash
curl -i http://localhost:1416/durable_execution/run-durable \
  -H 'content-type: application/json' \
  -d '{"value": 20, "require_approval": true, "fail_once": true}'
```

Use the returned links to inspect, resume, cancel, or stream the execution:

```bash
curl -X POST http://localhost:1416/durable_execution/executions/EXECUTION_ID/resume \
  -H 'content-type: application/json' -d '{"approved": true}'
curl -N http://localhost:1416/durable_execution/executions/EXECUTION_ID/stream
```

Restart Hayhooks after the `checkpoint` progress event to exercise recovery.
The `prepare` component is not repeated after its saved boundary. Execution is
at least once: any real external write should use an idempotency key such as
`f"{context.execution_id}:publish"`.

`report_progress` keeps only the newest
`HAYHOOKS_DURABLE_MAX_PROGRESS_EVENTS` entries while buffering and after they
are persisted, so frequent progress reporting remains bounded.
