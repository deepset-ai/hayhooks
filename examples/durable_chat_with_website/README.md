# Durable chat-with-website with token streaming

A durable Pipeline that fetches live web pages, answers a question about them,
and streams the answer token by token over Server-Sent Events. It is the
streaming counterpart to `examples/durable_execution`: the same engine owns
records, the Redis runnable queue, fenced workers, checkpoints, and retention,
while the SSE stream carries display chunks alongside it.

The two halves are deliberately separate:

- **Durable state** is fenced and checkpointed. A `PipelineSnapshot` is
  persisted once the pages are fetched and converted, so a restart resumes into
  generation instead of hitting the network again.
- **Chunks are display data.** They live in a bounded append-only log outside
  the fence, so a token cannot contend with the lease heartbeat, and a dropped
  token can never fail or replay the execution.

The streaming callback is bound to the `llm` component in `setup()` rather than
passed per run the way `async_streaming_generator` passes it. Per-run injection
cannot work under checkpointing: run data is serialized into the
`PipelineSnapshot`, Haystack drops the callable it cannot serialize, and
`Pipeline.run` rebuilds its `data` from the snapshot when resuming, so the
callback disappears at the first checkpoint.

Sharing one bound callback across concurrent executions is safe, and for exactly
the reason `async_streaming_generator` is: its `_async_streaming_callback` is
also a single module-level function handed to every concurrent run, and the
per-run destination comes from a `ContextVar` resolved on each call. Hayhooks
routes on `_ASYNC_STREAMING_QUEUE`; this routes on the durable execution
context. Per-run injection buys that helper control over *which* components
stream, not isolation between runs. A run-time `streaming_callback` still takes
precedence, so ordinary streaming endpoints on the same wrapper behave normally.

The included `httpx` client submits a question, streams the answer, then
deliberately drops the connection mid-answer and reattaches with
`Last-Event-ID` to show that nothing in between is lost.

Run each command from the repository root. This is a local demonstration, not a
production Redis configuration.

1. Start Redis and install the example dependencies.

```bash
docker compose -f examples/durable-compose.yaml up -d && python -m pip install -e ".[durable]" httpx rich
```

2. Point Hayhooks at Redis and supply an OpenAI key. The five-second lease keeps
   the restart demonstration in step 5 short; the 30-second default would leave a
   killed execution waiting that long before another worker may reclaim it.

```bash
export HAYHOOKS_DURABLE_REDIS_URL=redis://localhost:6379/0 OPENAI_API_KEY=sk-... HAYHOOKS_DURABLE_LEASE_DURATION_MS=5000
```

3. In a first terminal, start Hayhooks.

```bash
hayhooks run --pipelines-dir examples/durable_chat_with_website/pipelines
```

4. In a second terminal, run the client.

```bash
python examples/durable_chat_with_website/demo.py
```

The answer prints token by token. After eight tokens the client drops the
connection on purpose, prints the `Last-Event-ID` it reached, reattaches, and
finishes the answer without a gap. The stream ends with a `completed` event
carrying the same projection the inspect route returns.

5. To watch the durable half, restart Hayhooks while a request is in flight.

The client reports the broken stream and keeps reattaching. Once the lease
expires, a worker in the restarted process reclaims the execution, and the
progress log records what the second attempt actually did:

```
fetch      Fetching 2 page(s)
checkpoint Checkpoint saved before pipeline component 'prompt'
resume     Resuming from the fetch checkpoint
completed  Answer complete
```

The checkpoint lands about a second after submission, so kill the server after
that to see the `resume` line. Kill it sooner and the second attempt honestly
reports a second `fetch`, because there was no snapshot to resume from yet.
The retried attempt replays from its checkpoint, so the client prints an
attempt marker when tokens repeat. A server-side `error` event is treated the
same way as a dropped connection: reattach from `Last-Event-ID`.

Expect recovery to take roughly the lease duration plus one poll interval, so
about six seconds with the settings above. The same bounded retry also covers a
transient generator failure, which is the everyday reason that checkpoint earns
its keep.

Streaming is bounded and can be switched off without touching the wrapper:

```bash
# 10 000 chunks per execution by default; 0 disables the log and leaves the endpoint working
export HAYHOOKS_DURABLE_MAX_STREAM_CHUNKS=0
```

Stop Hayhooks with Ctrl-C, then stop Redis. Add `-v` to remove the retained
volume before a clean rehearsal:

```bash
docker compose -f examples/durable-compose.yaml down
# docker compose -f examples/durable-compose.yaml down -v
```
