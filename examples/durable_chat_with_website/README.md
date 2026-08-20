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

The built-in `durable_streaming_callback` resolves the execution from a
`ContextVar` on each call, so one callback bound to a shared component still
keeps concurrent streams isolated. Per-run injection buys the ordinary helper
control over *which* components stream, not isolation between runs. A run-time
`streaming_callback` still takes precedence, so ordinary streaming endpoints on
the same wrapper behave normally.

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

### Run the three-pane recovery show

For the full demo, set the durable concurrency ceiling to two:

```bash
HAYHOOKS_DURABLE_EXECUTION_CONCURRENCY=2 hayhooks run \
  --pipelines-dir examples/durable_chat_with_website/pipelines
```

Then launch the show from another terminal:

```bash
python examples/durable_chat_with_website/showcase.py
```

Two long answers stream concurrently in the top panes. **ATLAS** cuts its SSE
connection after 12 seconds and **COMET** after 16 seconds. Each execution keeps
running while its client is away, an inspect request proves that from the
control plane, and the client reattaches three seconds later with its saved
`Last-Event-ID`. A successful pane finishes with an exact-replay proof: the
chunks seen across both connections reconstruct the durable result with no gap
or duplicates.

The bottom HTTP flight recorder makes the sequence visible across both clients:

```text
POST → 202    submit execution
GET  → 200    open SSE stream
CLOSE ✂       drop only the client connection
GET  → 200    execution is still running
GET  → 200    reattach with Last-Event-ID
SSE  completed
```

The live dashboard uses Rich, which Hayhooks already installs. Enlarge the
terminal to at least 100 columns by 28 rows so both streams have room to breathe.

### Manual curl handoff

To show the resumable stream in two side-by-side terminals, run the start script
in the first terminal. It requests a deliberately long answer, saves the stream
URL and raw SSE transcript in the system temporary directory, and uses
`jq --unbuffered` to print only the generated text:

```bash
./examples/durable_chat_with_website/start_stream.sh
```

After at least ten seconds, press Ctrl-C in the first terminal. Then run the
resume script in the second terminal. It reads the saved cursor and asks
Hayhooks for only the events after it:

```bash
./examples/durable_chat_with_website/resume_stream.sh
```

The generated text continues in the second terminal without repeating completed
chunks, while the printed `Last-Event-ID` makes the resume cursor visible. If
the bounded chunk log has discarded that cursor, Hayhooks prints the `gap`
detail before replaying the retained tail.

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
