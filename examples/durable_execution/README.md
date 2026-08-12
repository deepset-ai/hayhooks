# Durable document-preparation Pipeline

This is the canonical REST durable-execution example. The wrapper owns typed
application behavior; Hayhooks owns records, the Redis runnable queue, fenced workers,
checkpoints, retry delay, waiting/resume, cancellation, and retention.
The wrapper declares the stable `durable_revision` required by the engine; use
an image digest or Git SHA instead for production releases.

The real Haystack Pipeline cleans and splits a document. Hayhooks persists a
`PipelineSnapshot` after `clean`, so a crash during the later delay resumes
from that checkpoint without repeating the completed cleaning step.

The included `httpx` client pauses five seconds between requests, prints every
URL and response with Rich, automatically handles the retry and approval, and
stays alive while you restart Hayhooks.

Run each command from the repository root. This is a local reliability
demonstration, not a production Redis configuration.

1. Start Redis and install the example dependencies.

```bash
docker compose -f examples/durable-compose.yaml up -d && python -m pip install -e ".[durable]" httpx
```

2. Set the durable settings. The five-second lease keeps recovery short.

```bash
export HAYHOOKS_DURABLE_REDIS_URL=redis://localhost:6379/0 HAYHOOKS_DURABLE_LEASE_DURATION_MS=5000 HAYHOOKS_DURABLE_MAX_ATTEMPTS=4
```

3. Open a first terminal and start Hayhooks. It prints the PID for the forced
   crash.

```bash
sh -c 'echo "Hayhooks PID: $$"; exec hayhooks run --pipelines-dir examples/durable_execution/pipelines'
```

4. Open a second terminal and run the client. It submits the document, shows
   the intentional retry, waits for approval, approves it, and polls every five
   seconds.

```bash
python examples/durable_execution/demo.py
```

5. When the client prints “The clean checkpoint is persisted,” return to the
   first terminal and press `Ctrl-C` to stop Hayhooks. The next client request
   reports the expected connection failure and waits five seconds before trying
   again.

6. In the first terminal, start Hayhooks again with the same command. The
   client detects recovery and prints the completed response.

```bash
sh -c 'echo "Hayhooks PID: $$"; exec hayhooks run --pipelines-dir examples/durable_execution/pipelines'
```

The recovered Pipeline skips `clean`, repeats the interrupted `demo_delay`,
then runs `split`. Durable execution is at least once: if a Pipeline has an
external side effect, use the execution ID and logical step as its idempotency
key.

Press Ctrl-C in the first terminal to stop Hayhooks, then stop Redis. Add `-v`
to remove the retained Redis volume before a clean rehearsal:

```bash
docker compose -f examples/durable-compose.yaml down
# docker compose -f examples/durable-compose.yaml down -v
```
