# Durable execution operations

Hayhooks durable execution provides fenced, at-least-once recovery. Use an
idempotency key derived from the execution ID and logical step for every
external write so recovered work remains safe to replay.

## Controlled beta deployment profile

- Use authenticated Redis 6.2+ with TLS, persistence, backups, and
  `maxmemory-policy noeviction`.
- Run one logical Hayhooks deployment, normally one replica and at most two or
  three, against one isolated namespace.
- Set a non-empty `durable_revision` on every durable Pipeline wrapper and
  managed A2A Agent. An image digest or Git SHA is the recommended value.
- Treat `HAYHOOKS_DURABLE_EXECUTION_CONCURRENCY` as the per-deployment ceiling.
  Keep its default of one until every Pipeline/Agent component and tool is proven
  concurrency-safe.
- Put REST and A2A behind authentication, request-size, rate, and tenant
  controls. `HAYHOOKS_DURABLE_MAX_NONTERMINAL_EXECUTIONS` is an optional
  deployment-wide secondary admission cap.

## Execution and recovery

The namespace holds a control record, opaque payloads, one bounded `chunks`
stream per execution, one `runnable` ZSET, one `lease-expiry` ZSET, a
`nonterminal` capacity field, and idempotency bindings. The control is
authoritative; the indexes are derived atomically with it.

Workers poll due runnable work at the configured interval using Redis `TIME`.
Candidate reads are non-destructive. A watched control hash and monotonically
increasing fence make concurrent replica claims safe. Lease maintenance uses
the same interval and processes up to 100 expired fences. Delayed retries
remain in `runnable` with their Redis-server due timestamp and are invisible
until due.

## Retention and rollout

Terminal control and payload keys and their idempotency binding receive the
configured Redis TTL when a run first becomes terminal.
`HAYHOOKS_DURABLE_MAX_STREAM_CHUNKS` bounds each execution's SSE chunk log and
`0` disables it, which is the kill switch if streaming misbehaves.
`HAYHOOKS_DURABLE_MAX_STREAM_CHUNK_BYTES` caps one chunk (64 KB by default);
oversized chunks are dropped, never failed. Every Redis chunk append refreshes
the configured TTL, so stale writers cannot create permanent keys; a nonterminal
execution quiet for that entire retention window may lose old display history
and reports a cursor gap on reattachment. Memory refuses appends after its
cleanup has run. Do not delete records manually while they are nonterminal.

Begin a new controlled-beta deployment with an empty durable namespace, then
retain its terminal records through the configured Redis TTL.

## Streaming load

A stream reads its chunk log without blocking, so an attached viewer holds a
Redis connection only for the microseconds of each read rather than for its
whole lifetime. That matters because streams share the engine's connection pool:
a blocking read pins one connection per viewer, which caps concurrent streams at
the pool size and starves the workers, whose heartbeats and terminal transitions
need that same pool. The cost of polling instead is up to 100 ms of extra
live-display latency.

Polling is two-speed: 100 ms while chunks are moving, backing off to one second
after a second of silence, since an execution can sit in `waiting` for hours with
a viewer attached. A partial page waits for the live interval; only a full
backlog page is drained immediately. An idle stream rereads its execution record
every second, so a terminal event arrives within about two seconds of the last
chunk; a run that keeps generating after a cancellation request reports terminal
only after it stops producing chunks. An idle stream costs about one chunk read,
three record reads, and one heartbeat per second per viewer, and an execution parked in
`waiting` keeps paying that until it is resumed. One read returns at most
`4 MB / durable_max_stream_chunk_bytes` chunks — 62 at the default cap — so a
client reattaching to a full log catches up over several reads rather than
materializing the whole log at once. Chunks are display data and are never a
reason to replay a run. A non-empty page also performs one control lookup before
delivery, which prevents a lease-lost worker from leaking stale chunks before
its replacement emits.

Each producer also waits for one two-command Redis pipeline per chunk (`XADD`
and rolling `EXPIRE`).
That keeps ordering and shutdown behavior simple, but caps token throughput at
roughly one chunk per Redis round trip. Batch only if production measurements
show that remote Redis latency is slowing generation.

## Health and incidents

Health exposes `nonterminal`, `runnable`, `lease_expiry`, and
`worker_store_error_streak`. A claim or transition store failure marks the
deployment health snapshot unhealthy until that worker completes a store
operation successfully; `/status/{pipeline_name}` then returns `503`. Investigate
a growing runnable count, repeated lease recovery, store failures, or executions
that remain running/waiting longer than expected. Pause submissions, preserve
the Redis namespace, and inspect controls and fences before changing code or
restarting workers.

Use Redis 6.2 or later. Monitor the durable counts alongside Redis availability
and latency to keep execution recovery healthy.
