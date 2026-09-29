# Durable Operations

Use Redis for every deployment that must survive process loss. The memory store
is intentionally process-local and is only suitable for development and tests.

## Roll out safely

1. Install `hayhooks[durable]` and provision Redis 6.2 or newer, or Valkey 7.2
   or newer.
2. Give every wrapper revision an immutable value and deploy the same revision
   to every replica that can claim its work.
3. Start with low worker concurrency and a finite nonterminal capacity.
4. Exercise submit, checkpoint, restart, resume, cancel, and terminal retention
   before increasing traffic.
5. Drain live work before overwriting or undeploying a durable wrapper.

Hayhooks rejects a dynamic change with `409` while queued, running, or waiting
work exists. For file-based wrappers, this gate runs before candidate source is
loaded or persisted, so the rejected deployment cannot affect the active
revision. Hayhooks never runs an old checkpoint under a new revision.

## Redis

- The built-in Hayhooks URL configuration uses a standalone
  `redis.asyncio.Redis` client. Redis Cluster and Sentinel topologies are not
  validated by this integration; an application that embeds the portable engine
  can pass and test a compatible client through the store API.
- Enable persistence appropriate for the recovery objective (AOF, RDB, or both)
  and test restore from backup.
- Use a TLS Redis URL and authenticated network path outside a trusted local
  environment.
- Use `noeviction`. Evicting control, payload, progress, or index keys can make
  an execution unrecoverable.
- Keep the configured key prefix private to Hayhooks. Each deployment uses a
  cluster-safe hash tag and stores control, payloads, progress, chunks,
  idempotency, runnable, lease, and capacity data.
- Monitor latency, memory, connection limits, persistence errors, replication
  lag, and failover behavior.

The terminal TTL applies to control, payloads, progress, chunks, and idempotency
bindings. Size it for inspection needs and Redis capacity; increasing it does
not improve in-flight durability.

Control hashes and checkpoint envelopes carry an explicit storage schema
version. Unsupported versions fail closed as store corruption instead of being
interpreted with a newer runtime. Records written by an earlier prototype that
did not carry a schema version are not auto-migrated; drain or migrate them
before upgrading a Redis namespace to this release.

## Capacity and stream load

`HAYHOOKS_DURABLE_MAX_NONTERMINAL_EXECUTIONS` is the admission ceiling per
deployment and defaults to `1000`; `0` explicitly opts into unlimited
admission. Worker concurrency controls claims, not accepted queue size. Stream
chunk count and byte limits bound display history per execution. Progress count
and byte limits bound both the worker's pending progress buffer and retained
progress history. Reduce them before scaling SSE fan-out if replay consumes too
much memory or bandwidth.

Lease duration must exceed the commit safety margin and comfortably cover Redis
latency and scheduler pauses. A short lease recovers faster but raises false
lease-loss risk. Application retry and run-attempt budgets are separate.

## Redis traffic

Measured on Redis 8.6.3 for one 40-second streaming agent run (6 LLM turns of
250 tokens, 5 tool calls, each with a cancellation check, a progress event, and
a checkpoint) with one SSE viewer, a 20 KB input, and a 30 KB checkpoint:

| Per run | Before | Now |
|---|---:|---:|
| Client commands | 14,392 | 1,307 |
| Round trips | 8,034 | 862 |
| Commands executed by Redis, including inside scripts | 14,385 | 4,147 |
| Bytes on the wire | 24.4 MB | 0.84 MB |
| Idle cost per deployment and process | 2 commands/s | 0.4 commands/s |

These measurements precede the key-type preflight checks described below,
which add reads inside scripts without adding client commands or round trips.

Most of the remaining round trips are the roughly 400 chunk flushes (one every
100 ms while tokens stream) and the matching viewer wake-ups. Public reads skip
the input and checkpoint, so an inspection transfers only control, progress,
and the public result, error, or wait payload. Redis 6.2 and Valkey 9.1 measure
within a few commands of these numbers.

Chunk appends, heartbeats, and owned transitions are each one Lua script call;
an owned transition first reads control and Redis time in one pipelined round
trip. Submit, claim, and complete are a small constant per run.

## Commit-time lease validation

Every worker-owned write, including heartbeats and stream chunks, runs as one
Lua script that checks ownership, fence, and Redis `TIME` against the lease
deadline minus `HAYHOOKS_DURABLE_LEASE_COMMIT_SAFETY_MS` before it writes. A
lifecycle transition also requires the stored control to equal the exact
snapshot the reducer decided from, so a worker that stalls between reading and
committing cannot write after its lease expired or after another worker claimed
the execution: the commit is rejected with nothing written. A changed snapshot
is retried from a fresh read within the transaction retry budget.

Before applying a transition, the script also checks the affected Redis key
types and validates any capacity decrement. Heartbeats check the lease index
before renewing control. Redis script errors do not roll back earlier writes,
so these checks reject corrupt targets before changing execution state.

## Streaming

Streaming callbacks never wait on Redis. `stream_chunk` appends to a
per-execution buffer bounded by `HAYHOOKS_DURABLE_MAX_STREAM_CHUNKS`, dropping
the oldest entries, and one flusher sends the buffer through the chunk script at
least every 100 ms. The buffer is flushed before the execution completes,
fails, suspends, schedules a retry, or releases its claim, so final chunks are
visible before the terminal event. Delivery is best-effort: a failed flush
drops those chunks without failing the execution.

SSE viewers block on the chunk stream with `XREAD` instead of polling, so the
flush interval is the display latency and an idle viewer issues no commands
between flushes. Every terminal transition appends a marker entry that ends
open streams, including cancellation of queued or waiting work, exhausted
recovery, and deployments with chunk persistence disabled. After 15 seconds
without entries, a stream sends a keepalive comment and reads control once, so
it still ends on a terminal execution whose marker was lost.

The cursor check follows the blocking read in the same pipeline, so history
trimmed while a viewer waits produces a `gap` event as well.

Each open viewer holds one Redis connection for up to 15 seconds. A portable
host that serves SSE must pass a separate `viewer_client` to
`RedisExecutionStore`, built from the same URL, and size its connection pool
for the expected concurrent viewers. Without one, viewers share the worker
client, and a surge of viewers can starve heartbeats, lose leases, and
re-execute work; that is acceptable only for a single-viewer development host.
Exhausting the viewer pool ends the affected stream with an `error` event, and
the client resumes from its cursor; workers are unaffected. `Redis.from_url`
creates an unbounded pool, so pass `max_connections` to make that limit real
instead of exhausting the server's shared `maxclients`. The viewer client must
use the default RESP2 protocol and a `socket_timeout` longer than 15 seconds,
or every blocked read fails. When a deployment closes, open streams end without
a terminal event so that clients resume from their cursor, possibly on another
replica.

A host with its own SSE transport or frame format can build it on the store:
`read_chunks` pages through retained entries, `wait_chunks` blocks for the
next ones, and an entry with `terminal=True` is the terminal marker, after
which the host reads the execution's final state.

## Pickup and recovery latency

Worker pickup and lease maintenance are configured independently. Both default
to five seconds:

```bash
export HAYHOOKS_DURABLE_POLL_INTERVAL_SECONDS=5
export HAYHOOKS_DURABLE_MAINTENANCE_INTERVAL_SECONDS=5
```

A submission or resume on the same process wakes an idle local worker
immediately, and so does a local retry when its delay elapses and a lease that
this process's maintenance recovers. The poll interval therefore only bounds
pickup of work that another replica submitted, retried, or recovered. Lease
recovery after a crash takes the remaining lease duration plus up to one
maintenance interval, and up to one poll interval more when another replica
recovers it.

For empty scheduling indexes, each scan uses one Redis sorted-set command and
does not call `TIME`. The idle floor is therefore:

```text
commands/second = deployments * processes * (
    worker_concurrency / poll_interval
    + 1 / maintenance_interval
)
```

With the defaults that is 0.4 commands per second, or about 35,000 per day, per
deployment and process. Shorter intervals trade Redis traffic for latency:

| Use case | Worker interval | Maintenance interval | Tradeoff |
|---|---:|---:|---|
| Default | `5s` | `5s` | Local work starts immediately; remote pickup and post-expiry recovery wait up to five seconds |
| Fast cross-replica pickup | `1s` | `5s` | Five times the idle worker scans |
| Faster expired-lease recovery | `5s` | `1s` | More maintenance traffic; useful with deliberately short leases |

The intervals are upper bounds added by polling; average delay under steady
arrival is usually about half the configured interval. Keep maintenance short
relative to customized short leases.

Maintenance cadence does not supervise local worker capacity. Hayhooks restarts
an unexpectedly stopped worker task immediately through local task supervision,
without waiting for the next Redis maintenance scan.

## Redis limits

- **Failover can lose acknowledged writes.** Replication is asynchronous, so a
  failover can drop an acknowledged checkpoint, completion, or idempotency
  binding. Work then replays from an older checkpoint under the at-least-once
  contract. `WAITAOF` (Redis and Valkey 7.2+) narrows this window but does not
  close it. If acknowledged state must survive any failover, use a
  synchronously replicated database.
- **Retained data lives in RAM.** Terminal payloads, chunks, and progress stay
  in memory until the terminal TTL expires. Control this with the terminal TTL
  and the chunk and progress limits.

## Health and recovery

`GET /status` includes durable deployment health, configured/running/draining
worker counts, maintenance state, store error streak, and bounded operational
counts. `active_executions` counts the claims this process is still running,
including thread-backed work retained after shutdown; report the process as
busy while it is non-zero so that an autoscaler does not reap it mid-run. Alert on unhealthy deployments, a growing nonterminal count, repeated
store errors, or sustained draining work.

After process loss, another worker recovers an expired lease and requeues or
fails the execution according to attempt rules. Revision-specific runnable
indexes ensure only a worker serving the pinned revision can claim it. Old
fences cannot commit. Clients inspect the execution again and reconnect SSE with
their last cursor; they do not resubmit unless no execution was created.

Applications that embed the portable engine may run multiple revisions during a
rollout. Claiming is
revision-safe, but a resume request must still be routed to the replica serving
the execution's pinned revision so that it uses the matching resume schema.
Hayhooks' dynamic deployment path avoids this requirement by rejecting a
revision change while live work exists.

## Incident checklist

- Stop new submissions before changing Redis or wrapper revisions.
- Preserve Redis data and collect control metadata only; do not copy input,
  checkpoints, results, chunks, credentials, or idempotency material into logs.
- Confirm all replicas resolve the same immutable revision.
- Check Redis time, persistence, memory policy, and lease/runnable indexes.
- Restart healthy replicas and allow lease expiry to drive fenced recovery.
- Resume waiting work through its typed endpoint; do not edit checkpoint keys.
- Cancel unwanted work through the API and wait for the terminal projection.
