# Durable Operations

Use Redis for every deployment that must survive process loss. The memory store
is intentionally process-local and is only suitable for development and tests.

## Roll out safely

1. Install `hayhooks[durable]` and provision Redis 6.2 or newer, or Valkey 7.2
   or newer.
2. Give every deployment revision an immutable value and serve the same
   revision on every replica that can claim its work.
3. Start with low worker concurrency and a finite nonterminal capacity.
4. Exercise submit, checkpoint, restart, resume, cancel, and terminal retention
   before increasing traffic.
5. On shutdown, close the runtime and await `wait_drained()` before closing
   its Redis clients. Serve a new revision from a new deployment instance.

Workers claim only their exact revision, so an old checkpoint never runs under a
new revision.

With the Hayhooks server, run durable wrappers in
[durable mode](../features/durable-execution.md#durable-mode):

- Ship the pipelines directory in the image or on a read-only, versioned mount,
  and roll out a change by replacing processes. A process never changes its
  pipeline set, and a failed startup exits.
- Keep every process that can claim a revision's work on the same pipeline
  names and wrapper module paths. Add each pipeline's package (the startup
  log's module path without `.pipeline_wrapper`) to
  `HAYSTACK_DESERIALIZATION_ALLOWLIST` when checkpoints hold wrapper-local
  types or handlers.
- The server performs step 5 itself. Graceful shutdown waits for retained
  work, which can take longer than `HAYHOOKS_DURABLE_SHUTDOWN_GRACE_SECONDS`;
  give the orchestrator a kill deadline that covers your longest step, or
  enable `HAYHOOKS_DURABLE_RELEASE_RUNNING_ON_SHUTDOWN` (see
  [Shutdown handoff](#shutdown-handoff)).
- Report the process as busy while `GET /status` shows nonzero
  `active_executions`, so an autoscaler does not reap it mid-run.

## Redis

- `RedisExecutionStore` takes an injected `redis.asyncio` client and viewer
  client, so the host chooses the database number, authentication, and
  topology. Redis Cluster and Sentinel topologies are not validated; test a
  compatible client through the store API.
- The store rejects clients that retry commands automatically, because
  resending a script can commit twice. Build clients with `Redis.from_url(...)`
  or pass `retry=None`; `Redis(host=...)` retries by default on redis-py 6 and
  newer. `retry_on_timeout`, `retry_on_error`, `protocol=3`, and
  `legacy_responses=False` are also rejected.
- Replies must be RESP2-shaped. Leave `protocol` unset, which redis-py 8
  translates to the legacy reply shapes, or pass `protocol=2`. Keep
  `decode_responses=False`.
- Enable persistence appropriate for the recovery objective (AOF, RDB, or both)
  and test restore from backup.
- Use a TLS Redis URL and authenticated network path outside a trusted local
  environment.
- Use `noeviction`. Evicting control, payload, progress, or index keys can make
  an execution unrecoverable.
- Keep the store's `key_prefix` private to the engine. Every key lives under it,
  so a per-tenant prefix maps to one Redis ACL key pattern. Each deployment uses a
  cluster-safe hash tag and stores control, payloads, progress, chunks,
  idempotency, per-revision runnable and lease indexes, and capacity data. The
  capacity hash has one `nonterminal:<sha256>` field per revision; there is no
  global runnable index.
- Monitor latency, memory, connection limits, persistence errors, replication
  lag, and failover behavior.

Finished executions keep their chunk stream for
`StoreConfig.stream_ttl_seconds` (one hour by default), or the terminal TTL when
it is shorter. Control, payloads, progress, and idempotency bindings keep the
terminal TTL. Size those periods for inspection needs and Redis capacity;
increasing them does not improve in-flight durability.

Control hashes and checkpoint envelopes carry an explicit storage schema
version. Unsupported versions fail closed as store corruption instead of being
interpreted with a newer runtime. Earlier prototype layouts are incompatible,
including versioned controls that lack the `lease_recoveries` lost-lease
counter. They are not auto-migrated. Drain and clean up those records, migrate
them explicitly, or start with a fresh namespace before upgrading to this
release.

Claims and maintenance remove undecodable controls from scheduling indexes and
log an error with the execution ID. Those records still require operator
cleanup of their run keys and any reserved `nonterminal` and
`nonterminal:<sha256>` capacity. Invalid claimed input, checkpoint, or progress
instead reaches a guarded terminal repair: it discards unusable progress when
needed, stores a publicly readable `stored_execution_invalid` error, and
releases capacity. Corrupt best-effort chunks cannot block that repair or lease
recovery. GET and SSE can inspect the resulting failure; do not decrement its
capacity again manually.

## Capacity and stream load

`StoreConfig.max_nonterminal_executions` is the admission ceiling per
deployment and defaults to `1000`; `0` explicitly opts into unlimited
admission. Worker concurrency controls claims, not accepted queue size. Stream
chunk count and byte limits bound display history per execution. Progress count
and byte limits bound both the worker's pending progress buffer and retained
progress history. Reduce them before scaling SSE fan-out if replay consumes too
much memory or bandwidth.

`max_payload_bytes`, `max_progress_event_bytes`, and
`max_stream_chunk_bytes` apply to new writes. Lowering them does not make valid
stored input, checkpoints, results, progress, or chunks unreadable, including
chunks larger than 4 MB that a previous setting admitted. JSON and structural
validation still apply. A resume whose new checkpoint exceeds the lowered
payload limit is rejected as too large.

Lease duration must exceed the commit safety margin and comfortably cover Redis
latency and scheduler pauses. A short lease recovers faster but raises false
lease-loss risk. Application retry and run-attempt budgets are separate; only
lost leases count toward `max_run_attempts`.

## Redis traffic

Measured on 2026-09-29 with `scripts/benchmark_durable_redis.py`, comparing
`3e4509ed` (the engine before the Redis protocol changes) with `000f388b`
(including the corruption preflight and viewer cleanup fixes). This synthetic
runner uses the real runtime and SSE generator: six turns of 250 Haystack
`StreamingChunk` objects paced over 40 seconds, five simulated tool boundaries,
11 cancellation checks, six checkpoints, and five progress events. It has one
SSE viewer, approximately 20 KB of input, 30 KB of checkpoint state, and a 5 KB
result. Every run asserts delivery of all 1,500 chunks and the completion event.

Redis 8.6.3 results below are per-metric medians of three runs, using Python
3.13.13, redis-py 8.1.0, Haystack 3.1.0, and a warm Lua script cache:

| Per run | Before | After | Reduction |
|---|---:|---:|---:|
| Client commands | 13,695 | 1,279 | 90.7% |
| Client exchanges (a pipeline counts once) | 7,783 | 846 | 89.1% |
| Server commands, including inside Lua, excluding INFO probes | 13,693 | 4,154 | 69.7% |
| RESP request + response bytes | 21.81 MB | 1.40 MB | 93.6% |

Client commands ranged from 13,689–13,705 before and 1,276–1,280 after;
exchange counts ranged from 7,781–7,786 and 844–847. Bytes use decimal MB and
exclude TCP/IP headers. Exchanges are counted client sends, not a latency
measurement. These results measure Redis traffic for this workload; application
throughput and latency depend on workload, network, server, and viewer count.

Single-run checks on the same host, with Redis 6.2 and Valkey in Docker:

| Server | Client commands, before → after | Exchanges, before → after | RESP MB, before → after |
|---|---:|---:|---:|
| Redis 6.2.24 | 13,616 → 1,264 | 7,752 → 836 | 21.37 → 1.40 |
| Valkey 9.1.2 | 13,599 → 1,265 | 7,745 → 837 | 21.36 → 1.40 |

Scheduling changes the number of flushes and viewer wake-ups. Roughly 390 chunk
flushes and their viewer reads account for most remaining exchanges. Public
inspection skips input and checkpoint payloads; resume still reads them
internally. The steady idle floor with one worker falls from 2 to 0.4 commands/s
at the default intervals, excluding startup (20 versus 4 empty-index commands
observed over a 10-second sampling window).

To reproduce, start an isolated Redis server on localhost port 16479 with
persistence disabled. From the repository root, using a Python environment
with `hayhooks[durable]` and the versions above installed:

```bash
# Both measured commits are kept in the pull request that introduced this engine.
git fetch origin pull/267/head
git worktree add --detach /tmp/hayhooks-redis-before 3e4509ed
# Warm the current scripts before collecting measurements.
PYTHONPATH=src python scripts/benchmark_durable_redis.py --duration 2
for run in 1 2 3; do
  PYTHONPATH=/tmp/hayhooks-redis-before/src python scripts/benchmark_durable_redis.py
  PYTHONPATH=src python scripts/benchmark_durable_redis.py
done
```

Use `--port` for another isolated Redis or Valkey instance. Server counts come
from `INFO commandstats`, so other clients must not use that instance during
measurement. A local TCP proxy counts RESP bytes in both directions; client
instrumentation counts command batches. The benchmark removes only its own
random key namespace afterward.

Chunk appends, heartbeats, and owned transitions are each one Lua script call;
an owned transition first reads control and Redis time in one pipelined round
trip. Submit, claim, and complete are a small constant per run.

## Commit-time lease validation

Every worker-owned write, including heartbeats and stream chunks, runs as one
Lua script that checks ownership, fence, and Redis `TIME` against the lease
deadline minus `StoreConfig.lease_commit_safety_ms` before it writes. A
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
per-execution buffer bounded by `StoreConfig.max_stream_chunks`, dropping
the oldest entries. One flusher sleeps for 100 ms between flushes, then sends
the buffer through the chunk script. Redis latency and scheduler delays add to
that interval. The buffer is flushed before the execution completes,
fails, suspends, schedules a retry, or releases its claim, so final chunks are
visible before the terminal event. Delivery is best-effort: a failed flush
drops those chunks without failing the execution.

SSE viewers block on the chunk stream with `XREAD` instead of polling. Chunk
delivery follows the next successful flush and viewer read. While blocked, a
viewer sends no new commands until entries arrive or the 15-second timeout
expires. Every terminal transition appends a marker entry that ends
open streams, including cancellation of queued or waiting work, exhausted
recovery, and deployments with chunk persistence disabled. After 15 seconds
without entries, a stream sends a keepalive comment and reads control once, so
it still ends on a terminal execution whose marker was lost.

The cursor check follows the blocking read in the same pipeline, so history
trimmed while a viewer waits produces a `gap` event as well.

Each open viewer holds one Redis connection for up to 15 seconds. The Hayhooks
server in durable mode runs a separate viewer client whose pool,
`HAYHOOKS_DURABLE_REDIS_MAX_VIEWERS`, bounds concurrent viewers per process. A
portable host that serves SSE must pass a separate `viewer_client` to
`RedisExecutionStore`, built from the same URL, and size its connection pool
for the expected concurrent viewers. Without one, viewers share the worker
client, and a surge of viewers can starve heartbeats, lose leases, and
re-execute work; that is acceptable only for a single-viewer development host.
Exhausting the viewer pool ends the affected stream with an `error` event, and
the client resumes from its cursor; workers are unaffected. The default
`Redis.from_url` pool is unbounded on redis-py 5–7 and has 100 non-blocking
connections on redis-py 8, so pass `max_connections` deliberately. The viewer
client must use RESP2-shaped replies and a `socket_timeout` longer than 15
seconds, or every blocked read fails. When a deployment closes, open streams
end without a terminal event so that clients resume from their cursor, possibly
on another replica.

A host with its own SSE transport or frame format can build it on the store:
`read_chunks` pages through retained entries, `wait_chunks` blocks for the
next ones, and an entry with `terminal=True` is the terminal marker, after
which the host reads the execution's final state.

## Pickup and recovery latency

Worker pickup and lease maintenance are configured independently. Both default
to five seconds:

```python
RuntimeConfig(poll_interval_seconds=5.0, maintenance_interval_seconds=5.0)
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

With one worker and the default intervals, that is 0.4 commands per second, or
about 35,000 per day, per deployment and process, excluding startup. Shorter
intervals trade Redis traffic for latency:

| Use case | Worker interval | Maintenance interval | Tradeoff |
|---|---:|---:|---|
| Default | `5s` | `5s` | Local work starts immediately; remote pickup and post-expiry recovery wait up to five seconds |
| Fast cross-replica pickup | `1s` | `5s` | Five times the idle worker scans |
| Faster expired-lease recovery | `5s` | `1s` | More maintenance traffic; useful with deliberately short leases |

The intervals are upper bounds added by polling; average delay under steady
arrival is usually about half the configured interval. Keep maintenance short
relative to customized short leases.

Maintenance cadence does not supervise local worker capacity. The runtime restarts
an unexpectedly stopped worker task immediately through local task supervision,
without waiting for the next Redis maintenance scan.

## Redis limits

- **Failover can lose acknowledged writes.** Replication is asynchronous, so a
  failover can drop an acknowledged checkpoint, completion, or idempotency
  binding. Work then replays from an older checkpoint under the at-least-once
  contract. `WAITAOF` (Redis and Valkey 7.2+) narrows this window but does not
  close it. If acknowledged state must survive any failover, use a
  synchronously replicated database.
- **Retained data lives in RAM.** Terminal payloads and progress stay until the
  terminal TTL expires. Chunks use the shorter of the stream and terminal TTLs.
  Control retained memory with those TTLs and the chunk and progress limits.

## Health and recovery

`runtime.health()`, which `GET /status` returns as `durable` in durable mode, reports durable deployment health, configured/running/draining
worker counts, maintenance state, store error streak, and bounded operational
counts; expose it through the host's health checks. `active_executions` counts the claims this process is still running,
including thread-backed work retained after shutdown; report the process as
busy while it is non-zero so that an autoscaler does not reap it mid-run. Alert on unhealthy deployments, a growing nonterminal count, repeated
store errors, or sustained draining work.

After process loss, another worker recovers an expired lease and requeues or
fails the execution according to attempt rules. Revision-specific runnable
indexes ensure only a worker serving the pinned revision can claim it. Old
fences cannot commit. Clients inspect the execution again and reconnect SSE with
their last cursor; they do not resubmit unless no execution was created.

A rollout may run multiple revisions at once. Claiming is revision-safe, but a
resume request must still be routed to the replica serving the execution's
pinned revision so that it uses the matching resume schema.

## Shutdown handoff

At the end of the shutdown grace, `close()` cancels async work and waits up to
another grace period for it to stop. Work that stops releases its claim: the run
is queued again at once without counting toward `max_run_attempts`; its next
claim is a new `attempt`. A pending cancellation ends it `canceled`.
Thread-backed work keeps its claim
until it exits, as does async work that suppresses cancellation or awaits cleanup;
`wait_drained()` waits for that retained work. On hosts that kill processes
shortly after SIGTERM, set `RuntimeConfig(release_running_on_close=True)` so
those claims are released too; see
[Hosts with short kill deadlines](../features/durable-execution.md#hosts-with-short-kill-deadlines)
for the overlap trade-off.

Because a handoff does not count toward `max_run_attempts`, a run that never
reaches a checkpoint can restart on every shutdown: on a fleet that replaces
processes routinely, a run whose first checkpoint takes longer than a process
lifetime never completes and never fails. Inspection shows it alternating
between `queued` and `running`; `attempt` increases with each handoff. Cancel it
through the cancel endpoint, then add a checkpoint before its first long step
so each restart resumes further along.

## Incident checklist

- Stop new submissions before changing Redis or wrapper revisions.
- Preserve Redis data and collect control metadata only; do not copy input,
  checkpoints, results, chunks, credentials, or idempotency material into logs.
- Confirm all replicas resolve the same immutable revision.
- Check Redis time, persistence, memory policy, and lease/runnable indexes.
  Stored timestamps do not move backwards when Redis time steps back; active
  leases last longer by the size of the step.
- For an undecodable control, clean up its run keys and reserved capacity after
  confirming no healthy replica can recover it. A guarded terminal repair has
  already released capacity and needs no manual counter change.
- Restart healthy replicas and allow lease expiry to drive fenced recovery.
- Resume waiting work through its typed endpoint; do not edit checkpoint keys.
- Cancel unwanted work through the API and wait for the terminal projection.
