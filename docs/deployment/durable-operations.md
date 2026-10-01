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
- In durable mode, the Hayhooks server's worker client has 5-second socket and
  connect timeouts, TCP keepalive, RESP2, no command retries, and an unbounded
  pool. Bound HTTP concurrency in a proxy or with uvicorn
  `--limit-concurrency` when the Redis connection count matters. URL query
  options override these client defaults.
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

The original baseline was measured on 2026-09-29 at `3e4509ed`, before the
Redis protocol changes. The final engine was measured on 2026-10-01 at
`2f91ee06`. Both used `scripts/benchmark_durable_redis.py`. This synthetic
runner uses the real runtime and SSE generator: six turns of 250 Haystack
`StreamingChunk` objects paced over 40 seconds, five simulated tool boundaries,
11 cancellation checks, six checkpoints, and five progress events. It has one
SSE viewer, approximately 20 KB of input, 30 KB of checkpoint state, and a 5 KB
result. Every run asserts delivery of all 1,500 chunks and the completion event.

Redis 8.6.3 results below are per-metric medians of three runs, using Python
3.13.13, redis-py 8.1.0, Haystack 3.1.0, and a warm Lua script cache:

| Per run | Before | After | Reduction |
|---|---:|---:|---:|
| Client commands | 13,695 | 1,255 | 90.8% |
| Client exchanges (a pipeline counts once) | 7,783 | 825 | 89.4% |
| Server commands, including inside Lua, excluding INFO probes | 13,693 | 4,093 | 70.1% |
| RESP request + response bytes | 21.81 MB | 1.40 MB | 93.6% |

Client commands ranged from 13,689–13,705 before and 1,255–1,258 after;
exchange counts ranged from 7,781–7,786 and 825–827. The final median was
1,395,203 RESP bytes. Bytes use decimal MB and exclude TCP/IP headers.
Exchanges are counted client sends, not a latency measurement. These results
measure Redis traffic for this workload; application throughput and latency
depend on workload, network, server, and viewer count.

An atomic new submission takes one exchange, and an idempotent replay takes
two. With `--submissions 100`, all 100 concurrent submissions completed without
an error at one exchange each.

Scheduling changes the number of flushes and viewer wake-ups. Roughly 390 chunk
flushes and their viewer reads account for most remaining exchanges. Public
inspection skips input and checkpoint payloads; resume still reads them
internally. The steady idle floor with one worker falls from 2 to 0.8 commands/s
at the default intervals, while round trips fall to 0.4/s. Each empty scan
pipelines its sorted-set read with Redis `TIME`.

To reproduce, start an isolated Redis server on localhost port 16479 with
persistence disabled. From the repository root, using a Python environment
with `hayhooks[durable]` and the versions above installed:

```bash
# Fetch the historical baseline from the original engine pull request;
# the current checkout supplies the after version.
git fetch origin pull/267/head
git worktree add --detach /tmp/hayhooks-redis-before 3e4509ed
# Warm the current scripts before collecting measurements.
PYTHONPATH=src python scripts/benchmark_durable_redis.py --duration 2 --submissions 0
for run in 1 2 3; do
  PYTHONPATH=/tmp/hayhooks-redis-before/src python scripts/benchmark_durable_redis.py
  PYTHONPATH=src python scripts/benchmark_durable_redis.py --submissions 100
done
```

Use `--port` for another isolated Redis or Valkey instance. Server counts come
from `INFO commandstats`, so other clients must not use that instance during
measurement. A local TCP proxy counts RESP bytes in both directions; client
instrumentation counts command batches. The benchmark removes only its own
random key namespace afterward.

Chunk appends, heartbeats, and owned transitions are each one Lua script call;
an owned transition first reads control and Redis time in one pipelined round
trip. Four round trips precede user code on a claim. A no-op transition stops
after its read and does not run the write script. A maintenance scan takes one
round trip.

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

Streaming callbacks check ownership, encode the chunk in the calling thread,
queue it, and return without waiting for Redis or the event loop.
`stream_chunk_sync()` may also run on the event loop; the other `*_sync`
methods still require a thread. The per-execution buffer is bounded by
`StoreConfig.max_stream_chunks`, and pending wake-ups stay bounded even when the
event loop stalls.

The first chunk after a quiet period flushes immediately. While chunks keep
arriving, the flusher sends at most one batch per 100 ms and sleeps only while
work is pending. The buffer is flushed before the execution completes, fails,
suspends, schedules a retry, or releases its claim, so final chunks are visible
before the terminal event. Delivery is best-effort: a failed flush drops those
chunks without failing the execution. A chunk queued as the lease is lost is
dropped, and the next streaming call raises `ExecutionLeaseLostError`.

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

Each open viewer holds one Redis connection for up to 15 seconds. In durable
mode, the Hayhooks server uses a 30-second socket timeout and a blocking pool of
`HAYHOOKS_DURABLE_REDIS_MAX_VIEWERS` connections. An extra viewer waits up to
one second for a connection; if none becomes available, its stream ends with
`event: error` and the client resumes from its cursor, possibly on another
replica. A portable host that serves SSE must pass a separate `viewer_client` to
`RedisExecutionStore`, built from the same URL, and size its connection pool
for the expected concurrent viewers. Without one, viewers share the worker
client, and a surge of viewers can starve heartbeats, lose leases, and
re-execute work; that is acceptable only for a single-viewer development host.
Workers are unaffected by an exhausted viewer pool. The default
`Redis.from_url` pool is unbounded on redis-py 5–7 and has 100 non-blocking
connections on redis-py 8, so portable hosts should pass `max_connections`
deliberately. The viewer client must use RESP2-shaped replies and a
`socket_timeout` longer than 15 seconds, or every blocked read fails. When a
deployment closes, open streams end without a terminal event so that clients
resume from their cursor, possibly on another replica.

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

A submission, resume, or due retry on the same process wakes one idle local
worker immediately. A maintenance pass that recovers N leases wakes up to N;
shutdown wakes every worker. The poll interval therefore only bounds pickup of
work that another replica submitted, retried, or recovered. Lease recovery
after a crash takes the remaining lease duration plus up to one maintenance
interval, and up to one poll interval more when another replica recovers it.

A claim selects one of the eight oldest due executions at random, so ordering
among near-simultaneous submissions is approximate. One maintenance pass
recovers up to 1,000 expired leases, without repeating work another maintainer
already completed. The claim itself confirms ownership; the first heartbeat is
due one third of a lease later, or immediately if loading consumed that long.

For empty scheduling indexes, each scan pipelines one Redis sorted-set command
with `TIME`. The idle floor is therefore:

```text
commands/second = 2 * deployments * processes * (
    worker_concurrency / poll_interval
    + 1 / maintenance_interval
)

exchanges/second = deployments * processes * (
    worker_concurrency / poll_interval
    + 1 / maintenance_interval
)
```

With one worker and the default intervals, that is 0.8 commands and 0.4
exchanges per second, or about 69,000 commands per day, per deployment and
process, excluding startup. Shorter intervals trade Redis traffic for latency:

| Use case | Worker interval | Maintenance interval | Tradeoff |
|---|---:|---:|---|
| Default | `5s` | `5s` | Local work starts immediately; remote pickup and post-expiry recovery wait up to five seconds |
| Fast cross-replica pickup | `1s` | `5s` | Five times the idle worker scans |
| Faster expired-lease recovery | `5s` | `1s` | More maintenance traffic; useful with deliberately short leases |

The intervals are upper bounds added by polling; average delay under steady
arrival is usually about half the configured interval. Keep maintenance short
relative to customized short leases.

`check_cancelled()` reuses execution state that the store confirmed within the
last 0.5 seconds, so frequent checks usually add no Redis exchange. A
cancellation can take up to 0.5 seconds longer to be noticed. An Agent may start
one more LLM call when cancellation arrives just after a step checkpoint; the
run still ends `canceled` at its next commit.

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

`runtime.health()`, which `GET /status` returns as `durable` in durable mode,
reports durable deployment health, configured/running/draining worker counts,
maintenance state, store error streak, and bounded operational counts. The
counts are `nonterminal`, `revision_nonterminal`, `revision_runnable` (including
delayed retries that are not due), and `lease_expiry`. `active_executions`
counts claims this process is still running, including thread-backed work
retained after shutdown. Report the process as busy while it is nonzero so an
autoscaler does not reap it mid-run. Alert on unhealthy deployments, a growing
nonterminal count, repeated store errors, or sustained draining work. Redis
store errors name their cause, for example
`Redis durable store operation failed: TimeoutError`.

`GET /status` always returns HTTP 200 and is safe for liveness probes. Its
durable data is at most one second old, and concurrent probes share one read. A
store read that exceeds one second reports the deployment as unhealthy with
`"operational_error": "TimeoutError"` and sets the top-level `status` to
`Degraded`. Use `status` or `durable.healthy` from the body for readiness and
alerts, not the HTTP code.

After process loss, another worker recovers an expired lease and requeues or
fails the execution according to attempt rules. Revision-specific runnable
indexes ensure only a worker serving the pinned revision can claim it. Old
fences cannot commit. Clients inspect the execution again and reconnect SSE with
their last cursor; they do not resubmit unless no execution was created.

A rollout may run multiple revisions at once. Claiming is revision-safe, but a
resume request must still be routed to the replica serving the execution's
pinned revision so that it uses the matching resume schema. An old revision has
drained when its `revision_nonterminal` count reaches zero.

Operational logs include `Durable execution lease lost` with `reason`,
`Durable store commit failed; retrying within the lease window`, and
`Durable execution has invalid stored data`. Application retries log when they
are scheduled or exhausted, with `retry_message`. Store, worker, maintenance,
close, and release failure logs include `error`.

## Shutdown handoff

Under `hayhooks run`, SIGTERM follows the HTTP server lifecycle:

1. uvicorn stops accepting connections and waits up to
   `HAYHOOKS_GRACEFUL_SHUTDOWN_TIMEOUT` (5 seconds by default) for open
   requests.
2. Open durable SSE streams hold that wait and are then cancelled; clients
   resume from their cursor.
3. Durable workers keep claiming work during the HTTP drain.
4. The durable runtime then closes, waits its shutdown grace, cancels remaining
   async work, and drains retained work.

Set the process kill deadline to at least the graceful HTTP timeout plus twice
`HAYHOOKS_DURABLE_SHUTDOWN_GRACE_SECONDS`, with additional time for work the
runtime retains.

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

`DurableRuntime.close()` closes deployments concurrently, so this phase takes
about one shutdown grace rather than one grace per deployment.

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
