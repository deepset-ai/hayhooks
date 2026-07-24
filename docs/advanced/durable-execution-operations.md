# Durable execution operations

Hayhooks durable execution provides fenced, at-least-once recovery. It does not
make application side effects exactly once. Use an idempotency key derived from
the execution ID and logical step for every external write.

## Supported Redis deployments

The first production release supports:

- standalone Redis 6.2 or newer;
- managed primary/replica services whose endpoint transparently follows the
  promoted primary.

Redis Sentinel and Redis Cluster are not supported. The current execution and
A2A Lua scripts use multiple keys that are not guaranteed to share one Cluster
hash slot. Do not point Hayhooks at a Cluster endpoint. Sentinel requires a
provider-managed client and failover test coverage before it can be claimed as
supported.

Redis 6.2 is the minimum because pending-delivery recovery uses `XAUTOCLAIM`.
Hayhooks rejects an older version during durable-store initialization when the
server exposes its version through `INFO`.

## Persistence and data-loss window

Production Redis must use `maxmemory-policy noeviction`. Eviction can remove a
record, lease, delayed retry, or A2A task independently and breaks recovery.
Hayhooks warns at startup when it can read an incompatible policy.

Choose persistence based on the accepted loss window:

- AOF `appendfsync always` minimizes acknowledged-write loss at the highest I/O
  cost.
- AOF `appendfsync everysec` commonly accepts roughly a one-second window.
- RDB alone accepts the interval since the last successful snapshot.
- AOF plus periodic RDB supports short recovery windows and convenient backups.

Test backup restoration and managed failover with the same settings used in
production. A successful Redis command can still be lost inside the documented
provider window. Monitor AOF rewrite failures, snapshot failures, replication
lag, disk capacity, and rejected writes.

Use TLS (`rediss://`) and authenticated Redis users. Keep separate key prefixes
for every environment. Redis encryption at rest is a responsibility of the
managed service or storage layer.

## Capacity and connections

One application-owned Redis client and pool is shared by durable execution
stores; the built-in A2A Redis store shares it when both use the durable
endpoint. Each deployment has an isolated Stream and delayed set. Acknowledged
deliveries are deleted explicitly. Live queued or pending entries are never
trimmed, so Stream length is real backlog and must be monitored.

Capacity planning must include:

- validated input, checkpoint, result, and bounded progress record sizes;
- queued and pending Stream entries;
- delayed retries;
- terminal execution TTL;
- A2A task TTL and active projection index;
- worker concurrency and claim/heartbeat traffic.

Idle workers use blocking Stream reads and wake immediately when work arrives.
`HAYHOOKS_DURABLE_REDIS_QUEUE_BLOCK_MS` controls the maximum idle block, while
`HAYHOOKS_DURABLE_REDIS_RECLAIM_INTERVAL` independently limits abandoned-claim
scans. Increasing the reclaim interval reduces Redis traffic but adds up to that
interval after lease expiry before another worker recovers the delivery. Each
blocked worker occupies one pooled Redis connection, so include total durable
deployment concurrency when sizing managed-service connection limits.

## Shutdown and replacement

Hayhooks stops new claims, waits for the configured grace period, and keeps
claims fenced after the grace period when underlying synchronous code is still
running. The old revision drains active work and its wrapper lifecycle remains
open until detached thread-backed work exits. A new revision atomically fails
incompatible queued or waiting work instead of silently running it under changed
code. Retirement runs again after old active work drains, covering executions
that enter waiting during replacement. An accepted cancellation still wins and
becomes terminal canceled. Actual process death is recovered after lease expiry.

## Cancellation, retry, and idempotency

Cancellation timestamps and reasons live in the versioned execution record and
do not expire while work is nonterminal. Completion, checkpoint, waiting, and
retry scripts preserve an accepted cancellation.

Retries use bounded exponential delay. `HAYHOOKS_DURABLE_MAX_ATTEMPTS` includes
the first attempt. Explicit application delay is clamped to
`HAYHOOKS_DURABLE_RETRY_MAX_DELAY`.

An `Idempotency-Key` is bound to deployment name, definition revision, validated
payload, and configured owner. Reusing it for another operation returns `409`.
Matching replay returns the existing resource and a `Location` header.

## Readiness and observability

`GET /status` includes the durable worker projection: configured/running/draining
slots, store error count, last successful claim time, and submission state. It
also returns constant-time store counts for queued, running, waiting, completed,
failed, and canceled records, plus Stream deliveries, pending deliveries, and
delayed retries. A prepared durable deployment that is not accepting work, has
missing worker slots, or cannot read its store health makes readiness return
`503`. The projection also exposes per-process active-claim gauges and bounded
operational counters for attempts, terminal outcomes, suspension, retry, lease
loss, worker restart, and record-size failures.

Logs cover claim/transition storage outages, backoff, lease loss, worker
restart, and shutdown draining without including validated input, checkpoints,
resume data, Agent messages, or tool arguments/results. Public progress must be
explicitly sanitized by the wrapper.

Monitor at minimum queue age and length, pending claims, delayed count,
retry exhaustion, cancellation latency, lease loss, worker health, Redis
errors, A2A active task count, projection retry count, and terminal retention.

## Ownership and deletion

Without `HAYHOOKS_DURABLE_TRUSTED_OWNER_HEADER`, an execution URL is a bearer
capability and is suitable only for a trusted/single-tenant boundary.

When the setting is configured, submission, inspection, cancellation, and
resume require the same owner value. The reverse proxy must remove any
client-supplied copy of that header and inject authenticated identity over a
trusted hop. Owner values longer than 512 characters are rejected rather than
truncated, so distinct identities cannot collapse onto the same authorization
boundary. Never expose this mode directly to untrusted clients.

Terminal records expire through `HAYHOOKS_DURABLE_TERMINAL_TTL_SECONDS`.
Terminal A2A tasks use their independent
`HAYHOOKS_A2A_TERMINAL_TASK_TTL_SECONDS`; cleanup also removes owner, active,
retention, and version indexes. Validated input and checkpoints are private and
never appear in the safe REST projection.
