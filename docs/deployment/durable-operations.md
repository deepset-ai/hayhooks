# Durable Operations

Use Redis for every deployment that must survive process loss. The memory store
is intentionally process-local and is only suitable for development and tests.

## Roll out safely

1. Install `hayhooks[durable]` and provision Redis 6.2 or newer.
2. Give every wrapper revision an immutable value and deploy the same revision
   to every replica that can claim its work.
3. Start with low worker concurrency and a finite nonterminal capacity.
4. Exercise submit, checkpoint, restart, resume, cancel, and terminal retention
   before increasing traffic.
5. Drain live work before overwriting or undeploying a durable wrapper.

Hayhooks rejects a dynamic change with `409` while queued, running, or waiting
work exists. It never runs an old checkpoint under a new revision.

## Redis

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

## Capacity and stream load

`HAYHOOKS_DURABLE_MAX_NONTERMINAL_EXECUTIONS` is the admission ceiling per
deployment; `0` is unlimited. Worker concurrency controls claims, not accepted
queue size. Stream chunk count and byte limits bound display history per
execution. Reduce them before scaling SSE fan-out if display replay consumes too
much memory or bandwidth.

Lease duration must exceed the commit safety margin and comfortably cover Redis
latency and scheduler pauses. A short lease recovers faster but raises false
lease-loss risk. Application retry and run-attempt budgets are separate.

## Polling tradeoffs

Worker pickup and lease maintenance are configured independently. The balanced
defaults are one second for both:

```bash
export HAYHOOKS_DURABLE_POLL_INTERVAL_SECONDS=1
export HAYHOOKS_DURABLE_MAINTENANCE_INTERVAL_SECONDS=1
```

For empty scheduling indexes, each scan uses one Redis sorted-set command and
does not call `TIME`. Approximate empty-idle scheduling traffic is therefore:

```text
commands/second = deployments * (
    worker_concurrency / worker_interval
    + 1 / maintenance_interval
)
```

Choose intervals from the latency requirements of the deployment rather than
using a shorter value preemptively:

| Use case | Worker interval | Maintenance interval | Tradeoff |
|---|---:|---:|---|
| Balanced default | `1s` | `1s` | Up to one second for ordinary pickup and one additional second after lease expiry |
| Interactive pickup | `0.25s` | `1s` | Faster queued-work pickup with four times the idle worker scans |
| Background or batch work | `2s` | `2s` | Lower Redis traffic with up to two seconds of pickup and post-expiry delay |
| Redis-sensitive, recovery-tolerant work | `2s` | `5s` | Lowest fixed maintenance traffic; use only when five seconds of additional recovery delay is acceptable |
| Faster expired-lease recovery | `1s` | `0.25s` | More maintenance traffic; useful with deliberately short leases |

The intervals are upper bounds added by polling; average delay under steady
arrival is usually about half the configured interval. After a process crash,
recovery can take the remaining lease duration plus up to one maintenance
interval. Keep maintenance short relative to customized short leases.

Maintenance cadence does not supervise local worker capacity. Hayhooks restarts
an unexpectedly stopped worker task immediately through local task supervision,
without waiting for the next Redis maintenance scan.

## Health and recovery

`GET /status` includes durable deployment health, configured/running/draining
worker counts, maintenance state, store error streak, and bounded operational
counts. Alert on unhealthy deployments, a growing nonterminal count, repeated
store errors, or sustained draining work.

After process loss, another worker recovers an expired lease and requeues or
fails the execution according to revision and attempt rules. Old fences cannot
commit. Clients inspect the execution again and reconnect SSE with their last
cursor; they do not resubmit unless no execution was created.

## Incident checklist

- Stop new submissions before changing Redis or wrapper revisions.
- Preserve Redis data and collect control metadata only; do not copy input,
  checkpoints, results, chunks, credentials, or idempotency material into logs.
- Confirm all replicas resolve the same immutable revision.
- Check Redis time, persistence, memory policy, and lease/runnable indexes.
- Restart healthy replicas and allow lease expiry to drive fenced recovery.
- Resume waiting work through its typed endpoint; do not edit checkpoint keys.
- Cancel unwanted work through the API and wait for the terminal projection.
