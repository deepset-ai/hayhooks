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
- Start with `HAYHOOKS_DURABLE_EXECUTION_CONCURRENCY=1`; increase it only after
  every Pipeline/Agent component and tool is proven concurrency-safe.
- Put REST and A2A behind authentication, request-size, rate, and tenant
  controls. `HAYHOOKS_DURABLE_MAX_NONTERMINAL_EXECUTIONS` is an optional
  deployment-wide secondary admission cap.

## Execution and recovery

The namespace holds a control record, opaque payloads, one `runnable`
ZSET, one `lease-expiry` ZSET, a `nonterminal` capacity field, and idempotency
bindings. The control is authoritative; the indexes are derived atomically
with it.

Workers poll due runnable work at the configured interval using Redis `TIME`.
Candidate reads are non-destructive. A watched control hash and monotonically
increasing fence make concurrent replica claims safe. Lease maintenance uses
the same interval and processes up to 100 expired fences. Delayed retries
remain in `runnable` with their Redis-server due timestamp and are invisible
until due.

## Retention and rollout

Terminal control/payload keys and their idempotency binding receive the
configured Redis TTL when a run first becomes terminal. Memory uses equivalent
internal cleanup. Do not delete records manually while they are nonterminal.

Begin a new controlled-beta deployment with an empty durable namespace, then
retain its terminal records through the configured Redis TTL.

## Health and incidents

Health exposes `nonterminal`, `runnable`, `lease_expiry`, and
`worker_store_error_streak`. A claim or transition store failure makes readiness
unhealthy until that worker completes a store operation successfully. Investigate
a growing runnable count, repeated lease recovery, store failures, or executions
that remain running/waiting longer than expected. Pause submissions, preserve the
Redis namespace, and inspect controls and fences before changing code or
restarting workers.

Use Redis 6.2 or later. Monitor the durable counts alongside Redis availability
and latency to keep execution recovery healthy.
