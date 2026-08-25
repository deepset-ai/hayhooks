# Durable engine implementation plan

This document is the implementation handoff for the `durable_engine` branch.
It uses `hayhooks_v2` / PR #253 as design evidence, not as a source tree to copy.
The target is a lean, portable `hayhooks.durable` engine that Hayhooks consumes
like any other host application.

Long-running A2A execution is explicitly out of scope. Existing request-bound
A2A behavior must continue working unchanged.

## Current checkpoint

Branch and references:

- Working branch: `durable_engine`, created from `main` at `279c1e79`.
- Reference implementation: `hayhooks_v2` at `633aa9b3`.
- Haystack baseline: `haystack-ai>=3.1,<4`.
  [Haystack 3.1.0](https://pypi.org/project/haystack-ai/3.1.0/) is now stable; it
  supersedes the originally requested `3.1.0rc3` while containing the required
  `State.to_dict(skip_keys=...)` API from
  [Haystack PR #12334](https://github.com/deepset-ai/haystack/pull/12334).

Implemented:

- [x] Pure immutable reducer in `src/hayhooks/durable/engine.py`.
- [x] Direct persistence protocol and in-memory store in
  `src/hayhooks/durable/store.py`.
- [x] Reducer tests in `tests/test_durable_engine.py`.
- [x] Initial memory-store contract tests in `tests/test_durable_store.py`.
- [x] Store-side timestamps, admission control, idempotency, bounded progress,
  terminal TTL, runnable and lease indexes, and bounded stream chunks.
- [x] Debug logs only after successful submission or lifecycle transition.
- [x] Duplicate `run_id` submissions are rejected before any mutation.
- [x] Importing `hayhooks.durable` no longer imports `hayhooks.server`; the
  existing root-level Hayhooks API is resolved lazily.

Current focused verification:

- Ruff and formatting pass for the changed package and tests.
- `ty` passes for `hayhooks.durable` and the lazy root initializer.
- 17 focused tests pass, including the existing root import compatibility test.
- The full local non-integration run reached 643 passing tests and reported four
  failures in existing OpenAPI/CORS tests under the locally installed
  FastAPI/Pydantic/OpenTelemetry versions. Re-run those through the repository's
  intended Hatch environments before assigning them to durable work.

## Non-negotiable architecture

### One lifecycle authority

`engine.decide()` is the only code allowed to decide lifecycle state. Stores,
workers, HTTP routes, Haystack adapters, and Hayhooks integration may construct
commands and persist transition plans; they may not assign statuses, fences,
attempts, retry counters, cancellation fields, or lifecycle timestamps.

If a required behavior cannot be expressed by an existing command, first add a
reducer test, then make the smallest reducer change. Never patch the behavior in
a store or transport.

### One persistence contract

Both memory and Redis implement `ExecutionStore` directly. Do not recreate the
reference branch's `backend.py` plus high-level `store.py` plus provider facade.
The current protocol already represents the real atomic boundary.

Common validation and plan-application helpers belong in `store.py` when both
stores need them. Redis-specific encoding and transactions belong in
`redis.py`.

### Portable core

Core durable modules must not import anything from `hayhooks.server`, the
Hayhooks registry, CLI code, or A2A. Use the Loguru singleton directly; Hayhooks
configures that same singleton when it hosts the package.

The portable runtime accepts an explicit store and an explicit runner contract.
It must not inspect global Hayhooks state or know what a `BasePipelineWrapper`
is. Hayhooks-specific wrapper discovery and deployment wiring stay under the
server package.

### No duplicate execution record

Do not port the mutable `ExecutionRecord` from the reference branch. It
duplicates reducer state and contains lifecycle mutation methods such as
`mark_failed`, `mark_canceled`, and `touch`.

Use:

- `ExecutionControl` for authoritative lifecycle data;
- opaque payload bytes for input, checkpoint, result, error, and wait data;
- immutable decoded projections for application and transport use; and
- explicit reducer commands for every state change.

### Authoritative time and fencing

The store binds `now_ms`. Redis uses Redis `TIME`; memory uses its injected
clock. Workers never calculate whether a lease is valid from their own clock.

Every running-state write carries the current fence and worker ID. A stale
worker may finish local code, but it must not commit a checkpoint, result,
failure, retry, suspension, or heartbeat.

### Durable data versus display data

Progress is durable, bounded, sequenced by the store, and committed with the
checkpoint or lifecycle transition that produced it.

Streaming chunks are bounded, best-effort display data outside the reducer.
Chunk loss must not fail or replay an execution. Every chunk carries the
producing attempt so consumers can ignore output from a stale attempt.

### Logging boundary

Log after successful durable changes, not before them. Include identifiers and
control metadata only:

- deployment;
- execution/run ID;
- command or outcome;
- status transition;
- version, attempt, and fence where useful; and
- exception type for operational failures.

Never log input, checkpoint, resume, result, error payload, stream chunk, owner
credentials, or idempotency material. Heartbeats should remain silent. Repeated
chunk drops should log at most once per execution context.

## Intended package shape

Start with these files and add another module only when a completed slice makes
the split materially clearer:

```text
src/hayhooks/durable/
├── __init__.py       public, preferably lazy exports
├── engine.py         pure reducer and commands
├── store.py          store protocol, shared limits, memory store
├── models.py         JSON codecs and immutable public/private projections
├── context.py        API exposed to running durable application code
├── runtime.py        deployment, workers, claims, health, lifecycle
├── haystack.py       Haystack 3.1 Pipeline/Agent checkpoint adapter
├── redis.py          Redis implementation of ExecutionStore
└── fastapi.py        portable HTTP/SSE adapter
```

Do not pre-create `backend.py`, `reference.py`, `provider.py`, `manager.py`,
`mode.py`, or `settings.py`. The current memory store replaces `reference.py`;
the direct protocol replaces `backend.py`; runtime configuration can start as a
frozen dataclass in `runtime.py`. Split a file only after the implementation
demonstrates a real boundary.

## Phase 1 — Freeze the foundation

Goal: establish the reducer and store as a trusted base before runtime code
depends on them.

Tasks:

- [ ] Review every reducer command against the transition table below.
- [ ] Confirm every plan writes mutually exclusive terminal payloads and removes
  obsolete wait/error/result payloads.
- [ ] Confirm heartbeat changes only the lease deadline and lease index, without
  incrementing the business version.
- [ ] Confirm cancellation wins over completion, failure, retry, and suspension.
- [ ] Confirm stale fences and the lease safety window reject owned writes.
- [ ] Confirm release-after-failed-post-claim-read returns the run to queued
  without consuming an attempt.
- [ ] Convert the existing store tests into a reusable async contract function
  that can later run unchanged against Redis. Keep store-specific tests outside
  that shared contract.
- [ ] Add only missing edge cases found during this review; avoid mirroring each
  reducer branch with redundant tests.

Transition table:

| Current | Command | Result |
|---|---|---|
| queued | claim | running with incremented attempt and fence |
| queued | cancel | canceled |
| running | heartbeat | running with renewed lease |
| running | checkpoint | running with checkpoint and progress |
| running | retry | queued at a due time |
| running | suspend | waiting with checkpoint and wait payload |
| running | complete/fail | terminal unless cancellation already won |
| running | cancel | running with a persisted cancellation request |
| waiting | resume | queued with optional updated checkpoint |
| waiting | cancel | canceled |
| expired running | recover | queued, canceled, failed, or repaired index |
| terminal | any lifecycle command | rejected or explicit no-op as defined by the reducer |

Acceptance:

- Shared reducer/store tests pass without runtime, Haystack, FastAPI, or Redis.
- `engine.py` remains deterministic and I/O-free.
- Importing `hayhooks.durable.engine` and `hayhooks.durable.store` imports no
  `hayhooks.server` module.

## Phase 2 — Add immutable models and codecs

Goal: decode opaque store payloads without creating a second mutable lifecycle
model.

Implement in `models.py`:

- [ ] `JsonValue` aliases and one strict JSON encoder/decoder using UTF-8,
  compact separators, `allow_nan=False`, and deterministic key ordering where a
  fingerprint is required.
- [ ] `ExecutionKind` with only `pipeline` and `agent` for this release.
- [ ] Immutable checkpoint envelope containing adapter kind, adapter checkpoint
  data, application state, and optional resume input.
- [ ] Sanitized persisted error value with type, bounded message, retryable flag,
  and optional code.
- [ ] Immutable public progress and execution-result models suitable for direct
  FastAPI response use.
- [ ] A projection function from `StoredExecution` to the public result.
- [ ] UTC conversion for internal millisecond timestamps.
- [ ] Canonical operation fingerprinting over deployment, revision, owner, and
  validated input. Sets must be sorted deterministically; list order must remain
  meaningful.
- [ ] Secret redaction for public exception messages before persistence.

Public projection rules:

- Expose execution ID, status, attempt, business version/sequence, progress,
  result, sanitized error, public wait description, cancellation time, creation
  time, update time, and transport-supplied links.
- Never expose validated input, full checkpoint, application state, resume
  payload, owner ID, lease owner, lease deadline, fence, idempotency digests, or
  backend keys.
- A wait projection may expose only an allowlisted shape such as `kind`,
  `message`, and `expected_input_schema`.

Do not add lifecycle methods to these models. Model construction validates and
projects data; reducer commands change it.

Tests:

- [ ] Round-trip every payload kind.
- [ ] Reject non-JSON values, NaN/infinity, malformed UTF-8/JSON, and oversized
  data at the encode boundary.
- [ ] Verify deterministic fingerprints for mapping order and set-valued
  Pydantic input.
- [ ] Verify list reordering changes the fingerprint.
- [ ] Verify public projections contain no private fields.
- [ ] Verify error messages redact common token, authorization, password, and
  query-string secret forms and respect the reducer scalar bound.

Acceptance:

- The store remains bytes-only and imports no Pydantic model.
- All serialization policy lives in one module.
- No state exists in both an `ExecutionControl` field and a mutable model field.

## Phase 3 — Add the claimed-execution context

Goal: expose checkpoint, progress, retry, wait/resume, cancellation, and
streaming controls to application code while keeping reducer commands explicit.

Implement in `context.py`:

- [ ] A `ContextVar` holding the active `DurableContext` and a scope context
  manager that always resets its token.
- [ ] `current_durable_context()` and a stateless synchronous
  `durable_streaming_callback(payload)` for Pipeline components.
- [ ] `DurableContext` properties for execution ID, run attempt, owner ID,
  application state, and resume input.
- [ ] `checkpoint()`, `report_progress()`, `check_cancelled()`, `retry()`,
  `suspend()`, and `stream_chunk()` async methods.
- [ ] Sync counterparts only for code that the runtime deliberately executes in
  a worker thread.
- [ ] A private claimed-execution handle containing store, run ID, fence, worker
  ID, lease duration, decoded checkpoint, and a lease-lost event.
- [ ] A heartbeat task that renews at a safe cadence and marks the claim lost on
  definitive lease rejection.

Behavioral rules:

- Buffer newly reported progress locally and commit it once with the next
  checkpoint or lifecycle transition. `report_progress()` may checkpoint
  immediately for its public contract, but must still use one store transition.
- `suspend()` encodes the current checkpoint envelope and sends `Suspend`; it
  then raises a private control-flow exception so the worker does not complete
  the run.
- `retry()` raises a private retry request; the worker sends `ScheduleRetry`.
- `check_cancelled()` rereads authoritative control and raises a public
  cooperative-cancellation exception when requested.
- Resume input is persisted inside the checkpoint envelope and consumed only in
  the local attempt. It remains recoverable until a later checkpoint commits
  the consumed state.
- Streaming serialization or append failures drop the chunk and never change
  lifecycle state.
- A sync method called on the runtime event loop must close its already-created
  coroutine and fail clearly instead of deadlocking or leaking a warning.

Tests:

- [ ] Checkpoint and progress share one transition.
- [ ] Concurrent cancellation cannot erase checkpoint progress.
- [ ] Stale or lost claims reject every owned context operation.
- [ ] Suspend persists checkpoint, application state, wait data, and progress
  atomically.
- [ ] Resume input survives process-style reconstruction and is consumed once
  per attempt.
- [ ] Sync bridge works from a thread and refuses the runtime loop.
- [ ] Concurrent contexts keep callbacks and chunks isolated.

Acceptance:

- Context code never assigns lifecycle fields.
- No server, wrapper, FastAPI, or A2A import exists in `context.py`.
- Heartbeat and context cleanup leave no background task behind.

## Phase 4 — Build the portable runtime and deployment

Goal: run an arbitrary typed durable callable against an explicit
`ExecutionStore`, without knowing about Hayhooks.

Implement in `runtime.py`:

- [ ] Frozen `RuntimeConfig` for worker concurrency, polling, shutdown grace,
  lease duration, maximum recovered run attempts, maximum application retries,
  retry base/max delay, and operational backoff bounds.
- [ ] Cross-field validation proving the lease duration exceeds the commit
  safety margin and leaves more than one heartbeat interval for a safe commit.
- [ ] `DurableDeployment` containing name, immutable definition revision,
  explicit store, request model, optional result/resume models, and runner.
- [ ] `DurableRuntime` that owns deployment worker tasks but not a global
  registry or implicit database client.
- [ ] Idempotent `start()` and `close()` plus asynchronous install/remove methods
  usable by a dynamic host.
- [ ] Submission gate: closing a deployment rejects new submissions and waits
  for already-admitted submissions before nonterminal work is counted.
- [ ] Fixed worker slots that restart if a slot exits unexpectedly.
- [ ] Separate bounded lease-maintenance task.
- [ ] Store-error retry with bounded jitter and a health error streak.
- [ ] Direct submit/get/cancel/resume methods with owner and revision checks.
- [ ] Aggregate runtime/deployment health with configured/running/draining slots,
  maintenance state, accepting state, store-error streak, and store counts.

Submission rules:

- Validate the request model before store admission.
- Generate an unguessable random run ID. Never use a client idempotency key as
  the public execution ID.
- Hash the scoped idempotency key for lookup. Bind it to the canonical operation
  fingerprint so reuse for different work returns a conflict.
- When no idempotency key is supplied, generate unique idempotency material so
  each call creates a run.
- Build `initial_control()` and submit encoded validated input. Store time
  replaces placeholder creation/update timestamps.
- An idempotent replay returns the original execution and never consumes
  admission capacity.

Worker outcome rules:

- Claim, then read and decode the stored execution. If that post-claim read
  fails, release the claim so the attempt is not consumed.
- Run application code inside the execution `ContextVar` scope.
- Race cooperative application work against the claim-lost event.
- On result, validate the declared result type, encode it, then send `Complete`.
- On cooperative cancellation, send a terminal command whose reducer outcome is
  canceled.
- On requested retry, compute bounded exponential backoff unless the request
  supplied a positive delay, then send `ScheduleRetry`.
- On suspension, do nothing after the already-committed `Suspend` transition.
- On ordinary exception, sanitize/encode it and send `Fail`.
- On oversized result/error/checkpoint, persist a small bounded terminal error;
  never retry an output that can never fit.
- On store or lease loss, do not invent a terminal result. Recovery owns the
  next decision.
- Thread-backed work cannot be force-killed. Keep its claim heartbeating while
  it drains, and prevent any late write after definitive fence loss.

Shutdown rules:

- Stop admission and new claims first.
- Stop maintenance.
- Allow active workers the configured grace period.
- Do not close caller-owned Redis resources while thread-backed work is still
  draining.
- Leave unfinished work recoverable by lease expiry.

Tests:

- [ ] Detached execution completes independently of submit request lifetime.
- [ ] Retry delay and both attempt budgets are enforced.
- [ ] Cancellation wins a result race.
- [ ] Revision mismatch never runs application code.
- [ ] Runtime instances and deployments are isolated.
- [ ] Quiesce waits for an admitted submission and rejects later submissions.
- [ ] Worker/store failures affect health and a later success clears the streak.
- [ ] Canceled or crashed worker tasks recreate their slot.
- [ ] Shutdown and lease loss retain non-cancellable thread work until it exits.

Acceptance:

- Runtime tests use a tiny fake runner, not Haystack.
- The runtime accepts an explicit store; no provider hierarchy or global runtime
  singleton is required.
- Starting an empty runtime performs no Redis or server work.

## Phase 5 — Add the Haystack 3.1 adapter

Goal: add real Pipeline and Agent checkpoint/resume behavior using only public
Haystack 3.1 APIs.

Dependency work:

- [ ] Add a `durable` optional dependency with `haystack-ai>=3.1,<4` and the
  supported Redis client range.
- [ ] Keep the existing non-durable Haystack 2 compatibility environment.
- [ ] Add a minimum-version CI environment that installs Haystack 3.1.0, plus a
  latest-supported Haystack 3 job if CI cost permits.
- [ ] Fail durable deployment with a targeted installation message when the
  environment is not Haystack 3.1+.

Pipeline adapter in `haystack.py`:

- [ ] Validate that the supplied object is a real Haystack 3.1 `Pipeline`.
- [ ] Run synchronous Pipeline execution in a shielded thread.
- [ ] Support declared component checkpoint boundaries with public
  `Breakpoint`, `BreakpointException`, and `PipelineSnapshot` APIs.
- [ ] Persist the latest snapshot from `PipelineRuntimeError` when present.
- [ ] On resume, pass an empty input together with the snapshot so completed
  components are not repeated.
- [ ] Remove checkpoint boundaries already passed according to snapshot visit
  counts.

Agent adapter in `haystack.py`:

- [ ] Validate a real Haystack 3.1 `Agent`.
- [ ] Install hooks once on the shared Agent; hooks locate the active execution
  from the `ContextVar`.
- [ ] Restore persisted state before a run.
- [ ] Check cancellation before each LLM call.
- [ ] Checkpoint after tool batches that will continue the Agent loop.
- [ ] Checkpoint application state on a continuing `on_exit` hook.
- [ ] Save a final checkpoint after run so recovery does not repeat the final LLM
  call.
- [ ] Preserve newly constructed per-run live tools and hook context during
  restore.
- [ ] Decode typed resume messages into the restored Agent state.
- [ ] Use `state.to_dict(skip_keys=["tools", "hook_context"])` directly. Do not
  port `_without_live_agent_resources` or inspect Haystack's serialized schema.

Adapter tests must use real Haystack 3.1 objects:

- [ ] Pipeline snapshot round trip skips completed components after retry.
- [ ] Pipeline runtime failure saves its attached snapshot.
- [ ] Agent custom state and typed resume messages restore.
- [ ] Agent checkpoints omit live tools/hook context through `skip_keys`.
- [ ] Agent checkpoint and progress share one store transition.
- [ ] Agent final checkpoint prevents another LLM call after recovery.
- [ ] Hooks do nothing during ordinary, non-durable Agent runs.
- [ ] Sync and async Pipeline/Agent paths obey the same fence behavior.

Acceptance:

- No private Haystack serialization surgery remains.
- Importing non-Haystack durable modules does not import Agent/Pipeline modules.
- No A2A request or task type appears in the adapter.

## Phase 6 — Implement Redis against the same store contract

Goal: add cross-process durability without changing runtime or reducer APIs.

Implement in `redis.py`:

- [ ] Strict deployment/key-prefix validation and a cluster-safe hash-tagged
  namespace.
- [ ] Explicit control-hash codec covering every `ExecutionControl` field and
  rejecting missing, unknown-invalid, negative, oversized, or contradictory
  values as corruption.
- [ ] Redis `TIME` binding for submissions and transitions.
- [ ] Atomic submission covering idempotency binding, duplicate run ID,
  admission capacity, control, input payload, and runnable index.
- [ ] Atomic reducer transition using `WATCH`/`MULTI` with bounded contention
  retries. Re-read authoritative control and rerun `decide()` after every watch
  conflict.
- [ ] Atomic updates for control, payload writes/deletes, progress, runnable
  index, lease-expiry index, nonterminal count, idempotency, and terminal TTL.
- [ ] Non-destructive due-candidate reads; the fenced claim transition chooses
  the winner across workers and processes.
- [ ] Bounded maintenance reads of expired lease members and reducer-driven
  repair/recovery.
- [ ] Bounded Redis Stream chunks with attempt metadata, resume cursor, page-size
  bound, and rolling TTL.
- [ ] Error normalization so redis-py failures appear as `ExecutionStoreError`.
- [ ] Explicit client ownership: the store accepts a binary Redis client and
  never closes a caller-owned client. Reject `decode_responses=True`.

Redis layout per deployment:

- one control hash per execution;
- one opaque key per payload kind;
- one bounded progress list;
- one bounded chunk stream;
- one `runnable` sorted set;
- one `lease-expiry` sorted set;
- one capacity hash containing the nonterminal count; and
- one idempotency binding per digest.

Do not add another backend abstraction. Small shared functions such as cursor
validation, command binding, payload validation, runnable score, and progress
sequence binding may move to module-level helpers in `store.py` if both stores
use them.

Tests:

- [ ] Run the Phase 1 shared store contract unchanged against Redis.
- [ ] Control codec round trip and corruption matrix.
- [ ] Concurrent submissions with the same idempotency key create one run.
- [ ] Same run ID with different idempotency material cannot overwrite data.
- [ ] Concurrent claims return exactly one fenced owner.
- [ ] Concurrent checkpoints preserve every progress event exactly once.
- [ ] Cancel/checkpoint and resume/checkpoint races preserve the winner's data.
- [ ] Stale lease-index entries repair only themselves.
- [ ] Heartbeat writes only lease fields and the index.
- [ ] Chunk append uses only bounded stream append and TTL operations; failure is
  non-fatal to the execution.
- [ ] Terminal TTL removes control, payloads, progress, chunks, and idempotency
  while keeping capacity correct.
- [ ] A real process-kill/restart test recovers and completes one Pipeline run.

Acceptance:

- Memory and Redis produce equivalent externally observable records for the
  shared contract.
- Redis transactions contain no lifecycle decisions beyond applying reducer
  plans.
- Redis 6.2+ is sufficient.

## Phase 7 — Add the portable FastAPI adapter

Goal: expose one `DurableDeployment` through typed REST and resumable SSE
without owning its workers or authentication.

Implement `create_durable_router()` in `fastapi.py` with these routes relative
to a host-selected prefix:

| Method | Path | Purpose |
|---|---|---|
| POST | `/run-durable` | validate and submit work |
| GET | `/executions/{execution_id}` | inspect public state |
| POST | `/executions/{execution_id}/cancel` | request cancellation |
| POST | `/executions/{execution_id}/resume` | validate resume input and requeue waiting work |
| GET | `/executions/{execution_id}/stream` | reattachable chunk and terminal SSE stream |

HTTP contract:

- [ ] The submit body uses the deployment's Pydantic request model and the
  response uses its typed result model.
- [ ] `Idempotency-Key` controls idempotency but never becomes the execution ID.
- [ ] Created/nonterminal submissions return 202; terminal idempotent replays may
  return 200 and set an explicit replay header.
- [ ] Set `Location` and links using FastAPI route names, so host prefixes and
  root paths remain correct.
- [ ] 404 hides both missing executions and owner mismatches.
- [ ] 409 represents idempotency, revision, and non-waiting resume conflicts.
- [ ] 422 represents request/resume/cursor/payload validation.
- [ ] 503 represents admission and store unavailability, with `Retry-After` for
  admission.
- [ ] The owner dependency is supplied by the host. Passing `None` explicitly
  enables bearer-by-unguessable-ID mode. Invalid configured owner values fail
  closed.
- [ ] Bound owner and idempotency header sizes before hashing or persistence.

SSE contract:

- [ ] Validate `Last-Event-ID` and ownership before response headers are sent.
- [ ] Emit heartbeat comments while idle.
- [ ] Emit `chunk` with cursor ID, attempt, and JSON payload.
- [ ] Ignore chunks older than the authoritative/current visible attempt.
- [ ] Emit `gap` and replay the retained tail when the requested cursor expired.
- [ ] Drain every retained chunk page before emitting the terminal event.
- [ ] Emit terminal event names matching `completed`, `failed`, or `canceled`.
- [ ] If failure occurs after headers, emit `error` and let clients reattach.
- [ ] Keep a waiting execution's stream alive; client disconnect never cancels
  the execution.
- [ ] Poll without pinning a Redis connection for the life of the SSE client.

Tests:

- [ ] Typed schema and every route under a non-empty prefix/root path.
- [ ] Owner isolation for inspect, cancel, resume, stream, and idempotency.
- [ ] Unscoped bearer-ID mode.
- [ ] Reconnect from `Last-Event-ID`, expired cursor gap, stale attempt filtering,
  zero chunk limit, terminal backlog drain, and midstream error framing.
- [ ] Failed/oversized chunk append never fails or retries the execution.

Acceptance:

- Router construction does not start or close a runtime.
- The adapter imports no Hayhooks server module.
- The same router works in a standalone FastAPI application.

## Phase 8 — Make Hayhooks dogfood the portable package

Goal: Hayhooks contributes wrapper discovery, configuration, lifecycle, and
route publication only. Execution behavior remains in `hayhooks.durable`.

Wrapper authoring contract:

- [ ] Add `durable_revision` and optional `durable_resume_model` class
  attributes to `BasePipelineWrapper`.
- [ ] Add `run_durable(context, request)` and
  `run_durable_async(context, request)` authoring methods.
- [ ] Detect whether the subclass overrides each method and require exactly one.
- [ ] Resolve annotations and require exactly
  `(context: DurableContext, request: PydanticModel)` plus an optional typed
  return annotation.
- [ ] Require a non-empty immutable revision and a real Haystack 3.1 Pipeline or
  Agent.

Hayhooks configuration:

- [ ] Add durable environment settings to `AppSettings`, then map them once into
  `StoreConfig` and `RuntimeConfig`. Do not make portable settings inherit server
  settings or vice versa.
- [ ] Default production storage to Redis; memory must be an explicit
  development/test choice and must never be an automatic Redis fallback.
- [ ] Keep Redis connection ownership in the Hayhooks app lifespan: create the
  binary client, create one namespaced store per durable deployment, close the
  runtime first, then the client.
- [ ] Expose limits for TTL, payload/progress/chunks, nonterminal admission,
  concurrency, attempts/retries, polling, leases, and shutdown grace. Avoid
  adding knobs not consumed by code.

Application lifecycle:

- [ ] Give each FastAPI app its own `DurableRuntime`; do not introduce a durable
  module singleton.
- [ ] Startup deploys/installs durable deployments before starting worker slots.
- [ ] App shutdown closes submissions, workers, maintenance, and then owned
  Redis resources.
- [ ] Empty/non-durable Hayhooks startup does not contact Redis.
- [ ] Include deployment health and store counts in status responses without
  exposing payloads.

Deployment and route lifecycle:

- [ ] Construct a portable `DurableDeployment` from the validated wrapper,
  explicit store, Haystack adapter, method runner, and models.
- [ ] Include its portable router under `/{pipeline_name}`.
- [ ] Track every durable route name/path so overwrite or undeploy removes the
  complete route family and invalidates OpenAPI once.
- [ ] Before overwrite/undeploy: close submission admission, wait for admitted
  submissions, stop new claims, then read authoritative nonterminal count.
- [ ] Reject replacement/removal with 409 while queued, running, or waiting work
  remains. This is the first clean policy; do not build cross-revision draining
  until a real rollout requirement demands it.
- [ ] If candidate preparation/publication fails, restart the old deployment and
  leave its wrapper, routes, files, and runtime association intact.
- [ ] Prepare a candidate store before publishing its routes; never publish a
  deployment whose store failed initialization.

Avoid porting the reference branch's broad registry/app refactor unless a
failing durable lifecycle test proves it necessary. Prefer adding a small
runtime association to the current deployment path over rewriting unrelated
OpenAI, dashboard, MCP, or registry behavior.

Tests:

- [ ] Startup deployment publishes typed durable routes and runs work.
- [ ] Two app instances do not share durable runtimes or workers.
- [ ] Overwrite binds routes to the new wrapper, models, runner, and revision.
- [ ] Overwrite/undeploy rejects live queued, running, and waiting work.
- [ ] Failed preparation/publication restores the old deployment.
- [ ] Durable-to-nondurable overwrite removes all durable routes.
- [ ] Store initialization failure publishes nothing.
- [ ] Existing ordinary run, streaming, MCP, and request-bound A2A tests remain
  unchanged and passing.

Acceptance:

- Hayhooks contains no second reducer, worker engine, or payload model.
- Removing the Hayhooks integration leaves `hayhooks.durable` usable by a
  standalone host.
- No long-running A2A file or dependency is added.

## Phase 9 — Port durable-only examples and showcases

Goal: demonstrate the released engine without implying long-running A2A
support.

Port and simplify from the reference branch:

- [ ] `examples/durable-compose.yaml` with Redis only.
- [ ] `examples/durable_execution/`: real Pipeline, intentional bounded retry,
  component checkpoint, typed approval wait/resume, cancellation, and restart
  recovery.
- [ ] `examples/durable_chat_with_website/`: real Pipeline checkpoint before
  generation, bounded display streaming, reconnect from cursor, concurrent
  execution isolation, and restart recovery.
- [ ] Keep one concise terminal showcase for concurrent REST durable streams.
  It must use only the durable REST/SSE API.

Do not port:

- `examples/a2a_long_running/`;
- durable A2A executors, task stores, cards, projections, or demos;
- A2A request/task IDs in durable examples; or
- docs that claim detached A2A, A2A input-required, A2A push delivery, or
  long-running A2A recovery.

Example quality rules:

- Use Haystack 3.1 APIs directly.
- Use `State.to_dict(skip_keys=...)` indirectly through the adapter; examples do
  not implement checkpoint serialization utilities.
- Keep external writes idempotent by deriving their idempotency key from
  execution ID plus logical step.
- Explain at-least-once execution and why external effects require
  idempotency.
- Do not expose `context.record`; examples use only the public context API.
- Keep environment variables and commands copy/paste runnable.
- Do not require paid APIs for the canonical non-streaming recovery example.
- Make long or network-dependent showcase steps opt-in and clearly labeled.

Tests:

- [ ] Import every example wrapper under Haystack 3.1.
- [ ] Run the durable execution example with fake/controlled components through
  retry, approval, resume, and completion.
- [ ] Verify a checkpoint prevents completed Pipeline components from rerunning.
- [ ] Verify concurrent streaming examples do not cross-deliver chunks.
- [ ] Verify retry/restart reconstructs the final stream/result according to the
  documented attempt semantics.

Acceptance:

- Every durable example uses REST/SSE, not A2A.
- The showcase and READMEs contain no instruction to start `hayhooks a2a run`.
- Existing `examples/a2a_multi_agent` remains request-bound and is not modified
  to use durable execution.

## Phase 10 — Documentation, CI, packaging, and release audit

Documentation:

- [ ] Add a durable-engine guide covering architecture, transition diagram,
  embedding, authoring, Redis layout, streaming semantics, revision policy,
  at-least-once effects, and operations.
- [ ] Add an operations page covering controlled rollout, capacity, retention,
  Redis persistence/TLS/noeviction, recovery, stream load, health, and incident
  response.
- [ ] Add API and environment-variable references generated from the final
  contract, not copied before names stabilize.
- [ ] Explicitly state that long-running A2A is not supported in this release and
  remains a later design topic.
- [ ] Update MkDocs navigation and run strict documentation build.

CI and packaging:

- [ ] Build wheel/sdist and inspect their contents for the complete durable
  package and no accidental files.
- [ ] Install the built wheel into a clean environment and verify portable
  imports before importing Hayhooks server code.
- [ ] Keep the existing Haystack 2 test job for non-durable compatibility.
- [ ] Add Haystack 3.1+ unit/type jobs with the durable extra.
- [ ] Add Redis 6.2 service-backed store and integration tests.
- [ ] Keep process-kill recovery as a bounded smoke test on one representative
  Python version.
- [ ] Run Ruff, format check, `ty`, unit tests, Redis integration tests, process
  recovery, example tests, and strict docs.

Final correctness audit:

- [ ] Reducer is still the sole lifecycle writer.
- [ ] Memory/Redis shared store contract passes.
- [ ] Concurrent claim, cancel/checkpoint, resume/checkpoint, and idempotency
  races pass repeatedly.
- [ ] Process kill after checkpoint resumes without repeating completed Pipeline
  work.
- [ ] Terminal TTL and nonterminal capacity remain correct after every terminal
  path.
- [ ] Old fences cannot commit after lease recovery or shutdown.
- [ ] Every public error and log is payload-safe.
- [ ] Owner mismatches are indistinguishable from missing executions.
- [ ] Dynamic deployment failure cannot strand or silently replace durable work.
- [ ] Importing `hayhooks.durable` loads no `hayhooks.server`, A2A, or Redis
  module unless the corresponding adapter is requested.

Release boundary:

- Include detached typed execution, Pipeline and Agent checkpoint recovery,
  retries, progress, cancellation, wait/resume, Redis fencing/recovery,
  reattachable SSE chunks, owner-aware REST, health, retention, portable
  embedding, Hayhooks dogfooding, and durable-only examples.
- Exclude long-running A2A execution, A2A task projection/storage, push
  notifications, and any A2A resume protocol.

## Reference-branch extraction map

Use this table to avoid wholesale ports:

| Reference file | Use | Do not copy |
|---|---|---|
| `durable/engine.py` | reducer semantics and tests | nothing further unless a reducer test requires it |
| `durable/backend.py` | Redis invariants, cursor validation, atomic-plan ideas | the extra backend layer |
| `durable/reference.py` | memory-store edge-case tests | the second memory implementation |
| `durable/store.py` | claim/context behavior and race tests | mutable record adapter and provider classes |
| `durable/models.py` | public projection, JSON bounds, error sanitization | mutable lifecycle methods and duplicated state |
| `durable/context.py` | user-facing methods, ContextVar, sync bridge | server logger import and record mutation |
| `durable/manager.py` | worker supervision, shutdown, retry outcomes | a separate manager until runtime size proves it necessary |
| `durable/runtime.py` | typed deployment contract, fingerprinting, revision checks | wrapper imports, global runtime, provider facade, A2A runner |
| `durable/adapters.py` | public Pipeline/Agent checkpoint hooks | `_without_live_agent_resources`; use Haystack 3.1 `skip_keys` |
| `durable/redis.py` | key layout, codec tests, Redis transactions | dependency on a separate backend abstraction |
| `durable/fastapi.py` | endpoint/SSE behavior and tests | server logger import and host lifecycle ownership |
| server integration | lifecycle failure cases and route tests | broad unrelated registry/router rewrites |
| durable examples | REST recovery and streaming scenarios | `a2a_long_running` and all durable A2A code |

## Agent handoff protocol

Every agent implementing a phase should:

1. Read this plan, the current `engine.py`, `store.py`, and tests before editing.
2. Inspect only the reference files mapped to that phase.
3. State the exact reducer/store/runtime invariant being implemented.
4. Reuse the existing protocol and helpers before adding a type or layer.
5. Add the smallest test that would fail if the invariant regressed.
6. Run focused Ruff, formatting, type, and tests for the slice.
7. Run the previous phase's acceptance suite as a regression gate.
8. Report changed files, tests, known limits, and the next unblocked phase.

Agents must not:

- modify or delete unrelated user changes;
- copy a reference module wholesale;
- add A2A long-running behavior;
- create speculative providers, factories, facades, repositories, or event buses;
- make storage or HTTP code assign lifecycle state;
- log durable payloads; or
- relax validation, fencing, owner isolation, or error handling to reduce code.

When two possible designs both satisfy the contract, choose the one with fewer
layers and fewer representations of the same state. Add a new abstraction only
after two real callers require it.
