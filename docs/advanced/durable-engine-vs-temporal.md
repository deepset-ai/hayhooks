# Hayhooks durable engine and Temporal

Hayhooks provides focused durable execution for Haystack 3 Pipelines and Agents. Temporal is a general-purpose
durable workflow platform.

## Core comparison

| Requirement | Hayhooks durable engine | Temporal |
|---|---|---|
| Persistence and recovery | Redis records plus explicit Pipeline or Agent checkpoints | Event History plus deterministic [Workflow replay](https://docs.temporal.io/workflows) |
| Delivery safety | At-least-once with fenced leases and one active owner | Workflow logic is effectively once; [Activities](https://docs.temporal.io/activity-execution) may retry |
| Retries | Explicit, bounded retries from the latest checkpoint | Declarative, independently configurable [Activity retry policies](https://docs.temporal.io/encyclopedia/retry-policies) |
| Interaction | Typed inspect, wait/resume, progress, and result APIs | [Queries, Signals, and Updates](https://docs.temporal.io/encyclopedia/workflow-message-passing) plus durable Workflow state |
| Cancellation | Cooperative checks at safe boundaries | Cooperative cancellation with propagation policies |
| Orchestration | One Pipeline or Agent execution with delayed retries | Durable timers, Activities, [schedules](https://docs.temporal.io/schedule), and [Child Workflows](https://docs.temporal.io/child-workflows) |
| Versioning | Exact revision gate prevents incompatible recovery | Replay-safe patching and [Worker Versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) |
| Haystack example | Run a RAG Pipeline with `checkpoint_at=["generator"]`; after a crash, restore its `PipelineSnapshot` before generation | Invoke retrieval and generation as separate Activities wrapping Haystack components; completed Activity results are not repeated during Workflow replay |
| Best fit | Focused, moderate-scale durable Haystack workloads | Large-scale or cross-service orchestration around Haystack |

## Hayhooks code map

| Concern | Relevant implementation |
|---|---|
| Lifecycle and fencing | [`engine.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/engine.py) |
| Store contract and Redis persistence | [`store.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/store.py), [`redis.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/redis.py) |
| Pipeline and Agent checkpoints | [`adapters.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/adapters.py) |
| Retries and worker recovery | [`context.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/context.py), [`manager.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/manager.py) |
| Progress, wait/resume, and inspection | [`context.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/context.py), [`routes.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/server/durable/routes.py) |
| Cooperative cancellation | [`context.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/context.py), [`engine.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/engine.py) |
| Revision and deployment safety | [`runtime.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/durable/runtime.py), [`deploy_utils.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/server/utils/deploy_utils.py) |
| Durable A2A projection and recovery | [`durable_executor.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/server/a2a/durable_executor.py), [`redis_task_store.py`](https://github.com/deepset-ai/hayhooks/blob/main/src/hayhooks/server/a2a/redis_task_store.py) |

Hayhooks makes an existing Haystack Pipeline or Agent durable with minimal restructuring. Temporal offers finer-grained
orchestration, but obtaining that granularity usually means deciding which Haystack operations should become separate
Activities; the whole Pipeline can still run as one Activity when independent step recovery is unnecessary.

## What Temporal adds

| Gain | Why it matters |
|---|---|
| Independent step policies | Retrieval, generation, payments, or notifications can have separate retries, timeouts, workers, and resource limits |
| Cross-service orchestration | A Workflow can coordinate Haystack with databases, external APIs, approval systems, and other services |
| Durable time and interaction | Native timers, schedules, callbacks, Signals, Queries, and Updates |
| Horizontal routing | Activities can run on different task queues, worker pools, languages, or infrastructure |
| Production operations | Execution history, search, UI, batch operations, metrics, and mature failure investigation |
| Long-running deployments | Worker versioning, pinning, gradual rollout, rollback, and Workflows that span releases |
| Reduced engine ownership | Temporal owns task delivery, persistence, recovery, and orchestration semantics instead of Hayhooks maintaining them in Redis |

For restart-safe Haystack Pipelines and Agents with checkpoints, retries, cancellation, and resume, Temporal provides
little immediate functional gain and adds infrastructure plus integration work. It becomes valuable when an execution
grows into a long-lived, multi-step workflow spanning Haystack and other systems.

Temporal does not remove the need for idempotent external side effects: an Activity may execute more than once when its
result is lost and the task is retried.

## References

- [Hayhooks durable engine](durable-engine.md)
- [Temporal Workflows](https://docs.temporal.io/workflows)
- [Temporal Activities](https://docs.temporal.io/activities)
- [Temporal Activity execution](https://docs.temporal.io/activity-execution)
- [Temporal retry policies](https://docs.temporal.io/encyclopedia/retry-policies)
- [Temporal Workflow message passing](https://docs.temporal.io/encyclopedia/workflow-message-passing)
- [Temporal Child Workflows](https://docs.temporal.io/child-workflows)
- [Temporal Schedules](https://docs.temporal.io/schedule)
- [Temporal Visibility](https://docs.temporal.io/visibility)
- [Temporal Worker Versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning)
