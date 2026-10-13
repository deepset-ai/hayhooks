# Tracing

Hayhooks tracing builds on Haystack tracing APIs and standard OpenTelemetry configuration.

## Install

Install tracing extras first:

```bash
pip install "hayhooks[tracing]"
```

This installs the OpenTelemetry SDK, OTLP exporters, and FastAPI/Starlette instrumentors used by Hayhooks.
The terminal UI uses the separate `hayhooks[tui]` extra. Local trace inspection works without an external exporter.

## Quick Start (OTLP)

Set OpenTelemetry environment variables, then run Hayhooks:

```bash
export OTEL_SERVICE_NAME=hayhooks
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=1.0
export HAYHOOKS_TRACING_EXCLUDED_SPANS='["send", "receive"]'

hayhooks run
```

!!! tip "Using OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    If you set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` directly for HTTP transport, use the full traces path
    (for example `http://localhost:4318/v1/traces`).

## Local Backends

### SigNoz

Start SigNoz locally:

```bash
git clone -b main https://github.com/SigNoz/signoz.git
cd signoz/deploy/docker
docker compose up -d --remove-orphans
```

Open SigNoz at `http://localhost:8080`, then go to **Traces** and filter by service `hayhooks`.

### Jaeger

If you want a lighter single-container setup:

```bash
docker run --rm -d \
  --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Open Jaeger at `http://localhost:16686` and use the same `OTEL_*` variables shown above.

## Terminal trace inspection

Tracing and durable execution indexing are enabled by default. On a local POSIX
machine, start the server and dashboard under the same OS account:

```bash
pip install "hayhooks[tui]"
hayhooks run
# In another terminal:
hayhooks dashboard
# e: recent ordinary runs + durable history; v: all process traces
```

No Node.js, npm build or browser is required. Local capture includes Hayhooks
operations and Haystack component spans; `hayhooks[tracing]` is needed only for
OpenTelemetry SDK/exporters and framework instrumentation. External exporters
continue to work independently of the local trace API.

The terminal trace view provides:

- Recent trace feed, entrypoint/kind/state filters and trace/tag search.
- Newest or slowest sorting, failure counts, and average **finished** duration
  for the visible recent sample. Running roots remain live even after children finish.
- Streaming indicators, trace IDs and timestamps, tags, error messages and
  stack traces when `HAYHOOKS_SHOW_TRACEBACKS=true`.
- An expandable span tree with component names, durations, offsets, a terminal
  waterfall and the slowest completed component above the configured threshold.
- Pause, manual refresh, themes, keyboard/mouse navigation and stale snapshot
  retention. `c` hides the current traces **locally**; it never clears the server.

The list refreshes every two seconds by default (`--interval` changes this).
Background requests use the same single worker as deployment and execution
views. Each refresh replaces a bounded snapshot of at most 100 recent traces;
server eviction, restart and clear are reflected without cursor backfill.
Filters and statistics describe this sample, not all historical activity.
Opening a trace keeps a selected span and expanded branches across updates.
An inspector outside the current buffer/filter is explicitly marked as retained.

### Operator authorization

Trace buffers include operations from every user handled by the responding
process. Owner-scoped durable access does not authorize viewing the whole buffer.
Local servers bound to `localhost`, `127.0.0.1` or `::1` automatically create a
private operator credential at `~/.cache/hayhooks/dashboard-token` (or under
`XDG_CACHE_HOME`). The directory and file must belong to the current user and
have private permissions (700 and 600). Concurrent workers share this credential.
The terminal discovers it only for those literal loopback hostnames, and never
sends it automatically to remote hosts. Windows requires explicit credentials.

All `/traces` requests still require operator authorization. Remote servers
require an explicit `HAYHOOKS_TRACING_TOKEN` or verified host dependency. Set
`HAYHOOKS_DASHBOARD_TOKEN` to the same token in the client terminal. Explicit
credentials take precedence over local discovery. Treat the local file as an
operator secret: it authorizes process-wide inspection, including other users'
operations. Owner-scoped durable access alone does not authorize this buffer.
Use `--without-tracing` / `HAYHOOKS_TRACING_ENABLED=false` to disable the API,
and `--without-monitoring` / `HAYHOOKS_MONITORING_ENABLED=false` to disable
durable indexing and collection routes.
The terminal also supports its existing API-key headers with a custom host dependency.

For an existing authentication system, override the dependency on your app:

```python
from hayhooks.server.app import create_app
from hayhooks.server.routers.traces import require_tracing_access

app = create_app()
app.dependency_overrides[require_tracing_access] = require_operator
```

`require_operator` must verify credentials and raise HTTP 401/403 for unauthorized
requests. Trusted authentication middleware may instead set
`request.state.tracing_operator = True` after verification. Do not derive this
flag directly from an unverified client header.

### Scope and limits

The buffer is bounded, in memory and process-local. It is separate from durable
execution indexes and an external tracing backend. Trace durations measure spans;
execution creation/update timestamps do not measure runtime duration.

With multiple workers or replicas, each process has its own buffer. Use a single
worker or a connection pinned to one process for consistent inspection. Use
OTLP for persistent or fleet-wide history. Local clearing only affects the
responding process. Payload values remain disabled by default; enable them only
for trusted debugging because prompts, documents and other sensitive data can
appear in tags.

| Setting | Default | Purpose |
| --- | --- | --- |
| `HAYHOOKS_TRACING_ENABLED` | `true` | Mount the authenticated local trace API |
| `HAYHOOKS_TRACING_TOKEN` | unset | Explicit operator bearer token; local POSIX hosts otherwise use a private credential |
| `HAYHOOKS_TRACING_BUFFER_CAPACITY` | `200` | Maximum retained process traces |
| `HAYHOOKS_TRACING_DEFAULT_LIMIT` | `25` | Default trace snapshot size |
| `HAYHOOKS_TRACING_MAX_LIMIT` | `100` | Maximum trace snapshot size (also capped at 100 and buffer capacity) |
| `HAYHOOKS_TRACING_SLOW_COMPONENT_MS` | `1000` | Minimum completed duration for slow-component highlighting |
| `HAYHOOKS_TRACING_INCLUDE_HAYSTACK_SPANS` | `true` | Mirror Haystack components into the local buffer |
| `HAYHOOKS_TRACING_INCLUDE_PAYLOAD_VALUES` | `false` | Include redacted raw request values in tags |

### Trace snapshot API

`GET /traces` is mounted only when local tracing is enabled and requires
operator access. It accepts a positive `limit`, capped at 100, the configured
maximum and buffer capacity. The response includes:

- `traces`: recent normalized traces, including running spans and their tree, tags and outcomes.
- `has_more`: whether the bounded snapshot omits older traces.
- `config`: `buffer_capacity`, `fetch_limit`, `slow_component_ms` and the process-local `scope`.

Span trees stop at depth 64 (root depth zero). The total captured `span_count`
is retained; the terminal shows visible versus total spans when descendants
are omitted. This keeps deeply nested traces usable within the snapshot limits.

Each terminal refresh reads one authoritative snapshot. There is no streaming
subscription or cursor to recover after restarts or evictions. Entrypoint
filters use `/status` and captured trace metadata; clearing the terminal list
hides traces locally without deleting the server buffer.

Generate representative deployment, REST and OpenAI traces with:

```bash
bash scripts/demo_traces.sh
```

### Migration from the browser dashboard

The React dashboard, bundled web sources, static mount and runtime npm build
have been removed. Replace `--with-tracing-dashboard` (including its old aliases)
with `hayhooks run` (local tracing is enabled by default), and open
`hayhooks dashboard --traces` instead of `/dashboard`.
`--dashboard-path` is removed. The local tracing API is now just `GET /traces`,
with inspection settings included in every snapshot. The separate config,
entrypoints, stream and server-clear endpoints, cursor fields/header, and
`since_ms`/`after_seq` parameters are removed. External clients must poll bounded
snapshots instead of consuming SSE or incremental deltas.

Rename `HAYHOOKS_DASHBOARD_TRACE_*` capture settings to `HAYHOOKS_TRACING_*`;
`HAYHOOKS_DASHBOARD_UI_SLOW_COMPONENT_MIN_DURATION_MS` becomes
`HAYHOOKS_TRACING_SLOW_COMPONENT_MS`. Trace stream/heartbeat settings, other
`HAYHOOKS_DASHBOARD_UI_*` and static frontend settings are removed;
`hayhooks dashboard --interval` controls terminal refreshes. The existing
`HAYHOOKS_DASHBOARD_TOKEN` and API-key variables remain terminal client credentials.

## Notes

- Hayhooks does not define a custom `HAYHOOKS_OTEL_*` exporter namespace. Use standard OpenTelemetry `OTEL_*` variables.
- `HAYHOOKS_TRACING_EXCLUDED_SPANS` is a Hayhooks-specific instrumentation tuning option for framework span noise.
- When `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set, Hayhooks attempts automatic OTLP bootstrap at startup using `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` (fallback: `OTEL_EXPORTER_OTLP_PROTOCOL`, default `http/protobuf`).
- For advanced setups, you can initialize your own OpenTelemetry tracer provider before importing Hayhooks/Haystack.

## Next Steps

- [Environment Variables](environment-variables.md)
- [Logging](logging.md)
