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

Tracing is part of `hayhooks dashboard`. Enable the local trace API on the server,
then open the **Traces** tab (`v`) in another terminal:

```bash
pip install "hayhooks[tui]"
# Choose a private operator token; keep it out of command arguments and URLs.
export HAYHOOKS_TRACING_TOKEN="your-private-operator-token"
hayhooks run --with-tracing

# In the client terminal, supply the same token:
export HAYHOOKS_DASHBOARD_TOKEN="your-private-operator-token"
hayhooks dashboard --traces
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
All `/traces` routes default to denied unless `HAYHOOKS_TRACING_TOKEN` validates
an incoming bearer token, or the host supplies verified operator authorization.
The terminal also supports its existing API-key headers with a custom host dependency.

For an existing authentication system, override the dependency on your app:

```python
from hayhooks.server.app import create_app
from hayhooks.server.routers.traces import require_tracing_access

app = create_app()  # HAYHOOKS_TRACING_ENABLED=true
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
| `HAYHOOKS_TRACING_ENABLED` | `false` | Mount the authenticated local trace API |
| `HAYHOOKS_TRACING_TOKEN` | unset | Private bearer token for operator access |
| `HAYHOOKS_TRACING_BUFFER_CAPACITY` | `200` | Maximum retained process traces |
| `HAYHOOKS_TRACING_DEFAULT_LIMIT` | `25` | Default API page size |
| `HAYHOOKS_TRACING_MAX_LIMIT` | `100` | Maximum API page size |
| `HAYHOOKS_TRACING_SLOW_COMPONENT_MS` | `1000` | Minimum completed duration for slow-component highlighting |
| `HAYHOOKS_TRACING_INCLUDE_HAYSTACK_SPANS` | `true` | Mirror Haystack components into the local buffer |
| `HAYHOOKS_TRACING_INCLUDE_PAYLOAD_VALUES` | `false` | Include redacted raw request values in tags |
| `HAYHOOKS_TRACING_STREAM_ENABLED` | `true` | Expose the optional SSE API for external consumers |
| `HAYHOOKS_TRACING_STREAM_HEARTBEAT_MS` | `15000` | SSE keepalive interval |

### API endpoints

These are mounted only when local tracing is enabled and require operator access:

| Endpoint | Method | Description |
| --- | --- | --- |
| `/traces/config` | GET | Buffer limits and inspection settings |
| `/traces/entrypoints` | GET | Currently deployed entrypoints |
| `/traces` | GET | Normalized recent traces; `limit`, `since_ms`, `after_seq` |
| `/traces/stream` | GET | Optional SSE snapshots/deltas for other consumers |
| `/traces/clear` | POST | Clear the process buffer (never used by the terminal client) |

`GET /traces` returns `traces`, `next_after_seq` and `has_more`; the cursor is
also exposed in `X-Hayhooks-Trace-Cursor`. The terminal uses bounded snapshots,
not SSE or incremental history. External incremental clients should start a
fresh snapshot after a server restart or buffer clear.

Generate representative deployment, REST and OpenAI traces with:

```bash
bash scripts/demo_traces.sh
```

### Migration from the browser dashboard

The React dashboard, bundled web sources, static mount and runtime npm build
have been removed. Replace `--with-tracing-dashboard` (including its old aliases)
with `--with-tracing`, and open `hayhooks dashboard --traces` instead of `/dashboard`.
`--dashboard-path` is removed. Trace routes move from `/dashboard/api/*` to
`/traces`, `/traces/config`, `/traces/entrypoints`, `/traces/stream` and `/traces/clear`.

Rename `HAYHOOKS_DASHBOARD_TRACE_*` capture settings to `HAYHOOKS_TRACING_*`;
`HAYHOOKS_DASHBOARD_UI_SLOW_COMPONENT_MIN_DURATION_MS` becomes
`HAYHOOKS_TRACING_SLOW_COMPONENT_MS`. Stream settings also use `HAYHOOKS_TRACING_*`.
Other `HAYHOOKS_DASHBOARD_UI_*` and static frontend settings are removed;
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
