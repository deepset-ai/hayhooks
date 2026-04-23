# Tracing

Hayhooks tracing builds on Haystack tracing APIs and standard OpenTelemetry configuration.

## Install

Install tracing extras first:

```bash
pip install "hayhooks[tracing]"
```

This installs the OpenTelemetry SDK, OTLP exporters, and FastAPI/Starlette instrumentors used by Hayhooks.

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

## Live Dashboard

Hayhooks includes a built-in trace dashboard at `/dashboard` that provides real-time visibility into pipeline operations.

### Features

- **Live trace feed** — auto-refreshes every 2.5 seconds with new-trace animations.
- **Entrypoint filter** — click a pipeline in the sidebar to filter traces; counts update per entrypoint.
- **Span waterfall** — expand any trace to see nested spans with duration bars and per-span pipeline badges.
- **Tags** — collapsed cards show transport and success/error status; expanded view shows all tags with tooltips.
- **Error highlighting** — failed traces get a red left border so they stand out immediately.
- **Sort** — toggle between newest-first and slowest-first ordering.
- **Stats** — entrypoint count, trace count, average duration, and last-trace time, all reflecting the active filter.
- **Dark mode** — toggle between light and dark themes via the header button.
- **Clear traces** — wipe the local trace buffer from the header.

### Setup

Build the dashboard frontend:

```bash
cd dashboard
npm install
npm run build
```

Enable and configure the dashboard:

```bash
export HAYHOOKS_DASHBOARD_ENABLED=true
export HAYHOOKS_DASHBOARD_DIST_DIR=./dashboard/dist
```

### Backend Modes

The dashboard can pull traces from three sources:

#### Local (in-memory buffer)

No external backend required. Hayhooks captures its own operation spans in an in-memory ring buffer.

```bash
export HAYHOOKS_DASHBOARD_TRACE_BACKEND=local
```

This is the default fallback when an external backend is configured but unreachable.

#### Jaeger

```bash
export HAYHOOKS_DASHBOARD_TRACE_BACKEND=jaeger
export HAYHOOKS_DASHBOARD_TRACE_BACKEND_URL=http://localhost:16686
export HAYHOOKS_DASHBOARD_TRACE_SERVICE_NAME=hayhooks
```

#### SigNoz

```bash
export HAYHOOKS_DASHBOARD_TRACE_BACKEND=signoz
export HAYHOOKS_DASHBOARD_TRACE_BACKEND_URL=http://localhost:8080
export HAYHOOKS_DASHBOARD_TRACE_SIGNOZ_API_KEY=<your-signoz-api-key>
export HAYHOOKS_DASHBOARD_TRACE_SERVICE_NAME=hayhooks
```

Then run Hayhooks and open `http://localhost:1416/dashboard`.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/dashboard/api/entrypoints` | GET | List deployed pipeline names |
| `/dashboard/api/traces` | GET | Fetch recent traces (supports `limit` and `since_ms` query params) |
| `/dashboard/api/traces/clear` | POST | Clear the local trace buffer |

## Notes

- Hayhooks does not define a custom `HAYHOOKS_OTEL_*` exporter namespace. Use standard OpenTelemetry `OTEL_*` variables.
- `HAYHOOKS_TRACING_EXCLUDED_SPANS` is a Hayhooks-specific instrumentation tuning option for framework span noise.
- When `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set, Hayhooks attempts automatic OTLP bootstrap at startup using `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` (fallback: `OTEL_EXPORTER_OTLP_PROTOCOL`, default `http/protobuf`).
- For advanced setups, you can initialize your own OpenTelemetry tracer provider before importing Hayhooks/Haystack.

## Next Steps

- [Environment Variables](environment-variables.md)
- [Logging](logging.md)
