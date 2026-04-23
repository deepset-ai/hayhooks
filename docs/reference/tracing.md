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

## Live Dashboard (Simple)

Hayhooks now includes a minimal dashboard that polls trace data from Jaeger and shows:

- deployed entry points
- recently observed traces
- connected child spans as an expandable tree

If Jaeger is temporarily unavailable, the dashboard automatically falls back to an in-process live span buffer
for Hayhooks operation spans (for example pipeline run/openai/deploy spans) so you can still inspect recent activity.

If you want to force this behavior and avoid backend trace querying entirely, set:

```bash
export HAYHOOKS_DASHBOARD_TRACE_BACKEND=local
```

Build the dashboard frontend:

```bash
cd dashboard
npm install
npm run build
```

Enable and configure dashboard serving:

```bash
export HAYHOOKS_DASHBOARD_ENABLED=true
export HAYHOOKS_DASHBOARD_DIST_DIR=./dashboard/dist
export HAYHOOKS_DASHBOARD_TRACE_SERVICE_NAME=hayhooks
```

For Jaeger:

```bash
export HAYHOOKS_DASHBOARD_TRACE_BACKEND=jaeger
export HAYHOOKS_DASHBOARD_TRACE_BACKEND_URL=http://localhost:16686
```

For SigNoz:

```bash
export HAYHOOKS_DASHBOARD_TRACE_BACKEND=signoz
export HAYHOOKS_DASHBOARD_TRACE_BACKEND_URL=http://localhost:8080
export HAYHOOKS_DASHBOARD_TRACE_SIGNOZ_API_KEY=<your-signoz-api-key>
```

Then run Hayhooks and open `http://localhost:1416/dashboard`.

## Notes

- Hayhooks does not define a custom `HAYHOOKS_OTEL_*` exporter namespace. Use standard OpenTelemetry `OTEL_*` variables.
- `HAYHOOKS_TRACING_EXCLUDED_SPANS` is a Hayhooks-specific instrumentation tuning option for framework span noise.
- When `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set, Hayhooks attempts automatic OTLP bootstrap at startup using `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` (fallback: `OTEL_EXPORTER_OTLP_PROTOCOL`, default `http/protobuf`).
- For advanced setups, you can initialize your own OpenTelemetry tracer provider before importing Hayhooks/Haystack.

## Next Steps

- [Environment Variables](environment-variables.md)
- [Logging](logging.md)
