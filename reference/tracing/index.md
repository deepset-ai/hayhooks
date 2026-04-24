# Tracing

Hayhooks tracing builds on Haystack tracing APIs and standard OpenTelemetry configuration.

## Install

Install tracing extras first:

```
pip install "hayhooks[tracing]"
```

This installs the OpenTelemetry SDK, OTLP exporters, and FastAPI/Starlette instrumentors used by Hayhooks.

## Quick Start (OTLP)

Set OpenTelemetry environment variables, then run Hayhooks:

```
export OTEL_SERVICE_NAME=hayhooks
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=1.0
export HAYHOOKS_TRACING_EXCLUDED_SPANS='["send", "receive"]'

hayhooks run
```

Using OTEL_EXPORTER_OTLP_TRACES_ENDPOINT

If you set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` directly for HTTP transport, use the full traces path (for example `http://localhost:4318/v1/traces`).

## Local Backends

### SigNoz

Start SigNoz locally:

```
git clone -b main https://github.com/SigNoz/signoz.git
cd signoz/deploy/docker
docker compose up -d --remove-orphans
```

Open SigNoz at `http://localhost:8080`, then go to **Traces** and filter by service `hayhooks`.

### Jaeger

If you want a lighter single-container setup:

```
docker run --rm -d \
  --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Open Jaeger at `http://localhost:16686` and use the same `OTEL_*` variables shown above.

## Notes

- Hayhooks does not define a custom `HAYHOOKS_OTEL_*` exporter namespace. Use standard OpenTelemetry `OTEL_*` variables.
- `HAYHOOKS_TRACING_EXCLUDED_SPANS` is a Hayhooks-specific instrumentation tuning option for framework span noise.
- When `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set, Hayhooks attempts automatic OTLP bootstrap at startup using `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` (fallback: `OTEL_EXPORTER_OTLP_PROTOCOL`, default `http/protobuf`).
- For advanced setups, you can initialize your own OpenTelemetry tracer provider before importing Hayhooks/Haystack.

## Next Steps

- [Environment Variables](https://deepset-ai.github.io/hayhooks/reference/environment-variables/index.md)
- [Logging](https://deepset-ai.github.io/hayhooks/reference/logging/index.md)
