# Production Best Practices

This page provides opinionated, Hayhooks-specific recommendations for running a secure and reliable production deployment. For infrastructure-level guidance (Docker, Kubernetes, scaling), see [Deployment Guidelines](../deployment/deployment_guidelines.md).

## Lock Down CORS

Hayhooks defaults to allowing all origins (`["*"]`). In production you should restrict CORS to only the domains that need access:

```bash
export HAYHOOKS_CORS_ALLOW_ORIGINS='["https://app.example.com", "https://admin.example.com"]'
export HAYHOOKS_CORS_ALLOW_METHODS='["GET", "POST"]'
export HAYHOOKS_CORS_ALLOW_HEADERS='["Content-Type", "Authorization"]'
export HAYHOOKS_CORS_ALLOW_CREDENTIALS=true
```

!!! warning
    Leaving `HAYHOOKS_CORS_ALLOW_ORIGINS=["*"]` in production means any website can make requests to your Hayhooks server from a browser. Always restrict origins to your own domains.

If your frontend is served from multiple subdomains, use a regex pattern instead of listing every origin:

```bash
export HAYHOOKS_CORS_ALLOW_ORIGIN_REGEX='https://.*\.example\.com'
```

See [Environment Variables Reference](../reference/environment-variables.md#cors) for all CORS settings.

## Disable Tracebacks

Stack traces in error responses can leak internal paths, library versions, and code structure. Always disable them in production:

```bash
export HAYHOOKS_SHOW_TRACEBACKS=false
```

This is the default, but it is worth verifying explicitly -- especially if you copied a development `.env` file.

## Configure Logging

Use `INFO` as the baseline log level. Switch to `WARNING` if log volume becomes a concern:

```bash
export LOG=INFO
```

!!! tip
    Avoid `DEBUG` in production. It logs every request body, pipeline step, and internal event, which can significantly increase log volume and may expose sensitive data.

See [Logging](../reference/logging.md) for log format and observability details.

## Deploy Pipelines via `HAYHOOKS_PIPELINES_DIR`

In production, pipelines should be deployed at startup from a directory -- not via CLI commands or the HTTP API at runtime:

```bash
export HAYHOOKS_PIPELINES_DIR=/app/pipelines
```

**Why this matters:**

- All workers and instances start with the same set of pipelines
- No manual deploy step is needed after a restart or scale-out event
- Pipeline definitions can be version-controlled alongside your deployment configuration

!!! info
    Runtime deploy/undeploy via the API is useful during development but introduces consistency risks in multi-worker or multi-instance setups. See [Deployment Guidelines](../deployment/deployment_guidelines.md#pipeline-deployment-strategy) for details.

## Tune Startup Deploy Performance

When deploying many pipelines from `HAYHOOKS_PIPELINES_DIR`, use parallel startup to reduce boot time:

```bash
export HAYHOOKS_STARTUP_DEPLOY_STRATEGY=parallel
export HAYHOOKS_STARTUP_DEPLOY_WORKERS=8
```

The default strategy is already `parallel` with 4 workers. Increase the worker count if you have many pipelines and available CPU cores.

For runtime deploy/undeploy operations, keep the default serialized mode unless you have a specific reason to change it:

```bash
export HAYHOOKS_DEPLOY_CONCURRENCY=serialized
```

See [Environment Variables Reference](../reference/environment-variables.md#deploy-performance) for all deploy tuning options.

## Optimize for Your Workload

Most Hayhooks pipelines fall into one of two categories. The right tuning strategy depends on which one yours is.

### I/O-Bound Pipelines

Pipelines that spend most of their time waiting on external services -- LLM API calls, database queries, HTTP requests, file downloads -- are I/O-bound.

**Recommendations:**

- **Use async methods.** Implement `run_api_async()` (or `run_chat_completion_async()`) and build your pipeline with `AsyncPipeline`. This lets a single worker handle many concurrent requests because it can switch to another request while waiting on I/O.
- **A single worker is usually enough.** Adding more workers does not help much when the bottleneck is network latency, not CPU.
- **Scale horizontally if needed.** If you need more throughput, add replicas (Kubernetes pods, Docker containers) rather than workers.

```python
from haystack import AsyncPipeline
from hayhooks import BasePipelineWrapper

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = AsyncPipeline()
        # ... build pipeline ...

    async def run_api_async(self, query: str) -> str:
        result = await self.pipeline.run_async({"prompt": {"query": query}})
        return result["llm"]["replies"][0]
```

See [Deployment Guidelines](../deployment/deployment_guidelines.md#async-pipelines) for more async examples.

### CPU-Bound Pipelines

Pipelines that perform heavy computation locally -- embedding generation, document processing, on-device model inference -- are CPU-bound.

**Recommendations:**

- **Add workers on multi-core machines.** Python's GIL limits a single worker to one CPU core for pure Python work. Use `--workers` to run multiple processes:

    ```bash
    hayhooks run --workers 4
    ```

    A common starting point is `(2 x CPU_cores) + 1`. Monitor actual CPU usage and adjust.

- **On Kubernetes, keep one worker per pod and scale via replicas.** This gives the orchestrator full control over scheduling, resource limits, and rolling updates.
- **Move heavy initialization into `setup()`.** Loading models or building indexes in `setup()` runs once at startup. Doing it inside `run_api()` would repeat the cost on every request.

!!! info
    Even with multiple workers, each worker has its own GIL. For truly CPU-intensive workloads, horizontal scaling (more pods/containers) is more effective than vertical scaling (more workers in one process). See [Deployment Guidelines](../deployment/deployment_guidelines.md#horizontal-scaling) for details.

## Add Authentication

Hayhooks does not include built-in authentication. For production, you should add one of:

- **Middleware-based API key auth** -- see the [API Key Auth example](https://github.com/deepset-ai/hayhooks/tree/main/examples/programmatic/api_key_auth) for a complete implementation
- **Reverse proxy auth** -- use Nginx, Traefik, or a cloud load balancer to handle authentication before requests reach Hayhooks
- **Custom middleware** -- add your own FastAPI middleware via [programmatic customization](../advanced/advanced-configuration.md)

!!! warning
    Without authentication, anyone who can reach your Hayhooks server can invoke pipelines and (if runtime deploy is enabled) deploy or undeploy them. At minimum, restrict network access to trusted sources.

## Set Up Health Checks

The `/status` endpoint returns the server status and list of deployed pipelines. Use it as a health check for container orchestrators:

```yaml
# Docker Compose
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:1416/status"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

For Kubernetes, use a liveness probe on the same endpoint:

```yaml
livenessProbe:
  httpGet:
    path: /status
    port: 1416
  initialDelaySeconds: 40
  periodSeconds: 30
```

## Docker and Container Tips

Follow these practices when running Hayhooks in containers:

- **Mount pipelines read-only** -- use `:ro` to prevent accidental writes:

    ```bash
    -v "$PWD/pipelines:/app/pipelines:ro"
    ```

- **Bind to `0.0.0.0`** -- the default `localhost` is not reachable from outside the container:

    ```bash
    -e HAYHOOKS_HOST=0.0.0.0
    ```

- **Set resource limits** -- prevent a single pipeline from consuming all host resources:

    ```yaml
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
    ```

- **Use a non-root user** -- if building a custom image, add a non-root user for defense in depth:

    ```dockerfile
    RUN useradd -m hayhooks
    USER hayhooks
    ```

- **Pin image tags** -- avoid `latest` or `main` in production. Use a specific release tag or SHA for reproducibility.

## Manage Environment Variables

- **Never hardcode secrets** (API keys, database passwords) in pipeline code or Docker Compose files. Use environment variables, mounted secret files, or your platform's secret manager.
- **Use `.env` files for local testing only.** In production, inject variables through your orchestrator (Docker secrets, Kubernetes ConfigMaps/Secrets, cloud provider secret stores).
- **Audit your configuration** before deploying. A quick check:

    ```bash
    # Verify no dev settings leaked into production
    env | grep HAYHOOKS_
    ```

## Recommended Production `.env`

For reference, a minimal production configuration:

```bash
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_PIPELINES_DIR=/app/pipelines
HAYHOOKS_SHOW_TRACEBACKS=false
HAYHOOKS_CORS_ALLOW_ORIGINS=["https://app.example.com"]
HAYHOOKS_CORS_ALLOW_CREDENTIALS=true
HAYHOOKS_STARTUP_DEPLOY_STRATEGY=parallel
HAYHOOKS_STARTUP_DEPLOY_WORKERS=4
LOG=INFO
```

## Next Steps

- [Deployment Guidelines](../deployment/deployment_guidelines.md) -- Infrastructure, Docker, Kubernetes, and scaling
- [Advanced Configuration](../advanced/advanced-configuration.md) -- Custom routes, middleware, and programmatic customization
- [Environment Variables Reference](../reference/environment-variables.md) -- Complete configuration reference
- [Development Best Practices](development-best-practices.md) -- Tips for local development workflow
