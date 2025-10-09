# Deployment Guidelines

This guide describes how to deploy Hayhooks in production environments.

Since Hayhooks is a FastAPI application, you can deploy it using any standard ASGI server deployment strategy. For comprehensive deployment concepts, see the [FastAPI deployment documentation](https://fastapi.tiangolo.com/deployment/concepts/).

This guide focuses on Hayhooks-specific considerations for production deployments.

## Quick Recommendations

- Use `HAYHOOKS_PIPELINES_DIR` to deploy pipelines in production environments
- Start with a single worker for I/O-bound pipelines, use multiple workers for CPU-bound workloads
- Implement async methods (`run_api_async`) for better I/O performance
- Configure health checks for container orchestration
- Set appropriate resource limits and environment variables
- Review security settings (CORS, tracebacks, logging levels)

!!! info "Configuration Resources"
    Review [Configuration](../getting-started/configuration.md) and [Environment Variables Reference](../reference/environment-variables.md) before deploying.

## Pipeline Deployment Strategy

For production deployments, use `HAYHOOKS_PIPELINES_DIR` to deploy pipelines at startup.

### Using HAYHOOKS_PIPELINES_DIR

Set the environment variable to point to a directory containing your pipeline definitions:

```bash
export HAYHOOKS_PIPELINES_DIR=/app/pipelines
hayhooks run
```

When Hayhooks starts, it automatically loads all pipelines from this directory.

**Benefits:**

- Pipelines are available immediately on startup
- Consistent across all workers/instances
- No runtime deployment API calls needed
- Simple to version control and deploy

**Directory structure:**

```text
pipelines/
├── my_pipeline/
│   ├── pipeline_wrapper.py
│   └── pipeline.yml
└── another_pipeline/
    ├── pipeline_wrapper.py
    └── pipeline.yml
```

See [YAML Pipeline Deployment](../concepts/yaml-pipeline-deployment.md) and [PipelineWrapper](../concepts/pipeline-wrapper.md) for file structure details.

!!! tip "Development vs Production"
    For local development, you can use CLI commands (`hayhooks pipeline deploy-files`) or API endpoints (`POST /deploy-files`). For production, always use `HAYHOOKS_PIPELINES_DIR`.

## Performance Tuning

### Single Worker vs Multiple Workers

**Single Worker Environment:**

```bash
hayhooks run
```

Best for:

- Development and testing
- I/O-bound pipelines (HTTP requests, file operations, database queries)
- Low to moderate concurrent requests
- Simpler deployment and debugging

**Multiple Workers Environment:**

```bash
hayhooks run --workers 4
```

Best for:

- CPU-bound pipelines (embedding generation, heavy computation)
- High concurrent request volumes
- Production environments with available CPU cores

!!! tip "Worker Count Formula"
    A common starting point: `workers = (2 x CPU_cores) + 1`. Adjust based on your workload - I/O-bound: More workers can help; CPU-bound: Match CPU cores to avoid context switching overhead.

### Concurrency Behavior

Pipeline `run()` methods execute synchronously but are wrapped in `run_in_threadpool` to avoid blocking the async event loop.

**I/O-bound pipelines** (HTTP requests, file operations, database queries):

- Can handle concurrent requests effectively in a single worker
- Worker switches between tasks during I/O waits
- Consider implementing async methods for even better performance

**CPU-bound pipelines** (embedding generation, heavy computation):

- Limited by Python's Global Interpreter Lock (GIL)
- Requests are queued and processed sequentially in a single worker
- Use multiple workers or horizontal scaling to improve throughput

### Async Pipelines

Implement async methods for better I/O-bound performance:

```python
from haystack import AsyncPipeline
from hayhooks import BasePipelineWrapper

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = AsyncPipeline.loads(
            (Path(__file__).parent / "pipeline.yml").read_text()
        )

    async def run_api_async(self, query: str) -> str:
        result = await self.pipeline.run_async({"prompt": {"query": query}})
        return result["llm"]["replies"][0]
```

### Streaming

Use streaming for chat endpoints to reduce perceived latency:

```python
from hayhooks import async_streaming_generator, get_last_user_message

async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict):
    question = get_last_user_message(messages)
    return async_streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt": {"query": question}},
    )
```

See [OpenAI Compatibility](../features/openai-compatibility.md) for more details on streaming.

### Horizontal Scaling

Deploy multiple instances behind a load balancer for increased throughput.

**Key considerations:**

- Use `HAYHOOKS_PIPELINES_DIR` to ensure all instances have the same pipelines
- Configure session affinity if using stateful components
- Distribute traffic evenly across instances
- Monitor individual instance health

**Example setup** (Docker Swarm, Kubernetes, or cloud load balancers):

```bash
# Each instance should use the same pipeline directory
export HAYHOOKS_PIPELINES_DIR=/app/pipelines
hayhooks run
```

!!! info "GIL Limitations"
    Even with multiple workers, individual workers have GIL limitations. CPU-bound pipelines benefit more from horizontal scaling (multiple instances) than vertical scaling (multiple workers per instance).

## Docker Deployment

### Single Container

```bash
docker run -d \
  -p 1416:1416 \
  -e HAYHOOKS_HOST=0.0.0.0 \
  -e HAYHOOKS_PIPELINES_DIR=/app/pipelines \
  -v "$PWD/pipelines:/app/pipelines:ro" \
  deepset/hayhooks:main
```

### Docker Compose

```yaml
version: '3.8'
services:
  hayhooks:
    image: deepset/hayhooks:main
    ports:
      - "1416:1416"
    environment:
      HAYHOOKS_HOST: 0.0.0.0
      HAYHOOKS_PIPELINES_DIR: /app/pipelines
      LOG: INFO
    volumes:
      - ./pipelines:/app/pipelines:ro
    restart: unless-stopped
```

See [Quick Start with Docker Compose](../getting-started/quick-start-docker.md) for a complete example with Open WebUI integration.

### Health Checks

Add health checks to monitor container health:

```yaml
services:
  hayhooks:
    image: deepset/hayhooks:main
    ports:
      - "1416:1416"
    environment:
      HAYHOOKS_HOST: 0.0.0.0
      HAYHOOKS_PIPELINES_DIR: /app/pipelines
    volumes:
      - ./pipelines:/app/pipelines:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:1416/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
```

The `/status` endpoint returns the server status and can be used for health monitoring.

## Production Deployment Options

### Docker

Deploy Hayhooks using Docker containers for consistent, portable deployments across environments. Docker provides isolation, easy versioning, and simplified dependency management. See the [Docker documentation](https://docs.docker.com/get-started/) for container deployment best practices.

### Kubernetes

Deploy Hayhooks on Kubernetes for automated scaling, self-healing, and advanced orchestration capabilities. Use Deployments, Services, and ConfigMaps to manage pipeline definitions and configuration. See the [Kubernetes documentation](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) for deployment strategies.

### Server/VPS Deployment

Deploy Hayhooks directly on a server or VPS using systemd or process managers like supervisord for production reliability. This approach offers full control over the environment and is suitable for dedicated workloads. See the [FastAPI deployment documentation](https://fastapi.tiangolo.com/deployment/manually/) for manual deployment guidance.

### AWS ECS

Deploy Hayhooks on AWS Elastic Container Service for managed container orchestration in the AWS ecosystem. ECS handles container scheduling, load balancing, and integrates seamlessly with other AWS services. See the [AWS ECS documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html) for deployment details.

## Production Best Practices

### Environment Variables

Store sensitive configuration in environment variables or secrets:

```bash
# Use a .env file
HAYHOOKS_PIPELINES_DIR=/app/pipelines
LOG=INFO
HAYHOOKS_SHOW_TRACEBACKS=false
```

See [Environment Variables Reference](../reference/environment-variables.md) for all options.

### Logging

Configure appropriate log levels for production:

```bash
# Production: INFO or WARNING
export LOG=INFO

# Development: DEBUG
export LOG=DEBUG
```

See [Logging](../reference/logging.md) for details.

### CORS Configuration

Configure CORS for production environments:

```bash
# Restrict to specific origins
export HAYHOOKS_CORS_ALLOW_ORIGINS='["https://yourdomain.com"]'
export HAYHOOKS_CORS_ALLOW_CREDENTIALS=true
```

## Troubleshooting

### Pipeline Not Available

If pipelines aren't available after startup:

1. Check `HAYHOOKS_PIPELINES_DIR` is correctly set
2. Verify pipeline files exist in the directory
3. Check logs for deployment errors: `docker logs <container_id>`
4. Verify pipeline wrapper syntax and imports

### High Memory Usage

For memory-intensive pipelines:

1. Increase container memory limits in Docker Compose
2. Profile pipeline components for memory leaks
3. Optimize component initialization and caching
4. Consider using smaller models or batch sizes

### Slow Response Times

For performance issues:

1. Check component initialization in `setup()` vs `run_api()`
2. Verify pipeline directory is mounted correctly
3. Review logs for errors or warnings
4. Consider implementing async methods or adding workers (see [Performance Tuning](#performance-tuning) above)

## Next Steps

- [Advanced Configuration](../advanced/advanced-configuration.md) - Custom routes, middleware, and programmatic customization
- [Environment Variables Reference](../reference/environment-variables.md) - Complete configuration reference
- [Pipeline Deployment](../concepts/pipeline-deployment.md) - Pipeline deployment concepts
- [Quick Start with Docker Compose](../getting-started/quick-start-docker.md) - Complete Docker Compose example
