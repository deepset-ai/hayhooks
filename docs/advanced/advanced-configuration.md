# Advanced Configuration

This guide covers performance tuning, custom routes/middleware, and deployment optimization.

For basic configuration, see [Configuration](../getting-started/configuration.md). For all environment variables, see [Environment Variables Reference](../reference/environment-variables.md).

## Performance Tuning

### Multiple Workers

Scale vertically for CPU-bound or high-concurrency workloads:

```bash
hayhooks run --workers 4
```

### Async Pipelines

Use async methods for I/O-bound operations:

```python
from haystack import AsyncPipeline

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = AsyncPipeline.loads((Path(__file__).parent / "pipeline.yml").read_text())

    async def run_api_async(self, query: str) -> str:
        result = await self.pipeline.run_async({"prompt": {"query": query}})
        return result["llm"]["replies"][0]
```

### Streaming

Use streaming for chat endpoints to reduce latency:

```python
from hayhooks import async_streaming_generator, get_last_user_message

async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict):
    question = get_last_user_message(messages)
    return async_streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt": {"query": question}},
    )
```

### Horizontal Scaling

Scale horizontally behind a load balancer:

## Custom Routes and Middleware

### When to add custom routes

- Add specialized endpoints for application-specific logic
- Provide admin/operations endpoints (restart, status, maintenance tasks)
- Expose health checks, metrics, and webhook handlers for integrations
- Implement authentication/authorization flows
- Offer file management or other utility endpoints

### When to add middleware

- Apply cross-cutting concerns (logging/tracing, correlation IDs)
- Enforce security controls (authn/z, rate limiting, quotas)
- Control headers, CORS, compression, and caching
- Normalize inputs/outputs and error handling consistently

For implementation examples of adding routes or middleware, see the README section “Run Hayhooks programmatically”.

## Programmatic Customization

You can create a custom Hayhooks app instance to add routes or middleware:

```python
import uvicorn
from hayhooks.settings import settings
from fastapi import Request
from hayhooks import create_app

# Create the Hayhooks app
hayhooks = create_app()

# Add a custom route
@hayhooks.get("/custom")
async def custom_route():
    return {"message": "Custom route"}

# Add custom middleware
@hayhooks.middleware("http")
async def custom_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Custom-Header"] = "value"
    return response

if __name__ == "__main__":
    uvicorn.run("app:hayhooks", host=settings.host, port=settings.port)
```

See the README section "Run Hayhooks programmatically" for more details.

## Next Steps

- [Code Sharing](code-sharing.md) - Reusable components
