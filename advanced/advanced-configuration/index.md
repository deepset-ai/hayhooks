# Advanced Configuration

This guide covers programmatic customization, custom routes, and middleware for advanced Hayhooks usage.

For basic configuration, see [Configuration](https://deepset-ai.github.io/hayhooks/getting-started/configuration/index.md). For deployment and performance tuning, see [Deployment Guidelines](https://deepset-ai.github.io/hayhooks/deployment/deployment_guidelines/index.md).

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

## Programmatic Customization

You can create a custom Hayhooks app instance to add routes or middleware:

```
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

This allows you to build custom applications with Hayhooks as the core engine while adding your own business logic and integrations.

## Next Steps

- [Deployment Guidelines](https://deepset-ai.github.io/hayhooks/deployment/deployment_guidelines/index.md) - Performance tuning, workers, scaling, and deployment strategies
- [Code Sharing](https://deepset-ai.github.io/hayhooks/advanced/code-sharing/index.md) - Reusable components across pipelines
