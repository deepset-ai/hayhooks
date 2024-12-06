# Hayhooks deployment guidelines

This document describes how to deploy Hayhooks in a production environment.
Since Hayhooks is a FastAPI application, it can be deployed in a variety of ways as well described in [its documentation](https://fastapi.tiangolo.com/deployment/concepts/?h=deploy).

Following are some guidelines about deploying and running Haystack pipelines.

## TL;DR

- Use a single worker environment if you have mainly I/O operations in your pipeline and/or a low number of concurrent requests.
- Use a multi-worker environment if you have mainly CPU-bound operations in your pipeline and/or a high number of concurrent requests.
- In any case, use `HAYHOOKS_PIPELINES_DIR` to share pipeline definitions across workers (if possible).

## Single worker environment

In a single worker environment, you typically run the application using:

```bash
hayhooks run
```

command (or having a single Docker container running). This will launch a **single `uvicorn` worker** to serve the application.

### Pipelines deployment (single worker)

You can deploy a pipeline using:

```bash
hayhooks deploy
```

command or do a `POST /deploy` request.

### Handling concurrent requests (single worker)

The `run()` method of the pipeline instance is synchronous code, and it's executed using `run_in_threadpool` to avoid blocking the main async event loop.

- If your pipeline is doing **mainly I/O operations** (like making HTTP requests, reading/writing files, etc.), the single worker should be able to handle concurrent requests.
- If your pipeline is doing **mainly CPU-bound operations** (like computing embeddings), the GIL (Global Interpreter Lock) will prevent the worker from handling concurrent requests, so they will be queued.

## Multiple workers environment

### Single instance with multiple workers

Currently, `hayhooks run` command does not support multiple `uvicorn` workers. However, you can run multiple instances of the application using directly the `uvicorn` command or [FastAPI CLI](https://fastapi.tiangolo.com/fastapi-cli/#fastapi-run) using `fastapi run` command.

For example, if you have enough cores to run 4 workers, you can use the following command:

```bash
fastapi run src/hayhooks/server/app.py --workers 4
```

This vertical scaling approach allows you to handle more concurrent requests (depending on available resources).

### Multiple single-worker instances behind a load balancer

In a multi-worker environment (for example on a Kubernetes `Deployment`) you typically have a `LoadBalancer` Service, which distributes the traffic to a number of `Pod`s running the application (using `hayhooks run` command).

This horizontal scaling approach allows you to handle more concurrent requests.

### Pipeline deployment (multiple workers)

In both the above scenarios, **it's NOT recommended** to deploy a pipeline using the `hayhooks deploy` command (or `POST /deploy` request) as it will deploy the pipeline only on one of the workers, which is not ideal.

Instead, you want to provide the env var `HAYHOOKS_PIPELINES_DIR` pointing to a shared folder where all the workers can read the pipeline definitions at startup and load them. This way, all the workers will have the same pipelines available and there will be no issues when calling the API to run a pipeline.

### Handling concurrent requests (multiple workers)

When having multiple workers and pipelines deployed using `HAYHOOKS_PIPELINES_DIR`, you will be able to handle concurrent requests as each worker will be able to run a pipeline independently. This should be enough to make your application scalable, according to your needs.

Note that even in a multiple-workers environment the individual single workers will have the same GIL limitation discussed above, so if your pipeline is mainly CPU-bound, you will need to scale horizontally according to your needs.
