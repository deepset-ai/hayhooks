# Hayhooks deployment guidelines

This document describes how to deploy Hayhooks in a production environment.
Since Hayhooks is a FastAPI application, it can be deployed in a variety of ways as well described in [its documentation](https://fastapi.tiangolo.com/deployment/concepts/?h=deploy).

Following are some guidelines about deploying and running Haystack pipelines.

## TL;DR

- Use a **single worker environment** if you have mainly I/O operations in your pipeline and/or a low number of concurrent requests.
- Use a **multi-worker environment** if you have mainly CPU-bound operations in your pipeline and/or a high number of concurrent requests.
- In any case, use `HAYHOOKS_PIPELINES_DIR` to share pipeline definitions across workers (if possible).
- You can use [any additional supported `uvicorn` environment variables](https://www.uvicorn.org/settings) to the `hayhooks run` command (or put them in a `.env` file).

## Single worker environment

In a single worker environment, you typically run the application using:

```bash
hayhooks run
```

command (or having a single Docker container running). This will launch a **single `uvicorn` worker** to serve the application.

### Pipelines deployment (single worker)

You can deploy a pipeline using:

```bash
hayhooks deploy-files # recommended

# or

hayhooks deploy ...
```

or make `POST /deploy` / `POST /deploy-files` requests.

### Handling concurrent requests (single worker)

The `run()` method of the pipeline instance is _synchronous_ code, and it's executed using `run_in_threadpool` to avoid blocking the main async event loop.

- If your pipeline is doing **mainly I/O operations** (like making HTTP requests, reading/writing files, etc.), the single worker should be able to handle concurrent requests.
- If your pipeline is doing **mainly CPU-bound operations** (like computing embeddings), the GIL (Global Interpreter Lock) will prevent the worker from handling concurrent requests, so they will be queued.

## Multiple workers environment

### Using `uvicorn` with multiple workers

Hayhooks supports multiple `uvicorn` workers running on a single instance, you can use the `hayhooks run` command with the `--workers` flag to start the application with the desired number of workers.

For example, if you have enough cores to run 4 workers, you can use the following command:

```bash
hayhooks run --workers 4
```

This vertical scaling approach allows you to handle more concurrent requests (depending on environment available resources).

### Multiple single-worker instances behind a load balancer

In a multi-worker environment (for example on a Kubernetes `Deployment`) you typically have a `LoadBalancer` Service, which distributes the traffic to a number of `Pod`s running the application (using `hayhooks run` command).

This horizontal scaling approach allows you to handle more concurrent requests.

### Pipeline deployment (multiple workers)

In both the above scenarios, **it's NOT recommended** to deploy a pipeline using Hayhooks CLI commands (or corresponding API requests) as **it will deploy the pipeline only on one of the workers**, which is not ideal.

Instead, set the environment variable `HAYHOOKS_PIPELINES_DIR` to point to a shared directory accessible by all workers. When Hayhooks starts up, each worker will load pipeline definitions from this shared location, ensuring consistent pipeline availability across all workers when handling API requests.

### Handling concurrent requests (multiple workers)

When having multiple workers and pipelines deployed using `HAYHOOKS_PIPELINES_DIR`, you will be able to handle concurrent requests as each worker should be able to run a pipeline independently. This may be enough to make your application scalable, according to your needs.

Note that even in a multiple-workers environment, the individual single worker will have the same GIL limitations discussed above, so if your pipeline is mainly CPU-bound, you will need to scale horizontally according to your needs.
