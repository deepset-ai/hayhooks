# Development Best Practices

This page collects the most useful tips for developing and debugging Hayhooks pipelines locally. Each section links to the full reference for deeper detail.

## Enable Stack Traces

By default, Hayhooks hides Python tracebacks from HTTP and MCP error responses. During development you almost always want them visible:

```bash
HAYHOOKS_SHOW_TRACEBACKS=true hayhooks run
```

Combine with `DEBUG` logging for maximum visibility:

```bash
LOG=DEBUG HAYHOOKS_SHOW_TRACEBACKS=true hayhooks run
```

!!! tip
    `LOG=DEBUG` prints every incoming request, deploy event, and internal pipeline step. Switch back to `INFO` once you no longer need the extra output.

See [Environment Variables Reference](../reference/environment-variables.md) for all available settings.

## Fast Iteration with `--overwrite`

Re-deploying a pipeline normally fails if a pipeline with the same name already exists. The `--overwrite` flag removes the old version first, so you can redeploy in one step:

```bash
hayhooks pipeline deploy-files -n my_pipeline --overwrite ./path/to/pipeline
```

For even faster cycles, add `--skip-saving-files` to skip writing the pipeline files to the server's pipeline directory:

```bash
hayhooks pipeline deploy-files -n my_pipeline --overwrite --skip-saving-files ./path/to/pipeline
```

**Typical workflow:**

1. Edit your `pipeline_wrapper.py`
2. Run the deploy command above
3. Test via the API or Swagger UI
4. Repeat

!!! info
    `--overwrite` and `--skip-saving-files` work with both `deploy-files` and `deploy-yaml`. See [CLI Commands](../features/cli-commands.md) for the full flag reference.

## Auto-Reload on Code Changes

If you are working on the Hayhooks server itself (not just on pipeline wrappers), you can start the server with `--reload` so it restarts automatically whenever the Hayhooks source files change:

```bash
hayhooks run --reload
```

!!! note
    `--reload` watches the Hayhooks **server** source code. It does **not** pick up changes to pipeline wrappers that are deployed at runtime -- for those, redeploy with `--overwrite` instead (see above).

!!! warning
    `--reload` is intended for local development only. It adds overhead and should never be used in production.

## Recommended Dev `.env`

Create a `.env` file in your project root with sensible development defaults:

```bash
# .env (development)
HAYHOOKS_HOST=localhost
HAYHOOKS_PORT=1416
HAYHOOKS_PIPELINES_DIR=./pipelines
LOG=DEBUG
HAYHOOKS_SHOW_TRACEBACKS=true
# Uncomment if you share code across wrappers:
# HAYHOOKS_ADDITIONAL_PYTHON_PATH=./common
```

Hayhooks loads `.env` automatically on startup. See [Configuration](../getting-started/configuration.md) for all configuration methods and [Code Sharing](../advanced/code-sharing.md) for organizing shared code across pipeline wrappers.

## Debugging Pipeline Errors

When a pipeline fails to deploy or returns an error at runtime, follow this checklist:

1. **Check the server logs** -- with `LOG=DEBUG` and `HAYHOOKS_SHOW_TRACEBACKS=true` the full traceback is printed to the console.
2. **Verify imports** -- a missing or misspelled import in `pipeline_wrapper.py` is the most common deploy failure. The traceback will show the exact `ImportError`.
3. **Check the wrapper signature** -- `setup()` must take no arguments (besides `self`), and `run_api()` / `run_api_async()` must match the expected parameter types. See [PipelineWrapper](../concepts/pipeline-wrapper.md) for the full API.
4. **Inspect the pipeline YAML** -- if you use a `pipeline.yml`, make sure component names and connections are valid. Deploy with `LOG=DEBUG` to see the Haystack deserialization output.
5. **Test components in isolation** -- run individual Haystack components in a plain Python script before wiring them into a wrapper.

!!! tip
    If you are unsure whether the issue is in Hayhooks or in your Haystack pipeline, try running the pipeline directly with `pipeline.run(...)` in a standalone script first.

## Using the REST API During Development

The Swagger UI at `http://localhost:1416/docs` is the quickest way to explore and test endpoints interactively. For scripted testing, use `curl`:

**Deploy a pipeline:**

```bash
curl -X POST http://localhost:1416/deploy-files \
  -F "name=my_pipeline" \
  -F "overwrite=true" \
  -F "files=@./my_pipeline/pipeline_wrapper.py"
```

**Invoke a pipeline:**

```bash
curl -X POST http://localhost:1416/my_pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Haystack?"}'
```

**List deployed pipelines:**

```bash
curl http://localhost:1416/status
```

See [API Reference](../reference/api-reference.md) for the complete endpoint documentation.

## Testing Streaming Locally

If your pipeline supports streaming, you can verify Server-Sent Events with `curl`. The `-N` flag (`--no-buffer`) disables output buffering so you see tokens as they arrive.

### Chat Completions API

For pipelines that implement `run_chat_completion` / `run_chat_completion_async`:

```bash
curl -N http://localhost:1416/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my_pipeline",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

You should see a stream of `data: {...}` lines followed by `data: [DONE]`.

### Responses API

For pipelines that implement `run_response` / `run_response_async`:

```bash
curl -N http://localhost:1416/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my_pipeline",
    "stream": true,
    "input": [{"role": "user", "type": "message", "content": [
      {"type": "input_text", "text": "Hello!"}
    ]}]
  }'
```

The Responses API uses named SSE events (e.g. `event: response.output_text.delta`), so the output format differs from Chat Completions.

!!! tip
    If you see the entire response arrive at once instead of streaming, check that your pipeline wrapper returns a streaming generator. See [OpenAI Compatibility](../features/openai-compatibility.md) for implementation details on both APIs.

## Next Steps

- [PipelineWrapper](../concepts/pipeline-wrapper.md) -- Full wrapper API and development workflow
- [CLI Commands](../features/cli-commands.md) -- All CLI flags and commands
- [Configuration](../getting-started/configuration.md) -- Configuration methods and examples
- [Production Best Practices](production-best-practices.md) -- Hardening your deployment for production
