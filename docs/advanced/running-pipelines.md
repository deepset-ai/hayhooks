# Running Pipelines

This section covers how to run Haystack pipelines deployed with Hayhooks through various interfaces including CLI, HTTP API, and programmatic access.

## Overview

Hayhooks provides multiple ways to execute your deployed pipelines:

- **CLI Commands**: Easy command-line execution
- **HTTP API**: RESTful API for integration
- **Programmatic**: Direct Python API access
- **File Uploads**: Handle file uploads with pipelines

## Quick Run

Use tabs to pick your preferred interface:

=== "HayhooksCLI"

    ```bash
    hayhooks pipeline run my_pipeline --param 'query="What is Haystack?"'
    ```

=== "Hayhooks HTTP API"

    ```bash
    curl -X POST http://localhost:1416/my_pipeline/run \
      -H 'Content-Type: application/json' \
      -d '{"query":"What is Haystack?"}'
    ```

=== "Python"

    ```python
    import requests

    resp = requests.post(
        "http://localhost:1416/my_pipeline/run",
        json={"query": "What is Haystack?"}
    )
    print(resp.json())
    ```

=== "Async Python"

    ```python
    import httpx
    import asyncio

    async def main():
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "http://localhost:1416/my_pipeline/run",
                json={"query": "What is Haystack?"}
            )
            print(r.json())

    asyncio.run(main())
    ```

=== "OpenAI Chat Completion (`curl`)"

    ```bash
    curl -X POST http://localhost:1416/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "my_pipeline",
        "messages": [{"role": "user", "content": "What is Haystack?"}]
      }'
    ```

=== "OpenAI Chat Completion (OpenWebUI)"
    Please refer to the [OpenWebUI Integration](../features/openwebui-integration.md) page for more information.

## CLI Execution

### Basic Pipeline Run

Execute a pipeline with simple parameters:

    ```bash
    # Run with JSON-compatible parameters
    hayhooks pipeline run my_pipeline --param 'query="What is Haystack?"'

    # Run with multiple parameters
    hayhooks pipeline run my_pipeline --param 'query="What is Haystack?"' --param 'temperature=0.7'

    # Run with complex JSON
    hayhooks pipeline run my_pipeline --param 'urls=["https://example.com"]' --param 'question="What is this about?"'
    ```

### File Upload Execution

Run pipelines with file uploads:

    ```bash
    # Upload single file
    hayhooks pipeline run rag_pipeline --file document.pdf --param 'query="Summarize this document"'

    # Upload directory
    hayhooks pipeline run rag_pipeline --dir ./documents --param 'query="Analyze all documents"'

    # Upload multiple files
    hayhooks pipeline run rag_pipeline --file doc1.pdf --file doc2.txt --param 'query="Compare these documents"'

    # Upload with additional parameters
    hayhooks pipeline run rag_pipeline --file document.pdf --param 'query="Analyze"' --param 'temperature=0.5'
    ```

## HTTP API Execution

### REST API Endpoints

Pipelines are accessible via HTTP endpoints:

```bash
# Basic API call
curl -X POST \
  http://localhost:1416/my_pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is Haystack?"}'

# With custom headers
curl -X POST \
  http://localhost:1416/my_pipeline/run \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer token' \
  -d '{"query": "What is Haystack?"}'
```

### Multipart Form Data

For file uploads:

```bash
# Upload files with curl
curl -X POST \
  http://localhost:1416/my_pipeline/run \
  -F 'files=@document.pdf' \
  -F 'query="Summarize this document"'

# Multiple files
curl -X POST \
  http://localhost:1416/my_pipeline/run \
  -F 'files=@doc1.pdf' \
  -F 'files=@doc2.txt' \
  -F 'query="Compare these documents"'
```

### OpenAI-Compatible Endpoints

For pipelines with chat completion support:

```bash
# Chat completion
curl -X POST \
  http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my_pipeline",
    "messages": [
      {"role": "user", "content": "What is Haystack?"}
    ]
  }'

# Streaming chat completion
curl -X POST \
  http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my_pipeline",
    "messages": [
      {"role": "user", "content": "What is Haystack?"}
    ],
    "stream": true
  }'
```

## Programmatic Execution

### Python (requests)

    ```python
    import requests

    # Simple pipeline run
    resp = requests.post(
        "http://localhost:1416/my_pipeline/run",
        json={"query": "What is Haystack?"}
    )
    print(resp.json())

    # Chat completion
    resp = requests.post(
        "http://localhost:1416/v1/chat/completions",
        json={
            "model": "chat_pipeline",
            "messages": [{"role": "user", "content": "What is Haystack?"}]
        }
    )
    print(resp.json())
    ```

### Async Python (httpx)

```python
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient() as client:
        # Simple pipeline run
        r = await client.post(
            "http://localhost:1416/my_pipeline/run",
            json={"query": "What is Haystack?"}
        )
        print(r.json())

        # Chat completion
        r = await client.post(
            "http://localhost:1416/v1/chat/completions",
            json={
                "model": "chat_pipeline",
                "messages": [{"role": "user", "content": "What is Haystack?"}]
            }
        )
        print(r.json())

asyncio.run(main())
```

## Advanced Execution Patterns

### Streaming Responses

For OpenAI-compatible streaming (aligned with README), implement `run_chat_completion` or `run_chat_completion_async` in your `PipelineWrapper` and call the OpenAI-compatible endpoints (`{pipeline_name}/chat` or `/v1/chat/completions`). See README sections “OpenAI compatibility and open-webui integration” and “Streaming responses in OpenAI-compatible endpoints” for canonical examples using `streaming_generator` and `async_streaming_generator`.

### Error Handling

Implement robust error handling:

    ```python
    import requests
    from requests.exceptions import RequestException

    class PipelineExecutionError(Exception):
        pass

    def run_pipeline_with_retry(pipeline_name, params, max_retries=3):
        """Run pipeline with retry logic"""
        url = f"http://localhost:1416/{pipeline_name}/run"

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=params)
                response.raise_for_status()
                return response.json()

            except RequestException as e:
                if attempt == max_retries - 1:
                    raise PipelineExecutionError(f"Failed after {max_retries} attempts: {e}")

                print(f"Attempt {attempt + 1} failed, retrying...")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff

        raise PipelineExecutionError("Max retries exceeded")

    # Usage
    try:
        result = run_pipeline_with_retry(
            "my_pipeline",
            {"query": "What is Haystack?"}
        )
        print(result)
    except PipelineExecutionError as e:
        print(f"Pipeline execution failed: {e}")
    ```

## File Upload Processing

### File uploads with requests

```python
import requests

# Upload one or more files
files = [
    ('files', open('document.pdf', 'rb')),
    # ('files', open('another.txt', 'rb')),
]
try:
    resp = requests.post(
        'http://localhost:1416/rag_pipeline/run',
        files=files,
        data={'query': 'Summarize this document'}
    )
    print(resp.json())
finally:
    for _, fh in files:
        fh.close()
```

### Upload a directory

    ```python
    from pathlib import Path
    import requests

    dir_path = Path('./documents')
    file_list = [('files', open(str(p), 'rb')) for p in dir_path.rglob('*') if p.is_file()]
    try:
        resp = requests.post(
            'http://localhost:1416/rag_pipeline/run',
            files=file_list,
            data={'query': 'Analyze all documents'}
        )
        print(resp.json())
    finally:
        for _, fh in file_list:
            fh.close()
    ```

## Monitoring and Logging

### Server-side execution logging

Use Hayhooks' logger in your `PipelineWrapper` for consistent, structured logs:

    ```python
    import time
    from hayhooks import log

    class PipelineWrapper(BasePipelineWrapper):
        def run_api(self, query: str) -> str:
            start_time = time.time()
            log.info({"event": "pipeline_start", "pipeline": "my_pipeline", "query_len": len(query)})

            try:
                result = self.pipeline.run({"prompt": {"query": query}})
                duration = time.time() - start_time
                log.info({
                    "event": "pipeline_complete",
                    "pipeline": "my_pipeline",
                    "duration_s": round(duration, 3)
                })
                return result["llm"]["replies"][0]
            except Exception as e:
                duration = time.time() - start_time
                log.error({
                    "event": "pipeline_error",
                    "pipeline": "my_pipeline",
                    "duration_s": round(duration, 3),
                    "error": str(e)
                })
                raise
    ```

## Best Practices

### 1. Error Handling

- Always check response status codes
- Implement retry logic for transient failures
- Use proper exception handling
- Log errors for debugging

### 2. Performance

- Use async clients for high-throughput scenarios
- Implement connection pooling
- Use appropriate timeouts
- Monitor resource usage

### 3. Security

- Use HTTPS in production
- Implement authentication
- Validate input parameters
- Use secure file handling

### 4. Monitoring

- Track execution times
- Monitor success/failure rates
- Log important events
- Set up alerts for failures

## Next Steps

- [CLI Commands](../features/cli-commands.md) - CLI usage details
- [File Upload Support](../features/file-upload-support.md) - File upload handling
- [Examples](../examples/overview.md) - Working examples
