# Running Pipelines

Execute deployed pipelines via CLI, HTTP API, or programmatically.

## Quick Reference

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

=== "Open WebUI"
    See [Open WebUI Integration](../features/openwebui-integration.md) for setup details.

For more CLI examples, see [CLI Commands](../features/cli-commands.md).

## File Uploads

### CLI

```bash
# Single file
hayhooks pipeline run my_pipeline --file document.pdf --param 'query="Summarize"'

# Multiple files
hayhooks pipeline run my_pipeline --file doc1.pdf --file doc2.txt

# Directory
hayhooks pipeline run my_pipeline --dir ./documents
```

### HTTP

```bash
curl -X POST http://localhost:1416/my_pipeline/run \
  -F 'files=@document.pdf' \
  -F 'query="Summarize this document"'
```

See [File Upload Support](../features/file-upload-support.md) for implementation details.

## Python Integration

### Requests

```python
import requests

resp = requests.post(
    "http://localhost:1416/my_pipeline/run",
    json={"query": "What is Haystack?"}
)
print(resp.json())
```

### Async (httpx)

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

## Streaming

Implement `run_chat_completion` or `run_chat_completion_async` in your wrapper. See [OpenAI Compatibility](../features/openai-compatibility.md) for details.

## Error Handling & Retry Logic

```python
import requests
from requests.exceptions import RequestException
import time

def run_with_retry(pipeline_name, params, max_retries=3):
    url = f"http://localhost:1416/{pipeline_name}/run"

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Logging

Add logging to your pipeline wrappers:

```python
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, query: str) -> str:
        log.info("Processing query: {}", query)
        result = self.pipeline.run({"prompt": {"query": query}})
        log.info("Pipeline completed")
        return result["llm"]["replies"][0]
```

See [Logging Reference](../reference/logging.md) for details.

## Next Steps

- [CLI Commands](../features/cli-commands.md) - Complete CLI reference
- [Examples](../examples/overview.md) - Working examples
