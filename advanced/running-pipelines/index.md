# Running Pipelines

Execute deployed pipelines via CLI, HTTP API, or programmatically.

## Quick Reference

```
hayhooks pipeline run my_pipeline --param 'query="What is Haystack?"'
```

```
curl -X POST http://localhost:1416/my_pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is Haystack?"}'
```

```
import requests

resp = requests.post(
    "http://localhost:1416/my_pipeline/run",
    json={"query": "What is Haystack?"}
)
print(resp.json())
```

```
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

```
curl -X POST http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my_pipeline",
    "messages": [{"role": "user", "content": "What is Haystack?"}]
  }'
```

See [Open WebUI Integration](https://deepset-ai.github.io/hayhooks/features/openwebui-integration/index.md) for setup details.

For more CLI examples, see [CLI Commands](https://deepset-ai.github.io/hayhooks/features/cli-commands/index.md).

## File Uploads

### CLI

```
# Single file
hayhooks pipeline run my_pipeline --file document.pdf --param 'query="Summarize"'

# Multiple files
hayhooks pipeline run my_pipeline --file doc1.pdf --file doc2.txt

# Directory
hayhooks pipeline run my_pipeline --dir ./documents
```

### HTTP

```
curl -X POST http://localhost:1416/my_pipeline/run \
  -F 'files=@document.pdf' \
  -F 'query="Summarize this document"'
```

See [File Upload Support](https://deepset-ai.github.io/hayhooks/features/file-upload-support/index.md) for implementation details.

## Python Integration

### Requests

```
import requests

resp = requests.post(
    "http://localhost:1416/my_pipeline/run",
    json={"query": "What is Haystack?"}
)
print(resp.json())
```

### Async (httpx)

```
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

Implement `run_chat_completion` or `run_chat_completion_async` in your wrapper. See [OpenAI Compatibility](https://deepset-ai.github.io/hayhooks/features/openai-compatibility/index.md) for details.

## Error Handling & Retry Logic

```
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

```
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, query: str) -> str:
        log.info("Processing query: {}", query)
        result = self.pipeline.run({"prompt": {"query": query}})
        log.info("Pipeline completed")
        return result["llm"]["replies"][0]
```

See [Logging Reference](https://deepset-ai.github.io/hayhooks/reference/logging/index.md) for details.

## Next Steps

- [CLI Commands](https://deepset-ai.github.io/hayhooks/features/cli-commands/index.md) - Complete CLI reference
- [Examples](https://deepset-ai.github.io/hayhooks/examples/overview/index.md) - Working examples
