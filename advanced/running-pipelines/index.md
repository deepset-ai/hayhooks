# Running Pipelines

Execute deployed pipelines via CLI, HTTP API, or programmatically.

## Quick Reference

```
hayhooks pipeline run my_pipeline --param 'query=What is Haystack?'
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
hayhooks pipeline run my_pipeline --file document.pdf --param 'query=Summarize'

# Multiple files
hayhooks pipeline run my_pipeline --file doc1.pdf --file doc2.txt

# Directory
hayhooks pipeline run my_pipeline --dir ./documents
```

### HTTP

```
curl -X POST http://localhost:1416/my_pipeline/run \
  -F 'files=@document.pdf' \
  -F 'query=Summarize this document'
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

> **Sending serialized Haystack objects?** If a request carries serialized `ChatMessage`, `Document`, or similar objects (for example through a `dict[str, Any]` input), see [Deserializing loosely-typed inputs](https://deepset-ai.github.io/hayhooks/concepts/pipeline-wrapper/#deserializing-loosely-typed-inputs) for how your wrapper turns them back into instances before running the pipeline.

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

For ordinary REST calls, return a generator from `run_api` or `run_api_async` and use `hayhooks pipeline run my_pipeline --stream`. See [streaming responses](https://deepset-ai.github.io/hayhooks/concepts/pipeline-wrapper/#streaming-responses). For OpenAI clients, implement `run_chat_completion` or `run_chat_completion_async` and send `"stream": true` in the request. See [OpenAI Compatibility](https://deepset-ai.github.io/hayhooks/features/openai-compatibility/index.md).

## Error Handling & Retry Logic

Use retries only when repeating the pipeline is safe. A failed HTTP response does not prove that execution had no side effects. For detached work, prefer [durable submissions with idempotency keys](https://deepset-ai.github.io/hayhooks/features/durable-execution/#rest-sse-and-ownership).

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
from hayhooks import BasePipelineWrapper, log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, query: str) -> str:
        log.info("Processing query: {}", query)
        result = self.pipeline.run({"prompt": {"query": query}})
        log.info("Pipeline completed")
        return result["llm"]["replies"][0].text
```

See [Logging Reference](https://deepset-ai.github.io/hayhooks/reference/logging/index.md) for details.

## Next Steps

- [CLI Commands](https://deepset-ai.github.io/hayhooks/features/cli-commands/index.md) - Complete CLI reference
- [Examples](https://deepset-ai.github.io/hayhooks/examples/overview/index.md) - Working examples
