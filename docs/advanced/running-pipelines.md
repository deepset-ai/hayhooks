# Running Pipelines

This section covers how to run Haystack pipelines through various interfaces including CLI, HTTP API, and programmatic access.

## Overview

Hayhooks provides multiple ways to execute your deployed pipelines:

- **CLI Commands**: Easy command-line execution
- **HTTP API**: RESTful API for integration
- **Programmatic**: Direct Python API access
- **File Uploads**: Handle file uploads with pipelines
- **Batch Processing**: Execute multiple requests

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

### Batch Processing

Execute multiple pipeline runs:

```bash
# Create a batch file
cat > batch.txt << EOF
{"query": "What is Haystack?"}
{"query": "How does Hayhooks work?"}
{"query": "What are the benefits?"}
EOF

# Execute batch
while IFS= read -r line; do
  hayhooks pipeline run my_pipeline --param "query=$line"
done < batch.txt
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

### Python Client

```python
import requests
import json

class HayhooksClient:
    def __init__(self, base_url="http://localhost:1416"):
        self.base_url = base_url

    def run_pipeline(self, pipeline_name, params=None, files=None):
        """Run a pipeline with optional parameters and files"""
        url = f"{self.base_url}/{pipeline_name}/run"

        if files:
            # Multipart form data for file uploads
            files_data = [('files', open(f, 'rb')) for f in files]
            data = params or {}
            response = requests.post(url, files=files_data, data=data)
        else:
            # JSON data for regular requests
            response = requests.post(url, json=params or {})

        response.raise_for_status()
        return response.json()

    def run_chat_completion(self, pipeline_name, messages, stream=False):
        """Run chat completion"""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": pipeline_name,
            "messages": messages,
            "stream": stream
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

# Usage example
client = HayhooksClient()

# Simple pipeline run
result = client.run_pipeline(
    "my_pipeline",
    params={"query": "What is Haystack?"}
)
print(result)

# Run with files
result = client.run_pipeline(
    "rag_pipeline",
    params={"query": "Summarize this document"},
    files=["document.pdf"]
)
print(result)

# Chat completion
result = client.run_chat_completion(
    "chat_pipeline",
    messages=[{"role": "user", "content": "What is Haystack?"}]
)
print(result)
```

### Async Python Client

```python
import aiohttp
import asyncio

class AsyncHayhooksClient:
    def __init__(self, base_url="http://localhost:1416"):
        self.base_url = base_url

    async def run_pipeline(self, pipeline_name, params=None, files=None):
        """Async pipeline execution"""
        url = f"{self.base_url}/{pipeline_name}/run"

        async with aiohttp.ClientSession() as session:
            if files:
                # Handle file uploads
                data = aiohttp.FormData()
                for file_path in files:
                    data.add_field('files', open(file_path, 'rb'))

                for key, value in (params or {}).items():
                    data.add_field(key, str(value))

                async with session.post(url, data=data) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                # JSON request
                async with session.post(url, json=params or {}) as response:
                    response.raise_for_status()
                    return await response.json()

    async def run_chat_completion(self, pipeline_name, messages, stream=False):
        """Async chat completion"""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": pipeline_name,
            "messages": messages,
            "stream": stream
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

# Usage example
async def main():
    client = AsyncHayhooksClient()

    result = await client.run_pipeline(
        "my_pipeline",
        params={"query": "What is Haystack?"}
    )
    print(result)

asyncio.run(main())
```

## Advanced Execution Patterns

### Streaming Responses

For OpenAI-compatible streaming (aligned with README), implement `run_chat_completion` or `run_chat_completion_async` in your `PipelineWrapper` and call the OpenAI-compatible endpoints (`{pipeline_name}/chat` or `/v1/chat/completions`). See README sections “OpenAI compatibility and open-webui integration” and “Streaming responses in OpenAI-compatible endpoints” for canonical examples using `streaming_generator` and `async_streaming_generator`.

### Batch Processing

Process multiple requests efficiently:

```python
import concurrent.futures
import requests

def run_batch_pipelines(pipeline_name, requests_list, max_workers=5):
    """Run multiple pipeline requests in parallel"""

    def single_request(params):
        url = f"http://localhost:1416/{pipeline_name}/run"
        response = requests.post(url, json=params)
        return response.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_request, params) for params in requests_list]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(f"Completed: {result}")
            except Exception as e:
                print(f"Error: {e}")

# Usage
requests_list = [
    {"query": "What is Haystack?"},
    {"query": "How does it work?"},
    {"query": "What are the benefits?"}
]

run_batch_pipelines("my_pipeline", requests_list)
```

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

### Advanced File Handling

```python
import requests
import os
from pathlib import Path

class FileUploader:
    def __init__(self, base_url="http://localhost:1416"):
        self.base_url = base_url

    def upload_and_run(self, pipeline_name, file_paths, params=None):
        """Upload files and run pipeline"""
        url = f"{self.base_url}/{pipeline_name}/run"

        files = []
        try:
            # Prepare files
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

                files.append(('files', open(file_path, 'rb')))

            # Make request
            data = params or {}
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()

        finally:
            # Close file handles
            for _, file_obj in files:
                file_obj[1].close()

    def upload_directory(self, pipeline_name, directory_path, params=None):
        """Upload all files from a directory"""
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Get all files from directory
        file_paths = []
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                file_paths.append(str(file_path))

        if not file_paths:
            raise ValueError(f"No files found in directory: {directory_path}")

        return self.upload_and_run(pipeline_name, file_paths, params)

# Usage
uploader = FileUploader()

# Upload single file
result = uploader.upload_and_run(
    "rag_pipeline",
    ["document.pdf"],
    {"query": "Summarize this document"}
)
print(result)

# Upload directory
result = uploader.upload_directory(
    "rag_pipeline",
    "./documents",
    {"query": "Analyze all documents"}
)
print(result)
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

### Client-side monitoring (optional)

If you need to time requests from a client, prefer Python's `logging` over `print`:

```python
import logging
import time
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hayhooks.client")

def call_pipeline(base_url: str, pipeline_name: str, params: dict) -> dict:
    start = time.time()
    try:
        resp = requests.post(f"{base_url}/{pipeline_name}/run", json=params)
        resp.raise_for_status()
        logger.info("%s completed in %.2fs (status=%s)", pipeline_name, time.time() - start, resp.status_code)
        return resp.json()
    except Exception as exc:
        logger.exception("%s failed after %.2fs", pipeline_name, time.time() - start)
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
