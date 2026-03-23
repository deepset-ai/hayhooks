# Responses API with File-Reading Agent

Demonstrates how to use the OpenAI Responses API (`/v1/responses`) with a Haystack **Agent** that can read files from disk. The agent uses server-side tool calling, so it works with any Responses API client — including Codex CLI.

## What this example shows

- **Haystack Agent with tools** — the agent has a `read_file` tool that reads any local file by path, and a `read_uploaded_file` tool for files uploaded via `/v1/files`
- **`run_response_async`** — converts Responses API input items to `ChatMessage` objects and streams the agent's answer via `async_streaming_generator`
- **`run_file_upload`** — stores uploaded file bytes in an in-memory dict for retrieval by `file_id`
- **`_strip_tool_calls`** — filters internal Agent tool calls from the stream to prevent agentic clients (e.g. Codex CLI) from looping
- **Works with Codex CLI** — the agent reads files server-side, so no client-side file upload or `input_file` references are needed

## Requirements

```bash
export OPENAI_API_KEY="sk-..."
```

## Deploy

```bash
hayhooks run --pipelines-dir examples/pipeline_wrappers/responses_with_file_upload
```

## Usage

### Ask about a local file

```bash
curl http://localhost:1416/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "responses_with_file_upload",
    "input": "What is in the LICENSE file?"
  }'
```

The agent calls its `read_file` tool to read `LICENSE` from disk and answers based on the content.

### Upload a file, then ask about it

```bash
# 1. Upload
curl http://localhost:1416/v1/files \
  -F "file=@document.txt" \
  -F "purpose=user_data"

# 2. Ask (use the file_id from the upload response)
curl http://localhost:1416/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "responses_with_file_upload",
    "input": "Summarize the uploaded file with id file-abc123"
  }'
```

The agent calls its `read_uploaded_file` tool to retrieve the content from the in-memory store.

### OpenAI Python client

#### Responses API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1416/v1", api_key="unused")

# Non-streaming
response = client.responses.create(
    model="responses_with_file_upload",
    input="What is in /absolute/path/to/README.md?",
)
print(response.output_text)

# Streaming
stream = client.responses.create(
    model="responses_with_file_upload",
    input="Summarize /absolute/path/to/README.md",
    stream=True,
)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print()
```

> **Note:** When using the OpenAI client (as opposed to Codex CLI), the agent
> doesn't have access to the client's working directory — use **absolute paths**
> when referencing files.

#### Files API + Responses API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1416/v1", api_key="unused")

# 1. Upload a file
uploaded = client.files.create(
    file=open("document.txt", "rb"),
    purpose="user_data",
)
print(f"Uploaded: {uploaded.id}")

# 2. Ask the agent about it (reference the file_id in the prompt)
response = client.responses.create(
    model="responses_with_file_upload",
    input=f"Summarize the uploaded file with id {uploaded.id}",
)
print(response.output_text)
```

## Notes

- The `read_file` tool has no sandboxing — this is a demo. Production use should restrict allowed paths.
- The in-memory upload store is for demonstration only — files are lost on restart.
- The pipeline name in the `model` field must match the directory name (`responses_with_file_upload`).
