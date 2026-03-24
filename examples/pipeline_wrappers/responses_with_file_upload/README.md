# Responses API with File-Reading Agent

Server-side Responses API demo for OpenAI client and curl usage.

The agent has two tools:
- **`read_file`** — reads any local file by absolute path
- **`read_uploaded_file`** — retrieves files uploaded via `/v1/files` from an in-memory store

Internal tool calls are stripped from the stream (`_strip_tool_calls`) so agentic clients don't try to execute them locally.

## Requirements

```bash
export OPENAI_API_KEY="sk-..."
```

## Deploy

```bash
hayhooks run
hayhooks pipeline deploy-files -n responses_with_file_upload examples/pipeline_wrappers/responses_with_file_upload
```

## Usage

### Ask about a local file

```bash
curl http://localhost:1416/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "responses_with_file_upload",
    "input": "What is in /absolute/path/to/LICENSE?"
  }'
```

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

### OpenAI Python client

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

### Files API + Responses API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1416/v1", api_key="unused")

# 1. Upload
uploaded = client.files.create(
    file=open("document.txt", "rb"),
    purpose="user_data",
)
print(f"Uploaded: {uploaded.id}")

# 2. Ask
response = client.responses.create(
    model="responses_with_file_upload",
    input=f"Summarize the uploaded file with id {uploaded.id}",
)
print(response.output_text)
```

## Notes

- The `read_file` tool has no sandboxing — this is a demo. Production use should restrict allowed paths.
- The in-memory upload store is for demonstration only — files are lost on restart.
