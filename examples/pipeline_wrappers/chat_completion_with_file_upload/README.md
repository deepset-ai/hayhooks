# Chat Completions API with File Upload

Demonstrates how to use the OpenAI [Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) (`/v1/chat/completions`) with [file inputs](https://platform.openai.com/docs/api-reference/chat/create) uploaded via the [Files API](https://platform.openai.com/docs/api-reference/files/create) (`/v1/files`).

A Haystack **Agent** reads uploaded files and local files server-side, streaming answers back to the client.

## What this example shows

- **`run_chat_completion_async`** — resolves file references in chat messages and streams the agent's answer
- **`run_file_upload`** — stores uploaded file bytes in an in-memory dict
- **`_resolve_file_references`** — converts `{"type": "file", "file": {"file_id": "..."}}` content parts into inline text, matching the [OpenAI file input format](https://platform.openai.com/docs/api-reference/chat/create)
- **`_strip_tool_calls`** — filters internal Agent tool calls from the stream to prevent agentic clients from looping
- **`read_file` tool** — the agent can also read local files by absolute path

## Requirements

```bash
export OPENAI_API_KEY="sk-..."
```

## Deploy

```bash
hayhooks run --pipelines-dir examples/pipeline_wrappers/chat_completion_with_file_upload
```

## Usage

### Upload a file and ask about it

Upload a file via `/v1/files`, then reference it with `{"type": "file", "file": {"file_id": "..."}}` in a multi-part chat message — the same format OpenAI uses.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1416/v1", api_key="unused")

# 1. Upload a file
uploaded = client.files.create(
    file=open("document.txt", "rb"),
    purpose="user_data",
)
print(f"Uploaded: {uploaded.id}")

# 2. Reference the file in a chat completion (multi-part content)
response = client.chat.completions.create(
    model="chat_completion_with_file_upload",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this file"},
                {"type": "file", "file": {"file_id": uploaded.id}},
            ],
        }
    ],
)
print(response.choices[0].message.content)
```

### Streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1416/v1", api_key="unused")

uploaded = client.files.create(file=open("document.txt", "rb"), purpose="user_data")

stream = client.chat.completions.create(
    model="chat_completion_with_file_upload",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the key points in this file?"},
                {"type": "file", "file": {"file_id": uploaded.id}},
            ],
        }
    ],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

### Ask about a local file (no upload needed)

The agent also has a `read_file` tool for reading files by path:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1416/v1", api_key="unused")

response = client.chat.completions.create(
    model="chat_completion_with_file_upload",
    messages=[{"role": "user", "content": "What is in /absolute/path/to/README.md?"}],
)
print(response.choices[0].message.content)
```

### curl

```bash
# 1. Upload
curl http://localhost:1416/v1/files \
  -F "file=@document.txt" \
  -F "purpose=user_data"

# 2. Chat (use the file_id from the upload response)
curl http://localhost:1416/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat_completion_with_file_upload",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Summarize this file"},
          {"type": "file", "file": {"file_id": "file-abc123"}}
        ]
      }
    ]
  }'
```

## Notes

- The `read_file` tool has no sandboxing — this is a demo. Production use should restrict allowed paths.
- The in-memory upload store is for demonstration only — files are lost on restart.
- The pipeline name in the `model` field must match the directory name (`chat_completion_with_file_upload`).
