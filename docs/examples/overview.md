# Examples Overview

This page lists all maintained Hayhooks examples with detailed descriptions and links to the source code.

## Pipeline wrapper examples

| Example | Docs | Code | Description |
|---|---|---|---|
| Chat with Website (Streaming) | [chat-with-website.md](chat-with-website.md) | [GitHub](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/chat_with_website_streaming) | Website Q&A with streaming |
| Chat with Website (basic) | — | [GitHub](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/chat_with_website) | Minimal website Q&A wrapper |
| Chat with Website (MCP) | — | [GitHub](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/chat_with_website_mcp) | Exposes website Q&A as MCP Tool |
| Async Question Answer (Streaming) | [async-operations.md](async-operations.md) | [GitHub](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/async_question_answer) | Async pipeline and streaming patterns |
| Open WebUI Agent Events | [openwebui-events.md](openwebui-events.md) | [GitHub](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/open_webui_agent_events) | UI events and status updates |
| Open WebUI Agent on Tool Calls | [openwebui-events.md](openwebui-events.md) | [GitHub](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/open_webui_agent_on_tool_calls) | Tool call interception & feedback |
| Shared Code Between Wrappers | — | [GitHub](https://github.com/deepset-ai/hayhooks/tree/main/examples/shared_code_between_wrappers) | Reusing code across wrappers |

## End-to-end examples & patterns

| Example | Docs | Code | Description |
|---|---|---|---|
| RAG: Indexing and Query with Elasticsearch | [rag-system.md](rag-system.md) | [GitHub](https://github.com/deepset-ai/hayhooks/tree/main/examples/rag_indexing_query) | Full indexing/query pipelines with Elasticsearch |

## How to use examples

**Prerequisites:**
- Install Hayhooks: `pip install hayhooks` (additional deps per example may apply)

**Deployment:**
- Pipeline wrappers: deploy directly with `hayhooks pipeline deploy-files -n <name> <example_dir>` and run via API (`POST /<name>/run`) or OpenAI-compatible chat endpoints if implemented
- End-to-end examples: follow the example's documentation for full setup (services like Elasticsearch, multi-pipeline deployment, datasets, etc.)

For general usage and CLI commands, see the [Getting Started Guide](../getting-started/quick-start.md).

## Try all examples with Docker Compose

The repository includes a `compose.yml` file that automatically deploys all pipeline wrapper examples:

```bash
# Clone the repository
git clone https://github.com/deepset-ai/hayhooks.git
cd hayhooks

# Set your API key (required for most examples)
export OPENAI_API_KEY=your-api-key

# Start Hayhooks with all examples
docker compose up -d
```

This will:

- Start Hayhooks on port `1416` (API) and `1417` (MCP server)
- Auto-deploy all pipeline wrappers from `examples/pipeline_wrappers/`
- Install required dependencies like `trafilatura`

Check deployed pipelines:

```bash
curl localhost:1416/status
```

Access the API documentation at <http://localhost:1416/docs>.
