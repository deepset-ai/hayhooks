# Quick Start

This guide will help you get started with Hayhooks quickly.

## Prerequisites

- Python 3.9+
- A Haystack pipeline or agent to deploy

## Installation

```bash
pip install hayhooks
```

If you want to use the [MCP Server](../features/mcp-support.md), you need to install the `hayhooks[mcp]` package:

```bash
pip install hayhooks[mcp]
```

!!! warning "Python 3.10+ Required for MCP"
    You'll need to run at least Python 3.10+ to use the MCP Server.

## Basic Usage

### 1. Start Hayhooks

```bash
hayhooks run
```

This will start the Hayhooks server on `http://localhost:1416` by default.

### 2. Deploy a Pipeline

Deploy a pipeline using the `deploy-files` command:

```bash
hayhooks pipeline deploy-files -n chat_with_website examples/pipeline_wrappers/chat_with_website_streaming
```

### 3. Check Status

Verify your pipeline is deployed:

```bash
hayhooks status
```

### 4. Run Your Pipeline

Run your pipeline via the API:

```bash
curl -X POST \
  http://localhost:1416/chat_with_website/run \
  -H 'Content-Type: application/json' \
  -d '{"urls": ["https://haystack.deepset.ai"], "question": "What is Haystack?"}'
```

## Quick Start with Docker Compose

For the fastest setup, see [Quick Start with Docker Compose](../getting-started/quick-start-docker.md) for full instructions.

This provides a ready-to-use setup with Open WebUI integration.

## Next Steps

- [Configuration](../getting-started/configuration.md) - Learn about configuration options
- [Examples](../examples/overview.md) - Explore example implementations
- [Features](../features/openai-compatibility.md) - Discover advanced features
