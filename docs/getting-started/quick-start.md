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

**NOTE: You'll need to run at least Python 3.10+ to use the MCP Server.**

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

For the fastest setup, use our Docker Compose configuration with pre-configured OpenWebUI integration:

```bash
git clone https://github.com/deepset-ai/hayhooks-open-webui-docker-compose.git
cd hayhooks-open-webui-docker-compose
docker-compose up -d
```

This will start both Hayhooks and OpenWebUI with all necessary configurations.

## Next Steps

- [Configuration](../getting-started/configuration.md) - Learn about configuration options
- [Examples](../examples/overview.md) - Explore example implementations
- [Features](../features/openai-compatibility.md) - Discover advanced features
