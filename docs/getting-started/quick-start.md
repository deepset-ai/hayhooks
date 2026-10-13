# Quick Start

This guide will help you get started with Hayhooks quickly.

## Prerequisites

- Python 3.10+
- A Haystack pipeline or agent to deploy

## Installation

See [Installation](installation.md) for detailed setup instructions.

Quick install:

```bash
pip install hayhooks
```

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

=== "cURL"

    ```bash
    curl -X POST \
      http://localhost:1416/chat_with_website/run \
      -H 'Content-Type: application/json' \
      -d '{"urls": ["https://haystack.deepset.ai"], "question": "What is Haystack?"}'
    ```

=== "Python"

    ```python
    import requests

    response = requests.post(
        "http://localhost:1416/chat_with_website/run",
        json={
            "urls": ["https://haystack.deepset.ai"],
            "question": "What is Haystack?"
        }
    )
    print(response.json())
    ```

=== "Hayhooks CLI"

    ```bash
    hayhooks pipeline run chat_with_website \
      --param 'urls=["https://haystack.deepset.ai"]' \
      --param 'question="What is Haystack?"'
    ```

### 5. Inspect the run

Install the optional terminal UI and open it in a second terminal under the same OS account:

```bash
pip install "hayhooks[tui]"
hayhooks dashboard
```

Press `e` for executions or `v` for traces, then Enter for details. Local capture is enabled by default, so the run you just made appears while it remains in the server's recent buffer. See the [dashboard guide](../features/terminal-dashboard.md) for screenshots, navigation, and remote connections.

## Quick Start with Docker Compose

For the fastest setup with Open WebUI integration, see [Quick Start with Docker Compose](quick-start-docker.md).

## Next Steps

- [Pipeline Deployment](../concepts/pipeline-deployment.md) - Learn deployment methods
- [Terminal Dashboard](../features/terminal-dashboard.md) - Follow executions and inspect component timing
- [Examples](../examples/overview.md) - Explore example implementations
