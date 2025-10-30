# MCP Support

Hayhooks supports the [Model Context Protocol](https://modelcontextprotocol.io/) and can act as an [MCP Server](https://modelcontextprotocol.io/docs/concepts/architecture), exposing pipelines and agents as MCP tools for use in AI development environments.

## Overview

The Hayhooks MCP Server:

- Exposes [Core Tools](#core-mcp-tools) for controlling Hayhooks directly from IDEs
- Exposes deployed Haystack pipelines as usable [MCP Tools](#pipeline-tools)
- Supports both [Server-Sent Events (SSE)](#sse-transport) and [Streamable HTTP](#streamable-http-transport) transports
- Integrates with AI development environments like [Cursor](https://cursor.com) and [Claude Desktop](https://claude.ai/download)

## Requirements

- Python 3.10+ for MCP support
- Install with `pip install hayhooks[mcp]`

## Getting Started

### Install with MCP Support

```bash
pip install hayhooks[mcp]
```

### Start MCP Server

```bash
hayhooks mcp run
```

This starts the MCP server on `HAYHOOKS_MCP_HOST:HAYHOOKS_MCP_PORT` (default: `localhost:1417`).

### Configuration

Environment variables for MCP server:

```bash
HAYHOOKS_MCP_HOST=localhost    # MCP server host
HAYHOOKS_MCP_PORT=1417         # MCP server port
```

## Transports

### Streamable HTTP (Recommended) {#streamable-http-transport}

The preferred transport for modern MCP clients:

```python
import mcp

client = mcp.Client("http://localhost:1417/mcp")
```

### Server-Sent Events (SSE) {#sse-transport}

Legacy transport maintained for backward compatibility:

```python
import mcp

client = mcp.Client("http://localhost:1417/sse")
```

## Core MCP Tools

Hayhooks provides core tools for managing pipelines:

### get_all_pipeline_statuses

Get status of all deployed pipelines:

```python
result = await client.call_tool("get_all_pipeline_statuses")
```

### get_pipeline_status

Get status of a specific pipeline:

```python
result = await client.call_tool("get_pipeline_status", {"pipeline_name": "my_pipeline"})
```

### deploy_pipeline

Deploy a pipeline from files:

```python
result = await client.call_tool("deploy_pipeline", {
    "name": "my_pipeline",
    "files": [
        {"name": "pipeline_wrapper.py", "content": "..."},
        {"name": "pipeline.yml", "content": "..."}
    ],
    "save_files": True,
    "overwrite": False
})
```

### undeploy_pipeline

Undeploy a pipeline:

```python
result = await client.call_tool("undeploy_pipeline", {"pipeline_name": "my_pipeline"})
```

## Pipeline Tools

### PipelineWrapper as MCP Tool

When you deploy a pipeline with `PipelineWrapper`, it's automatically exposed as an MCP tool.

**MCP Tool Requirements:**

A [MCP Tool](https://modelcontextprotocol.io/docs/concepts/tools) requires:

- `name`: The name of the tool
- `description`: The description of the tool
- `inputSchema`: JSON Schema describing the tool's input parameters

**How Hayhooks Creates MCP Tools:**

For each deployed pipeline, Hayhooks will:

- Use the pipeline wrapper `name` as MCP Tool `name` (always present)
- Parse **`run_api` method docstring**:
  - If you use Google-style or reStructuredText-style docstrings, use the first line as MCP Tool `description` and the rest as `parameters` (if present)
  - Each parameter description will be used as the `description` of the corresponding Pydantic model field (if present)
- Generate a Pydantic model from the `inputSchema` using the **`run_api` method arguments as fields**

**Example:**

```python
from pathlib import Path
from haystack import Pipeline
from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: list[str], question: str) -> str:
        #
        # NOTE: The following docstring will be used as MCP Tool description
        #
        """
        Ask a question about one or more websites using a Haystack pipeline.
        """
        result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
        return result["llm"]["replies"][0]
```

### YAML Pipeline as MCP Tool

YAML-deployed pipelines are also automatically exposed as MCP tools. When you deploy via `hayhooks pipeline deploy-yaml`, the pipeline becomes available as an MCP tool with its input schema derived from the YAML `inputs` section.

For complete examples and detailed information, see [YAML Pipeline Deployment](../concepts/yaml-pipeline-deployment.md).

### Skip MCP Tool Listing

To prevent a pipeline from being listed as an MCP tool:

```python
class PipelineWrapper(BasePipelineWrapper):
    skip_mcp = True  # This pipeline won't be listed as an MCP tool

    def setup(self) -> None:
        ...

    def run_api(self, ...) -> str:
        ...
```

## IDE Integration

### Cursor Integration

Add Hayhooks MCP Server in Cursor Settings → MCP:

```json
{
  "mcpServers": {
    "hayhooks": {
      "url": "http://localhost:1417/mcp"
    }
  }
}
```

Once configured, you can deploy, manage, and run pipelines directly from Cursor chat using the Core MCP Tools.

For more information about MCP in Cursor, see the [Cursor MCP Documentation](https://cursor.com/docs/context/mcp).

### Claude Desktop Integration

Configure Claude Desktop to connect to Hayhooks MCP Server:

!!! info "Claude Desktop Tiers"
    === "Free Tier"
        Use [supergateway](https://github.com/supercorp-ai/supergateway) to bridge the connection

        ```json
        {
          "mcpServers": {
            "hayhooks": {
              "command": "npx",
              "args": ["-y", "supergateway", "--streamableHttp", "http://localhost:1417/mcp"]
            }
          }
        }
        ```

    === "Pro/Max/Teams/Enterprise"
        Direct connection via Streamable HTTP or SSE

        ```json
        {
          "mcpServers": {
            "hayhooks": {
              "url": "http://localhost:1417/mcp"
            }
          }
        }
        ```

## Development Workflow

**Basic workflow:**

1. Start Hayhooks server: `hayhooks run`
2. Start MCP server: `hayhooks mcp run` (in another terminal)
3. Configure your IDE to connect to the MCP server
4. Deploy and manage pipelines through your IDE using natural language

## Tool Development

### Custom Tool Descriptions

Use docstrings to provide better tool descriptions:

```python
def run_api(self, urls: list[str], question: str) -> str:
    """
    Ask questions about website content using AI.

    This tool analyzes website content and provides answers to user questions.
    It's perfect for research, content analysis, and information extraction.

    Args:
        urls: List of website URLs to analyze
        question: Question to ask about the content

    Returns:
        Answer to the question based on the website content
    """
    result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
    return result["llm"]["replies"][0]
```

### Input Validation

Hayhooks automatically validates inputs based on your method signature:

```python
def run_api(
    self,
    urls: list[str],           # Required: List of URLs
    question: str,             # Required: User question
    max_tokens: int = 1000     # Optional: Max tokens
) -> str:
    ...
```

## Security Considerations

### Authentication

Currently, Hayhooks MCP server doesn't include built-in authentication. Consider:

- Running behind a reverse proxy with authentication
- Using network-level security (firewalls, VPNs)
- Implementing custom middleware for authentication

### Resource Management

- Monitor tool execution for resource usage
- Implement timeouts for long-running operations
- Consider rate limiting for production deployments

## Troubleshooting

### Common Issues

#### Connection Refused

If you cannot connect to the MCP server, ensure the MCP server is running with `hayhooks mcp run`. Check that the port configuration matches (default is `1417`), and verify network connectivity between the client and server.

#### Tool Not Found

If an MCP tool is not showing up, verify that the pipeline is properly deployed using `hayhooks status`. Check if the `skip_mcp` class attribute is set to `True` in your `PipelineWrapper`, which would prevent it from being listed. Ensure the `run_api` method is properly implemented with correct type hints.

#### Input Validation Errors

If you're getting validation errors when calling tools, check that your method signatures match the expected input types. Verify that all required parameters are being passed and that data types match the type hints in your `run_api` method signature. Review the MCP tool's `inputSchema` to ensure parameter names and types are correct.

### Debug Commands

The MCP server exposes the following endpoints:

- **Streamable HTTP endpoint**: `http://localhost:1417/mcp` - Main MCP protocol endpoint
- **SSE endpoint**: `http://localhost:1417/sse` - Server-Sent Events transport (deprecated)
- **Status/Health Check**: `http://localhost:1417/status` - Returns `{"status": "ok"}` for health monitoring

#### Testing the health endpoint

```bash
# Check if MCP server is running
curl http://localhost:1417/status

# Expected response:
# {"status":"ok"}
```

This status endpoint is useful for:

- Container health checks in Docker/Kubernetes deployments
- Load balancer health probes
- Monitoring and alerting systems
- Verifying the MCP server is running before connecting clients

Use an MCP-capable client like [supergateway](https://github.com/supercorp-ai/supergateway), [Cursor](https://cursor.com), or [Claude Desktop](https://claude.ai/download) to list and call tools. Example supergateway usage is shown above.

## Next Steps

- [PipelineWrapper Guide](../concepts/pipeline-wrapper.md) - Learn how to create MCP-compatible pipeline wrappers
- [Examples](../examples/overview.md) - See working examples of deployed pipelines
