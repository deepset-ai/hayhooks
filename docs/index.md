---
hide:
  - navigation
  - toc
---

<div class="landing-hero" markdown>
<div class="landing-copy" markdown>

<span class="landing-eyebrow">Haystack · production APIs · terminal observability</span>

# Serve agents.<br>See every run.

Turn Haystack Pipelines and Agents into REST APIs, streaming chat backends, MCP tools, and A2A agents. Keep long-running work durable, and inspect it from your terminal.

<div class="landing-actions" markdown>

[Get started](getting-started/quick-start.md){ .md-button .md-button--primary }
[Explore the dashboard](features/terminal-dashboard.md){ .md-button }

</div>

`pip install "hayhooks[tui]"`

</div>
<div class="landing-preview" markdown>
<div class="dashboard-frame" markdown>

[![Hayhooks terminal dashboard showing ordinary and durable executions alongside run details](assets/terminal-dashboard/browse-160x48.png)](features/terminal-dashboard.md)

</div>
<p class="dashboard-caption">The actual terminal UI, shown with demo runs.</p>
</div>
</div>

<div class="landing-section" markdown>

<span class="landing-eyebrow">The terminal dashboard</span>

## One place to understand a run.

Start with the health of a deployment. Follow its execution. Open the trace and find the component that took the time.

<div class="dashboard-path" markdown>

- **01 / Deployments**<br>Pipeline and agent metadata, endpoint capabilities, and durable process health. Press `d`.
- **02 / Executions**<br>Recent ordinary runs beside paginated durable history. Inspect state, attempts, progress, and waiting reasons. Press `e`.
- **03 / Traces**<br>Filter activity, compare timing, and open an expandable component tree with a terminal waterfall. Press `v`.

</div>

=== "Start the server"

    ```bash
    pip install "hayhooks[tui]"
    hayhooks run
    ```

=== "Open the dashboard"

    In a second terminal, under the same OS account:

    ```bash
    hayhooks dashboard
    ```

    Run a deployed pipeline, then press `e` for executions or `v` for traces.

The dashboard is read-only. Local capture is on by default, and same-user POSIX clients discover the local operator credential automatically. Ordinary history is bounded and process-local; durable history follows the execution store's retention.

[Dashboard guide](features/terminal-dashboard.md) · [Remote connections](features/terminal-dashboard.md#connect-to-a-remote-server) · [Tracing and OTLP](reference/tracing.md)

</div>

<div class="landing-section" markdown>

<span class="landing-eyebrow">Beyond the request</span>

## Let the work survive the restart.

An agent researching for ten minutes. A pipeline waiting for approval. A deployment while a job is still running. Durable execution checkpoints work in Redis and resumes it from the last checkpoint on any replica.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> queued: submit
    queued --> running: claim
    running --> waiting: human approval
    waiting --> queued: resume
    running --> queued: retry or recovery
    running --> completed
```

[Durable execution](features/durable-execution.md){ .md-button }
[Operations guide](deployment/durable-operations.md){ .md-button }

</div>

## Serve it where your users work.

| Interface | What it gives you |
| --- | --- |
| [REST APIs](concepts/pipeline-deployment.md) | Typed inputs, validation, OpenAPI docs, file uploads, and custom routes |
| [OpenAI-compatible endpoints](features/openai-compatibility.md) | Streaming chat completions and Responses API support |
| [MCP tools](features/mcp-support.md) | Pipeline tools for Cursor, Claude Desktop, and other MCP clients |
| [A2A agents](features/a2a-support.md) | Discoverable agent cards and agent-to-agent delegation |
| [Open WebUI](features/openwebui-integration.md) / [Chainlit](features/chainlit-integration.md) | Chat interfaces for deployed pipelines and agents |

[Installation options](getting-started/installation.md) · [Configuration](getting-started/configuration.md) · [Examples](examples/overview.md) · [GitHub](https://github.com/deepset-ai/hayhooks)
