# Hayhooks

**Hayhooks** makes it easy to deploy and serve [Haystack](https://haystack.deepset.ai/) [Pipelines](https://docs.haystack.deepset.ai/docs/pipelines) and [Agents](https://docs.haystack.deepset.ai/docs/agents).

With Hayhooks, you can:

- 📦 **Deploy your Haystack pipelines and agents as REST APIs** with maximum flexibility and minimal boilerplate code.
- 🛠️ **Expose your Haystack pipelines and agents over the MCP protocol**, making them available as tools in AI dev environments like [Cursor](https://cursor.com) or [Claude Desktop](https://claude.ai/download). Under the hood, Hayhooks runs as an [MCP Server](https://modelcontextprotocol.io/docs/concepts/architecture), exposing each pipeline and agent as an [MCP Tool](https://modelcontextprotocol.io/docs/concepts/tools).
- 💬 **Integrate your Haystack pipelines and agents with [open-webui](https://openwebui.com)** as OpenAI-compatible chat completion backends with streaming support.
- 🕹️ **Control Hayhooks core API endpoints through chat** - deploy, undeploy, list, or run Haystack pipelines and agents by chatting with [Claude Desktop](https://claude.ai/download), [Cursor](https://cursor.com), or any other MCP client.

[![PyPI - Version](https://img.shields.io/pypi/v/hayhooks.svg)](https://pypi.org/project/hayhooks)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hayhooks.svg)](https://pypi.org/project/hayhooks)
[![Docker image release](https://github.com/deepset-ai/hayhooks/actions/workflows/docker.yml/badge.svg)](https://github.com/deepset-ai/hayhooks/actions/workflows/docker.yml)
[![Tests](https://github.com/deepset-ai/hayhooks/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/hayhooks/actions/workflows/tests.yml)

## Quick Start

```bash
# Install Hayhooks
pip install hayhooks

# Start the server
hayhooks run

# Deploy a pipeline
hayhooks pipeline deploy-files -n chat_with_website examples/pipeline_wrappers/chat_with_website_streaming
```

## Key Features

### 🚀 Easy Deployment

- Deploy Haystack pipelines and agents as REST APIs with minimal setup
- Support for both YAML-based and wrapper-based pipeline deployment
- Automatic OpenAI-compatible endpoint generation

### 🌐 Multiple Integration Options

- **MCP Protocol**: Expose pipelines as MCP tools for use in AI development environments
- **OpenWebUI Integration**: Use Hayhooks as a backend for OpenWebUI with streaming support
- **OpenAI Compatibility**: Seamless integration with OpenAI-compatible tools and frameworks

### 🔧 Developer Friendly

- CLI for easy pipeline management
- Flexible configuration options
- Comprehensive logging and debugging support
- Custom route and middleware support

### 📁 File Upload Support

- Built-in support for handling file uploads in pipelines
- Perfect for RAG systems and document processing

## Next Steps

- [Quick Start Guide](getting-started/quick-start.md) - Get started with Hayhooks
- [Installation](getting-started/installation.md) - Install Hayhooks and dependencies
- [Configuration](getting-started/configuration.md) - Configure Hayhooks for your needs
- [Examples](examples/overview.md) - Explore example implementations

## Community & Support

- **GitHub**: [deepset-ai/hayhooks](https://github.com/deepset-ai/hayhooks)
- **Issues**: [GitHub Issues](https://github.com/deepset-ai/hayhooks/issues)
- **Documentation**: [Full Documentation](https://deepset-ai.github.io/hayhooks/)

Hayhooks is actively maintained by the [deepset](https://deepset.ai/) team.
