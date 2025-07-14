# Hayhooks Examples

This directory contains various examples demonstrating different use cases and features of Hayhooks. Each example showcases specific capabilities and patterns for building AI pipelines with Hayhooks.

## Examples Overview

| Example | Description | Key Features | Use Case |
|---------|-------------|--------------|----------|
| [async_question_answer](./pipeline_wrappers/async_question_answer/) | Async question-answering pipeline with streaming support | • Async pipeline execution<br/>• Streaming responses<br/>• OpenAI Chat Generator<br/>• Both API and chat completion interfaces | Building conversational AI systems that need async processing and real-time streaming responses |
| [chat_with_website](./pipeline_wrappers/chat_with_website/) | Answer questions about website content | • Web content fetching<br/>• HTML to document conversion<br/>• Content-based Q&A<br/>• Configurable URLs | Creating AI assistants that can answer questions about specific websites or web-based documentation |
| [chat_with_website_mcp](./pipeline_wrappers/chat_with_website_mcp/) | MCP-compatible website chat pipeline | • MCP (Model Context Protocol) support<br/>• Website content analysis<br/>• API-only interface<br/>• Simplified deployment | Integrating website analysis capabilities into MCP-compatible AI systems and tools |
| [chat_with_website_streaming](./pipeline_wrappers/chat_with_website_streaming/) | Streaming website chat responses | • Real-time streaming<br/>• Website content processing<br/>• Progressive response generation<br/>• Enhanced user experience | Building responsive web applications that provide real-time AI responses about website content |
| [open_webui_agent_events](./pipeline_wrappers/open_webui_agent_events/) | Agent pipeline with OpenWebUI status events | • OpenWebUI event integration<br/>• Status event generation<br/>• Details tag creation<br/>• Agent-based responses<br/>• Real-time UI feedback | Creating interactive AI agents with rich status updates and progress indicators in OpenWebUI |
| [open_webui_agent_on_tool_calls](./pipeline_wrappers/open_webui_agent_on_tool_calls/) | Agent with tool call interception and OpenWebUI events | • Tool call lifecycle hooks<br/>• OpenWebUI notifications<br/>• Weather API tool integration<br/>• Real-time tool execution feedback<br/>• Status and result tracking | Building agents that provide detailed feedback about tool execution with rich UI interactions |
| [rag_indexing_query](./rag_indexing_query/) | Complete RAG system with Elasticsearch | • Document indexing pipeline<br/>• Query pipeline<br/>• Elasticsearch integration<br/>• Multiple file format support (PDF, Markdown, Text)<br/>• Sentence transformers embeddings | Implementing production-ready RAG systems for document search and knowledge retrieval |
| [shared_code_between_wrappers](./shared_code_between_wrappers/) | Code sharing between pipeline wrappers | • Shared library imports<br/>• HAYHOOKS_ADDITIONAL_PYTHON_PATH<br/>• Multiple deployment strategies<br/>• Code reusability | Organizing complex projects with multiple pipelines that share common functionality |

## Getting Started

Each example includes:

- **Pipeline wrapper implementation** (`pipeline_wrapper.py`)
- **Pipeline configuration** (`.yml` files where applicable)
- **Dependencies** (`requirements.txt` where applicable)
- **Documentation** (individual README files with setup instructions)

## Common Prerequisites

Most examples require:

- Python 3.9+
- A virtual environment (recommended)
- The Hayhooks package: `pip install hayhooks`
- Additional dependencies (as specified in each example)

## Quick Start

1. Navigate to the `/examples` directory
2. Create and activate a virtual environment (recommended)
3. Install dependencies: `pip install -r requirements.txt` (if present)
4. Follow the specific example's README for deployment and testing

## Support

For questions about specific examples, refer to their individual README files or the main Hayhooks documentation.
