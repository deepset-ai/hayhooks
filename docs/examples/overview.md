# Examples Overview

This directory contains various examples demonstrating different use cases and features of Hayhooks. Each example showcases specific capabilities and patterns for building AI pipelines with Hayhooks.

## Available Examples

| Example | Description | Key Features | Use Case |
|---------|-------------|--------------|----------|
| [Chat with Website](chat-with-website.md) | Answer questions about website content | • Web content fetching<br>• HTML to document conversion<br>• Content-based Q&A<br>• Configurable URLs | Creating AI assistants that can answer questions about specific websites or web-based documentation |
| [RAG System](rag-system.md) | Complete RAG system | • Document indexing pipeline<br>• Query pipeline<br>• Multiple file format support (PDF, Markdown, Text)<br>• Sentence transformers embeddings | Implementing production-ready RAG systems for document search and knowledge retrieval |
| [Async Operations](async-operations.md) | Async pipeline with streaming support | • Async pipeline execution<br>• Streaming responses<br>• OpenAI Chat Generator<br>• Both API and chat completion interfaces | Building conversational AI systems that need async processing and real-time streaming responses |
| [OpenWebUI Events](openwebui-events.md) | Agent pipeline with OpenWebUI status events | • OpenWebUI event integration<br>• Status event generation<br>• Details tag creation<br>• Agent-based responses<br>• Real-time UI feedback | Creating interactive AI agents with rich status updates and progress indicators in OpenWebUI |

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

## Example Structure

```
examples/
├── pipeline_wrappers/
│   ├── async_question_answer/
│   ├── chat_with_website/
│   ├── chat_with_website_mcp/
│   ├── chat_with_website_streaming/
│   ├── open_webui_agent_events/
│   └── open_webui_agent_on_tool_calls/
├── rag_indexing_query/
│   ├── indexing_pipeline/
│   ├── query_pipeline/
│   ├── files_to_index/
│   ├── docker-compose.yml
│   └── README.md
└── shared_code_between_wrappers/
    ├── common/
    ├── input_pipelines/
    └── README.md
```

## Next Steps

- Explore individual examples for detailed implementation guides
- Check the [Hayhooks documentation](../index.md) for general usage
- Visit the [GitHub repository](https://github.com/deepset-ai/hayhooks) for the latest examples
