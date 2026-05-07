# Hayhooks Examples

This directory contains various examples demonstrating different use cases and features of Hayhooks. Each example showcases specific capabilities and patterns for building AI pipelines with Hayhooks.

## Examples Overview

| Example | Description | Key Features | Use Case |
|---------|-------------|--------------|----------|
| [multi_llm_streaming](./pipeline_wrappers/multi_llm_streaming/) | Multiple LLM components with automatic streaming | • Two sequential LLMs<br/>• Automatic multi-component streaming<br/>• No special configuration needed<br/>• Shows default streaming behavior | Demonstrating how hayhooks automatically streams from all components in a pipeline |
| [async_question_answer](./pipeline_wrappers/async_question_answer/) | Async question-answering pipeline with streaming support | • Async pipeline execution<br/>• Streaming responses<br/>• OpenAI Chat Generator<br/>• Both API and chat completion interfaces | Building conversational AI systems that need async processing and real-time streaming responses |
| [async_hybrid_streaming](./pipeline_wrappers/async_hybrid_streaming/) | AsyncPipeline with legacy sync-only components using hybrid mode | • AsyncPipeline with OpenAIGenerator<br/>• `allow_sync_streaming_callbacks=True`<br/>• Automatic sync-to-async bridging<br/>• Migration example | Using legacy components (OpenAIGenerator) in async pipelines, migrating from sync to async gradually, handling third-party sync-only components |
| [reasoning_agent](./pipeline_wrappers/reasoning_agent/) | Open WebUI reasoning stream with GPT-5.4 mini | • `OpenAIResponsesChatGenerator` backend<br/>• Streams reasoning summaries to Open WebUI "Thinking"<br/>• Supports both Chat Completions and Responses endpoints | Use reasoning models with Open WebUI when you need visible reasoning blocks in streamed responses |
| [chat_with_website](./pipeline_wrappers/chat_with_website/) | Answer questions about website content | • Web content fetching<br/>• HTML to document conversion<br/>• Content-based Q&A<br/>• Configurable URLs | Creating AI assistants that can answer questions about specific websites or web-based documentation |
| [chat_with_website_mcp](./pipeline_wrappers/chat_with_website_mcp/) | MCP-compatible website chat pipeline | • MCP (Model Context Protocol) support<br/>• Website content analysis<br/>• API-only interface<br/>• Simplified deployment | Integrating website analysis capabilities into MCP-compatible AI systems and tools |
| [chat_with_website_streaming](./pipeline_wrappers/chat_with_website_streaming/) | Streaming website chat responses | • Real-time streaming<br/>• Website content processing<br/>• Progressive response generation<br/>• Enhanced user experience | Building responsive web applications that provide real-time AI responses about website content |
| [run_api_streaming](./pipeline_wrappers/run_api_streaming/) | Stream responses directly from `/run` endpoint | • `streaming_generator()` for `run_api`<br/>• `async_streaming_generator()` for `run_api_async`<br/>• Text/plain streaming output<br/>• Real-time token streaming | Building streaming APIs with the generic `/run` endpoint instead of chat-specific endpoints |
| [custom_tracing](./pipeline_wrappers/custom_tracing/) | YAML classifier with direct LLM call and tracing | • Accepts pipeline YAML input<br/>• OpenAI API classification (`io_bound` vs `cpu_bound`)<br/>• Prompt template in separate file<br/>• Nested custom spans in dashboard | Classifying Haystack pipelines by likely bottleneck type while tracing each classifier stage |
| [open_webui_agent_events](./pipeline_wrappers/open_webui_agent_events/) | Agent pipeline with OpenWebUI status events | • OpenWebUI event integration<br/>• Status event generation<br/>• Details tag creation<br/>• Agent-based responses<br/>• Real-time UI feedback | Creating interactive AI agents with rich status updates and progress indicators in OpenWebUI |
| [open_webui_agent_on_tool_calls](./pipeline_wrappers/open_webui_agent_on_tool_calls/) | Agent with tool call interception and OpenWebUI events | • Tool call lifecycle hooks<br/>• OpenWebUI notifications<br/>• Weather API tool integration<br/>• Real-time tool execution feedback<br/>• Status and result tracking | Building agents that provide detailed feedback about tool execution with rich UI interactions |
| [chainlit_weather_agent](./pipeline_wrappers/chainlit_weather_agent/) | Weather agent with Chainlit custom widget | • Real weather data via Open-Meteo API<br/>• Chainlit CustomElement (WeatherCard)<br/>• Tool call events and notifications<br/>• Async streaming | Demonstrating Chainlit custom UI widgets rendered from pipeline events, with a real external API |
| [image_generation](./pipeline_wrappers/image_generation/) | Return binary files (images) from `run_api` | • `FileResponse` return type<br/>• HuggingFace Inference API<br/>• Text-to-image generation<br/>• Direct binary response (no JSON wrapping) | Returning images, PDFs, or other binary files from pipeline endpoints |
| [responses_with_file_upload](./pipeline_wrappers/responses_with_file_upload/) | Agent-based Responses API with file reading | • Haystack Agent with `read_file` tool<br/>• `run_response_async` with streaming<br/>• `run_file_upload` with in-memory store<br/>• `_strip_tool_calls` for agentic clients<br/>• Codex CLI compatible | Building an agent that reads local files and uploaded files via the Responses API, compatible with Codex CLI and the OpenAI Python client |
| [chat_completion_with_file_upload](./pipeline_wrappers/chat_completion_with_file_upload/) | Chat Completions API with `/v1/files` upload | • `run_chat_completion_async` with streaming<br/>• `run_file_upload` with in-memory store<br/>• Resolves `{"type": "file"}` content parts<br/>• OpenAI file input format | Using the Chat Completions API with files uploaded via `/v1/files` and referenced using OpenAI's multi-part content format |
| [rag_indexing_query](./rag_indexing_query/) | Complete RAG system with Elasticsearch | • Document indexing pipeline<br/>• Query pipeline<br/>• Elasticsearch integration<br/>• Multiple file format support (PDF, Markdown, Text)<br/>• Sentence transformers embeddings | Implementing production-ready RAG systems for document search and knowledge retrieval |
| [shared_code_between_wrappers](./shared_code_between_wrappers/) | Code sharing between pipeline wrappers | • Shared library imports<br/>• HAYHOOKS_ADDITIONAL_PYTHON_PATH<br/>• Multiple deployment strategies<br/>• Code reusability | Organizing complex projects with multiple pipelines that share common functionality |

## Programmatic examples

| Example | Description | Key Features | Use Case |
|---------|-------------|--------------|----------|
| [api_key_auth](./programmatic/api_key_auth/) | API key authentication with a weather agent | • Programmatic `create_app()` usage<br/>• Middleware-based auth enforcement<br/>• Multiple keys via `HAYHOOKS_API_KEYS`<br/>• Swagger **Authorize** button support<br/>• Weather agent with tool calling | Protecting all Hayhooks endpoints with API-key authentication |

## Getting Started

Each example includes:

- **Pipeline wrapper implementation** (`pipeline_wrapper.py`)
- **Pipeline configuration** (`.yml` files where applicable)
- **Dependencies** (`requirements.txt` where applicable)
- **Documentation** (individual README files with setup instructions)

## Common Prerequisites

Most examples require:

- Python 3.10+
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
