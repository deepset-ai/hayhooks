# PipelineWrapper

The `PipelineWrapper` class is the core component for deploying Haystack pipelines with Hayhooks. It provides maximum flexibility for pipeline initialization and execution.

## Why PipelineWrapper?

The pipeline wrapper provides a flexible foundation for deploying Haystack pipelines, agents or any other component by allowing users to:

- Choose their preferred initialization method (YAML files, Haystack templates, or inline code)
- Define custom execution logic with configurable inputs and outputs
- Optionally expose OpenAI-compatible chat endpoints with streaming support for integration with interfaces like [open-webui](https://openwebui.com/)

## Basic Structure

```python
from pathlib import Path
from typing import List, Generator, Union, AsyncGenerator
from haystack import Pipeline, AsyncPipeline
from hayhooks import BasePipelineWrapper, get_last_user_message, streaming_generator, async_streaming_generator

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "pipeline.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: List[str], question: str) -> str:
        result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
        return result["llm"]["replies"][0]
```

## Required Methods

### setup()

The `setup()` method is called once when the pipeline is deployed. It should initialize the `self.pipeline` attribute as a Haystack pipeline.

```python
def setup(self) -> None:
    # Initialize your pipeline here
    pass
```

**Initialization patterns:**

### 1. Programmatic Initialization (Recommended)

Define your pipeline directly in code for maximum flexibility and control:

```python
def setup(self) -> None:
    from haystack import Pipeline
    from haystack.components.fetchers import LinkContentFetcher
    from haystack.components.converters import HTMLToDocument
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator

    # Create components
    fetcher = LinkContentFetcher()
    converter = HTMLToDocument()
    prompt_builder = PromptBuilder(
        template="Based on: {{documents}}\nAnswer: {{query}}"
    )
    llm = OpenAIGenerator(model="gpt-4o-mini")

    # Build pipeline
    self.pipeline = Pipeline()
    self.pipeline.add_component("fetcher", fetcher)
    self.pipeline.add_component("converter", converter)
    self.pipeline.add_component("prompt", prompt_builder)
    self.pipeline.add_component("llm", llm)

    # Connect components
    self.pipeline.connect("fetcher.streams", "converter.sources")
    self.pipeline.connect("converter.documents", "prompt.documents")
    self.pipeline.connect("prompt.prompt", "llm.prompt")
```

!!! success "Benefits of Programmatic Initialization"
    - :material-code-braces: Full IDE support with autocomplete and type checking
    - :material-bug: Easier debugging and testing
    - :material-pencil: Better refactoring capabilities
    - :material-cog: Dynamic component configuration based on runtime conditions

### 2. Load from YAML

Load an existing YAML pipeline file:

```python
def setup(self) -> None:
    from pathlib import Path
    from haystack import Pipeline

    pipeline_yaml = (Path(__file__).parent / "pipeline.yml").read_text()
    self.pipeline = Pipeline.loads(pipeline_yaml)
```

**When to use:**

- You already have a YAML pipeline definition
- You want to version control pipeline structure separately
- You need to share pipeline definitions across different deployments

!!! tip "Consider YAML-only deployment"
    If your pipeline is simple and doesn't need custom logic, consider using [YAML Pipeline Deployment](yaml-pipeline-deployment.md) instead, which doesn't require a wrapper at all.

### run_api()

The `run_api()` method is called for each API request to the `{pipeline_name}/run` endpoint.

```python
def run_api(self, urls: List[str], question: str) -> str:
    result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
    return result["llm"]["replies"][0]
```

**Key features:**

- **Flexible Input**: You can define any input arguments you need
- **Automatic Validation**: Hayhooks creates Pydantic models for request validation
- **Type Safety**: Use proper type hints for better validation
- **Error Handling**: Implement proper error handling for production use

**Input argument rules:**

- Arguments must be JSON-serializable
- Use proper type hints (`List[str]`, `Optional[int]`, etc.)
- Default values are supported
- Complex types like `Dict[str, Any]` are allowed

## Optional Methods

### run_api_async()

The asynchronous version of `run_api()` for better performance under high load.

```python
async def run_api_async(self, urls: List[str], question: str) -> str:
    result = await self.pipeline.run_async({"fetcher": {"urls": urls}, "prompt": {"query": question}})
    return result["llm"]["replies"][0]
```

**When to use `run_api_async`:**

- Working with `AsyncPipeline` instances
- Handling many concurrent requests
- Integrating with async-compatible components
- Better performance for I/O-bound operations

### run_chat_completion()

Enable OpenAI-compatible chat endpoints for integration with chat interfaces.

```python
def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
    question = get_last_user_message(messages)
    result = self.pipeline.run({"prompt": {"query": question}})
    return result["llm"]["replies"][0]
```

**Fixed signature:**

- `model`: The pipeline name
- `messages`: OpenAI-format message list
- `body`: Full request body (for additional parameters)

### run_chat_completion_async()

Async version of chat completion with streaming support.

```python
async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
    question = get_last_user_message(messages)
    return async_streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt": {"query": question}},
    )
```

## Hybrid Streaming: Mixing Async and Sync Components

!!! tip "Compatibility for Legacy Components"
    When working with legacy pipelines or components that only support sync streaming callbacks (like `HuggingFaceLocalGenerator`), use `allow_sync_streaming_callbacks="auto"` to enable hybrid mode. For new code, prefer async-compatible components and use the default strict mode.

Some Haystack components only support synchronous streaming callbacks and don't have async equivalents. Examples include:

- `HuggingFaceLocalGenerator` - Local text generation (⚠️ Note: `HuggingFaceLocalChatGenerator` IS async-compatible)
- Other components without `run_async()` support

### The Problem

By default, `async_streaming_generator` requires all streaming components to support async callbacks:

```python
async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
    # This will FAIL if pipeline contains HuggingFaceLocalGenerator
    return async_streaming_generator(
        pipeline=self.pipeline,  # AsyncPipeline with HuggingFaceLocalGenerator
        pipeline_run_args={"prompt": {"query": question}},
    )
```

**Error:**

```text
ValueError: Component 'llm' of type 'HuggingFaceLocalGenerator' seems to not support
async streaming callbacks...
```

### The Solution: Hybrid Streaming Mode

Enable hybrid streaming mode to automatically handle both async and sync components:

```python
async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
    question = get_last_user_message(messages)
    return async_streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt": {"query": question}},
        allow_sync_streaming_callbacks="auto"  # ✅ Auto-detect and enable hybrid mode
    )
```

### How It Works

When `allow_sync_streaming_callbacks="auto"`:

1. **Detection**: Automatically scans pipeline components for async support
2. **Adaptation**: For sync-only components, wraps callbacks to work with async event loop
3. **Mixed Handling**: Async-capable components use native async callbacks, sync-only use wrapped callbacks
4. **Performance**: No overhead for fully async pipelines - hybrid mode only activates when needed

### Configuration Options

```python
# Option 1: Strict mode (Default - Recommended)
allow_sync_streaming_callbacks=False
# → Raises error if sync-only components found
# → Best for: New code, ensuring proper async components, best performance

# Option 2: Auto-detection (Compatibility mode)
allow_sync_streaming_callbacks="auto"
# → Automatically detects and enables hybrid mode only when needed
# → Best for: Legacy pipelines, components without async support, gradual migration
```

### Example: HuggingFace Local Model with Async Pipeline

```python
from typing import List, AsyncGenerator
from haystack import AsyncPipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from hayhooks import BasePipelineWrapper, get_last_user_message, async_streaming_generator

class LocalLLMWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # HuggingFaceLocalGenerator only supports sync streaming
        llm = HuggingFaceLocalGenerator(
            model="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            generation_kwargs={"max_new_tokens": 100, "num_beams": 1}
        )

        prompt_builder = PromptBuilder(
            template="Answer this question: {{question}}"
        )

        self.pipeline = AsyncPipeline()
        self.pipeline.add_component("prompt", prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("prompt.prompt", "llm.prompt")

    async def run_chat_completion_async(
        self, model: str, messages: List[dict], body: dict
    ) -> AsyncGenerator:
        question = get_last_user_message(messages)

        # Enable hybrid mode for HuggingFaceLocalGenerator
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt": {"question": question}},
            allow_sync_streaming_callbacks="auto"  # ✅ Handles sync component
        )
```

### When to Use Each Mode

**Use strict mode (default) when:**

- ✅ Building new pipelines (recommended default)
- ✅ You want to ensure all components are async-compatible
- ✅ Performance is critical (pure async is ~1-2μs faster per chunk)
- ✅ You're building a production system with controlled dependencies

**Use `allow_sync_streaming_callbacks="auto"` when:**

- ✅ Working with legacy pipelines that use `HuggingFaceLocalGenerator`
- ✅ Deploying YAML pipelines with unknown/legacy component types
- ✅ Migrating old code that doesn't have async equivalents yet
- ✅ Third-party components without async support

### Performance Considerations

- **Pure async pipeline**: No overhead
- **Hybrid mode (auto-detected)**: Minimal overhead (~1-2 microseconds per streaming chunk for sync components)
- **Network-bound operations**: The overhead is negligible compared to LLM generation time

!!! success "Best Practice"
    **For new code**: Use the default strict mode (`allow_sync_streaming_callbacks=False`) to ensure you're using proper async components.

    **For legacy/compatibility**: Use `allow_sync_streaming_callbacks="auto"` when working with older pipelines or components that don't support async streaming yet.

## Streaming from Multiple Components

!!! info "Smart Streaming Behavior"
    By default, Hayhooks streams only the **last** streaming-capable component in your pipeline. This is usually what you want - the final output streaming to users.

For advanced use cases, you can control which components stream using the `streaming_components` parameter.

When your pipeline contains multiple components that support streaming (e.g., multiple LLMs), you can control which ones stream their outputs as the pipeline executes.

### Default Behavior: Stream Only the Last Component

By default, only the last streaming-capable component will stream:

```python
class MultiLLMWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage

        self.pipeline = Pipeline()

        # First LLM - initial answer
        self.pipeline.add_component(
            "prompt_1",
            ChatPromptBuilder(
                template=[
                    ChatMessage.from_system("You are a helpful assistant."),
                    ChatMessage.from_user("{{query}}")
                ]
            )
        )
        self.pipeline.add_component("llm_1", OpenAIChatGenerator(model="gpt-4o-mini"))

        # Second LLM - refines the answer using Jinja2 to access ChatMessage attributes
        self.pipeline.add_component(
            "prompt_2",
            ChatPromptBuilder(
                template=[
                    ChatMessage.from_system("You are a helpful assistant that refines responses."),
                    ChatMessage.from_user(
                        "Previous response: {{previous_response[0].text}}\n\nRefine this."
                    )
                ]
            )
        )
        self.pipeline.add_component("llm_2", OpenAIChatGenerator(model="gpt-4o-mini"))

        # Connect components - LLM 1's replies go directly to prompt_2
        self.pipeline.connect("prompt_1.prompt", "llm_1.messages")
        self.pipeline.connect("llm_1.replies", "prompt_2.previous_response")
        self.pipeline.connect("prompt_2.prompt", "llm_2.messages")

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Generator:
        question = get_last_user_message(messages)

        # By default, only llm_2 (the last streaming component) will stream
        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt_1": {"query": question}}
        )
```

**What happens:** Only `llm_2` (the last streaming-capable component) streams its responses token by token. The first LLM (`llm_1`) executes normally without streaming, and only the final refined output streams to the user.

### Advanced: Stream Multiple Components with `streaming_components`

For advanced use cases where you want to see outputs from multiple components, use the `streaming_components` parameter:

```python
def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Generator:
    question = get_last_user_message(messages)

    # Enable streaming for BOTH LLMs
    return streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt_1": {"query": question}},
        streaming_components=["llm_1", "llm_2"]  # Stream both components
    )
```

**What happens:** Both LLMs stream their responses token by token. First you'll see the initial answer from `llm_1` streaming, then the refined answer from `llm_2` streaming.

You can also selectively enable streaming for specific components:

```python
# Stream only the first LLM
streaming_components=["llm_1"]

# Stream only the second LLM (same as default)
streaming_components=["llm_2"]

# Stream ALL capable components (shorthand)
streaming_components="all"

# Stream ALL capable components (specific list)
streaming_components=["llm_1", "llm_2"]
```

### Using the "all" Keyword

The `"all"` keyword is a convenient shorthand to enable streaming for all capable components:

```python
return streaming_generator(
    pipeline=self.pipeline,
    pipeline_run_args={...},
    streaming_components="all"  # Enable all streaming components
)
```

This is equivalent to explicitly enabling every streaming-capable component in your pipeline.

### Global Configuration via Environment Variable

You can set a global default using the `HAYHOOKS_STREAMING_COMPONENTS` environment variable. This applies to all pipelines unless overridden:

```bash
# Stream all components by default
export HAYHOOKS_STREAMING_COMPONENTS="all"

# Stream specific components (comma-separated)
export HAYHOOKS_STREAMING_COMPONENTS="llm_1,llm_2"
```

**Priority order:**

1. Explicit `streaming_components` parameter (highest priority)
2. `HAYHOOKS_STREAMING_COMPONENTS` environment variable
3. Default behavior: stream only last component (lowest priority)

!!! tip "When to Use Each Approach"
    - **Default (last component only)**: Best for most use cases - users see only the final output
    - **"all" keyword**: Useful for debugging, demos, or transparent multi-step workflows
    - **List of components**: Enable multiple specific components by name
    - **Environment variable**: For deployment-wide defaults without code changes

!!! note "Async Streaming"
    All streaming_components options work identically with `async_streaming_generator()` for async pipelines.

### YAML Pipeline Streaming Configuration

You can also specify streaming configuration in YAML pipeline definitions:

```yaml
components:
  prompt_1:
    type: haystack.components.builders.PromptBuilder
    init_parameters:
      template: "Answer this question: {{query}}"
  llm_1:
    type: haystack.components.generators.OpenAIGenerator
  prompt_2:
    type: haystack.components.builders.PromptBuilder
    init_parameters:
      template: "Refine this response: {{previous_reply}}"
  llm_2:
    type: haystack.components.generators.OpenAIGenerator

connections:
  - sender: prompt_1.prompt
    receiver: llm_1.prompt
  - sender: llm_1.replies
    receiver: prompt_2.previous_reply
  - sender: prompt_2.prompt
    receiver: llm_2.prompt

inputs:
  query: prompt_1.query

outputs:
  replies: llm_2.replies

# Option 1: List specific components
streaming_components:
  - llm_1
  - llm_2

# Option 2: Stream all components
# streaming_components: all
```

YAML configuration follows the same priority rules: YAML setting > environment variable > default.

See the [Multi-LLM Streaming Example](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/multi_llm_streaming) for a complete working implementation.

## File Upload Support

Hayhooks can handle file uploads by adding a `files` parameter:

```python
from fastapi import UploadFile
from typing import Optional, List

def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
    if files:
        # Process uploaded files
        filenames = [f.filename for f in files if f.filename]
        file_contents = [f.file.read() for f in files]
        return f"Processed {len(files)} files: {', '.join(filenames)}"

    return "No files uploaded"
```

## PipelineWrapper Development

### During Development

Use the `--overwrite` flag for rapid development:

```bash
hayhooks pipeline deploy-files -n my_pipeline --overwrite ./path/to/pipeline
```

**Development workflow:**

1. Make changes to your pipeline wrapper
2. Redeploy with `--overwrite`
3. Test the changes
4. Repeat as needed

### For even faster iterations

Combine `--overwrite` with `--skip-saving-files`:

```bash
hayhooks pipeline deploy-files -n my_pipeline --overwrite --skip-saving-files ./path/to/pipeline
```

This avoids writing files to disk and speeds up development.

## Additional Dependencies

Your pipeline wrapper may require additional dependencies:

```python
# pipeline_wrapper.py
import trafilatura  # Additional dependency

def run_api(self, urls: List[str], question: str) -> str:
    # Use additional library
    content = trafilatura.fetch(urls[0])
    # ... rest of pipeline logic
```

**Install dependencies:**

```bash
pip install trafilatura
```

**Debugging tip:** Enable tracebacks to see full error messages:

```bash
HAYHOOKS_SHOW_TRACEBACKS=true hayhooks run
```

## Error Handling

Implement proper error handling in production:

```python
from hayhooks import log
from fastapi import HTTPException

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        try:
            self.pipeline = self._create_pipeline()
        except Exception as e:
            log.error(f"Failed to initialize pipeline: {e}")
            raise

    def run_api(self, query: str) -> str:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        try:
            result = self.pipeline.run({"prompt": {"query": query}})
            return result["llm"]["replies"][0]
        except Exception as e:
            log.error(f"Pipeline execution failed: {e}")
            raise HTTPException(status_code=500, detail="Pipeline execution failed")
```

## MCP Tool Configuration

### Skip MCP Tool Listing

To skip MCP tool registration:

```python
class PipelineWrapper(BasePipelineWrapper):
    skip_mcp = True  # This pipeline won't be listed as an MCP tool

    def setup(self) -> None:
        ...

    def run_api(self, ...) -> str:
        ...
```

### MCP Tool Description

Use docstrings to provide MCP tool descriptions:

```python
def run_api(self, urls: List[str], question: str) -> str:
    """
    Ask questions about website content.

    Args:
        urls: List of website URLs to analyze
        question: Question to ask about the content

    Returns:
        Answer to the question based on the website content
    """
    result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
    return result["llm"]["replies"][0]
```

## Examples

For complete, working examples see:

- **[Chat with Website (Streaming)](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/chat_with_website_streaming)** - Pipeline with streaming chat completion support
- **[Async Question Answer](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/async_question_answer)** - Async pipeline patterns with streaming
- **[RAG Indexing & Query](https://github.com/deepset-ai/hayhooks/tree/main/examples/rag_indexing_query)** - Complete RAG system with file uploads and Elasticsearch

## Next Steps

- [YAML Pipeline Deployment](yaml-pipeline-deployment.md) - Alternative deployment method
- [Agent Deployment](agent-deployment.md) - Deploy Haystack agents
