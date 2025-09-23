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

**Common initialization patterns:**

1. **From YAML file:**
```python
def setup(self) -> None:
    pipeline_yaml = (Path(__file__).parent / "pipeline.yml").read_text()
    self.pipeline = Pipeline.loads(pipeline_yaml)
```

2. **From Haystack template:**
```python
def setup(self) -> None:
    from haystack.pipelines import TemplatePipeline
    self.pipeline = TemplatePipeline.from_template("rag")
```

3. **Inline code:**
```python
def setup(self) -> None:
    from haystack.components import Fetcher, PromptBuilder, OpenAIGenerator

    fetcher = Fetcher()
    prompt_builder = PromptBuilder(template="Answer: {{query}}")
    llm = OpenAIGenerator()

    self.pipeline = Pipeline()
    self.pipeline.add_component("fetcher", fetcher)
    self.pipeline.add_component("prompt_builder", prompt_builder)
    self.pipeline.add_component("llm", llm)
    self.pipeline.connect("fetcher.content", "prompt_builder.documents")
    self.pipeline.connect("prompt_builder", "llm")
```

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

## Best Practices

### 1. Use Type Hints

```python
from typing import List, Optional, Dict, Any

def run_api(
    self,
    urls: List[str],
    question: str,
    temperature: Optional[float] = 0.7,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    ...
```

### 2. Add Logging

```python
from hayhooks import log

def run_api(self, urls: List[str], question: str) -> str:
    log.info(f"Processing question: {question}")
    log.debug(f"URLs: {urls}")

    result = self.pipeline.run(...)
    log.info(f"Pipeline completed successfully")
    return result["llm"]["replies"][0]
```

### 3. Validate Inputs

```python
def run_api(self, urls: List[str], question: str) -> str:
    if not urls:
        raise ValueError("At least one URL is required")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    # Continue with pipeline execution
    ...
```

## Examples

### Simple Q&A Pipeline

```python
class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components import PromptBuilder, OpenAIGenerator

        prompt_builder = PromptBuilder(template="Answer: {{query}}")
        llm = OpenAIGenerator()

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("prompt_builder", "llm")

    def run_api(self, query: str) -> str:
        result = self.pipeline.run({"prompt_builder": {"query": query}})
        return result["llm"]["replies"][0]
```

### Streaming Chat Pipeline

```python
from typing import AsyncGenerator
from hayhooks import async_streaming_generator, get_last_user_message

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components import PromptBuilder, OpenAIChatGenerator

        prompt_builder = PromptBuilder(template="Answer: {{query}}")
        llm = OpenAIChatGenerator(model="gpt-4", streaming_callback=lambda x: None)

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", prompt_builder)
        self.pipeline.add_component("llm", llm)
        self.pipeline.connect("prompt_builder", "llm")

    async def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        question = get_last_user_message(messages)
        return async_streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={"prompt_builder": {"query": question}},
        )
```

## Next Steps

- [YAML Pipeline Deployment](yaml-pipeline-deployment.md) - Alternative deployment method
- [Agent Deployment](agent-deployment.md) - Deploy Haystack agents
- [Examples](../examples/overview.md) - See working examples
