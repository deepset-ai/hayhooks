# Pipeline Deployment

Hayhooks provides flexible options for deploying Haystack pipelines and agents. This section covers the core concepts of pipeline deployment.

## Deployment Methods

### 1. PipelineWrapper Deployment (Recommended)

The most flexible approach is to create a `PipelineWrapper` class that encapsulates your pipeline logic.

**Key Features:**
- Maximum flexibility for initialization
- Custom execution logic
- OpenAI-compatible endpoint support
- Streaming support
- File upload handling

**Basic Structure:**
```python
from pathlib import Path
from typing import List
from haystack import Pipeline
from hayhooks import BasePipelineWrapper

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Initialize your pipeline
        pipeline_yaml = (Path(__file__).parent / "pipeline.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: List[str], question: str) -> str:
        # Custom execution logic
        result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
        return result["llm"]["replies"][0]
```

### 2. YAML Pipeline Deployment

Deploy pipelines directly from YAML definitions with automatic schema generation.

**Key Features:**
- Simple deployment from YAML files
- Automatic request/response schema generation
- No wrapper code required
- Perfect for straightforward pipelines

**Requirements:**
- YAML must include `inputs` and `outputs` sections
- Pipeline components must be properly defined

**Example YAML:**
```yaml
components:
  fetcher:
    type: haystack.components.fetchers.LinkContentFetcher
  prompt_builder:
    type: haystack.components.builders.PromptBuilder
    init_parameters:
      template: "Answer this question: {{query}} based on this content: {{documents}}"
  llm:
    type: haystack.components.generators.OpenAIGenerator

connections:
  - sender: fetcher.content
    receiver: prompt_builder.documents
  - sender: prompt_builder
    receiver: llm

inputs:
  urls: fetcher.urls
  query: prompt_builder.query

outputs:
  replies: llm.replies
```

## Core Components

### BasePipelineWrapper Class

All pipeline wrappers inherit from `BasePipelineWrapper`:

#### Required Methods

- **`setup()`**: Called once when the pipeline is deployed
  - Initialize your pipeline instance
  - Set up any required resources

- **`run_api()`**: Called for each API request
  - Define your custom execution logic
  - Return the pipeline result

#### Optional Methods

- **`run_api_async()`**: Async version of `run_api()`
  - Better performance for concurrent requests
  - Supports async pipeline execution

- **`run_chat_completion()`**: OpenAI-compatible chat endpoint
  - Enable OpenWebUI integration
  - Support chat completion format

- **`run_chat_completion_async()`**: Async chat completion
  - Streaming support for chat interfaces
  - Better performance for concurrent chat requests

### Input/Output Handling

Hayhooks automatically handles:

- **Request Validation**: Pydantic models for input validation
- **Response Serialization**: JSON serialization of responses
- **File Uploads**: Automatic handling of multipart/form-data requests
- **Type Conversion**: Automatic type conversion between JSON and Python types

## Lifecycle Management

### Pipeline Registration

When you deploy a pipeline, Hayhooks:

1. **Validates** the wrapper implementation
2. **Creates** the pipeline instance using `setup()`
3. **Registers** the pipeline with the server
4. **Generates** API endpoints and schemas
5. **Creates** OpenAI-compatible endpoints (if implemented)

### Pipeline Execution

For each request:

1. **Validates** the request against the schema
2. **Calls** the appropriate method (`run_api`, `run_chat_completion`, etc.)
3. **Handles** errors and exceptions
4. **Returns** the response in the correct format

### Pipeline Undeployment

When you undeploy a pipeline:

1. **Removes** the pipeline from the registry
2. **Deletes** the pipeline files (if saved)
3. **Unregisters** all API endpoints
4. **Cleans up** resources

## Configuration Options

### Deployment Options

| Option | Description | Default |
|--------|-------------|---------|
| `--name, -n` | Pipeline name | Directory name |
| `--description` | Human-readable description | Pipeline name |
| `--overwrite, -o` | Overwrite existing pipeline | `false` |
| `--skip-saving-files` | Don't save pipeline files | `false` |
| `--skip-mcp` | Skip MCP tool registration | `false` |

### MCP Integration

All deployed pipelines can be exposed as MCP tools:

- **Automatic Discovery**: Pipelines are automatically listed as available tools
- **Schema Generation**: Input schemas are generated from method signatures
- **Tool Execution**: Tools can be called from MCP clients

## Best Practices

### 1. Pipeline Wrapper Structure

```python
class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Initialize pipeline with error handling
        try:
            self.pipeline = self._create_pipeline()
        except Exception as e:
            log.error(f"Failed to initialize pipeline: {e}")
            raise

    def run_api(self, urls: List[str], question: str) -> str:
        # Add logging and error handling
        log.info(f"Processing question: {question}")
        try:
            result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
            return result["llm"]["replies"][0]
        except Exception as e:
            log.error(f"Pipeline execution failed: {e}")
            raise
```

### 2. Type Hints

Use proper type hints for better validation and documentation:

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

### 3. Error Handling

Implement proper error handling:

```python
def run_api(self, query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    try:
        result = self.pipeline.run({"prompt": {"query": query}})
        return result["llm"]["replies"][0]
    except Exception as e:
        log.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail="Pipeline execution failed")
```

## Next Steps

- [PipelineWrapper Details](pipeline-wrapper.md) - Learn about PipelineWrapper implementation
- [YAML Pipeline Deployment](yaml-pipeline-deployment.md) - Deploy from YAML files
- [Agent Deployment](agent-deployment.md) - Deploy Haystack agents
- [Examples](../examples/overview.md) - See working examples