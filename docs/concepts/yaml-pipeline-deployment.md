# YAML Pipeline Deployment

Hayhooks supports deploying Haystack pipelines directly from YAML definitions. This approach builds request/response schemas automatically from the YAML-declared `inputs` and `outputs`.

## Overview

YAML pipeline deployment is ideal for:

- Simple pipelines with clear inputs and outputs
- Quick deployment without wrapper code
- Automatic schema generation
- CI/CD pipeline deployments

!!! tip "Converting Existing Pipelines"
    If you already have a Haystack `Pipeline` instance, you can serialize it with `pipeline.dumps()` and then manually add the required `inputs` and `outputs` sections before deploying.

## Requirements

### YAML Structure

Your YAML file must include both `inputs` and `outputs` sections:

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

### Key Requirements

1. **`inputs` Section**: Maps friendly names to pipeline component fields
2. **`outputs` Section**: Maps pipeline outputs to response fields
3. **Valid Components**: All components must be properly defined
4. **Valid Connections**: All connections must reference existing components

## Deployment Methods

=== "CLI"

    ```bash
    # Deploy with default settings
    hayhooks pipeline deploy-yaml pipelines/chat_pipeline.yml

    # Deploy with custom name
    hayhooks pipeline deploy-yaml -n my_chat_pipeline pipelines/chat_pipeline.yml

    # Deploy with description
    hayhooks pipeline deploy-yaml -n my_chat_pipeline --description "Chat pipeline for Q&A" pipelines/chat_pipeline.yml

    # Overwrite existing pipeline
    hayhooks pipeline deploy-yaml -n my_chat_pipeline --overwrite pipelines/chat_pipeline.yml
    ```

=== "HTTP API"

    ```bash
    curl -X POST \
      http://localhost:1416/deploy-yaml \
      -H 'Content-Type: application/json' \
      -d '{
        "name": "my_chat_pipeline",
        "description": "Chat pipeline for Q&A",
        "yaml_content": "...",
        "overwrite": false
      }'
    ```

=== "Python"

    ```python
    import requests

    response = requests.post(
        "http://localhost:1416/deploy-yaml",
        json={
            "name": "my_chat_pipeline",
            "description": "Chat pipeline for Q&A",
            "yaml_content": "...",  # Your YAML content as string
            "overwrite": False
        }
    )
    print(response.json())
    ```

## CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--name` | `-n` | Override the pipeline name | YAML file stem |
| `--description` | | Human-readable description | Pipeline name |
| `--overwrite` | `-o` | Overwrite if pipeline exists | `false` |
| `--skip-mcp` | | Skip MCP tool registration | `false` |
| `--save-file` | | Save YAML to server | `true` |
| `--no-save-file` | | Don't save YAML to server | `false` |

## Input/Output Mapping

### Input Mapping

The `inputs` section maps friendly names to pipeline component fields:

```yaml
inputs:
  # friendly_name: component.field
  urls: fetcher.urls
  query: prompt_builder.query
```

**Mapping rules:**

- Use `component.field` syntax
- Field must exist in the component
- Multiple inputs can map to the same component field
- Input names become API parameters

### Output Mapping

The `outputs` section maps pipeline outputs to response fields:

```yaml
outputs:
  # response_field: component.field
  replies: llm.replies
  documents: fetcher.documents
  metadata: prompt_builder.metadata
```

**Mapping rules:**

- Use `component.field` syntax
- Field must exist in the component
- Response fields are serialized to JSON
- Complex objects are automatically serialized

## API Usage

### After Deployment

Once deployed, your pipeline is available at:

- **Run Endpoint**: `/{pipeline_name}/run`
- **Schema**: `/{pipeline_name}/schema`
- **OpenAPI**: `/openapi.json`

### Example Request

```bash
curl -X POST \
  http://localhost:1416/my_chat_pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{
    "urls": ["https://haystack.deepset.ai"],
    "query": "What is Haystack?"
  }'
```

### Example Response

```json
{
  "replies": ["Haystack is an open source framework..."],
  "documents": [...],
  "metadata": {...}
}
```

## Schema Generation

Hayhooks automatically generates:

### Request Schema

```json
{
  "type": "object",
  "properties": {
    "urls": {
      "type": "array",
      "items": {"type": "string"}
    },
    "query": {"type": "string"}
  },
  "required": ["urls", "query"]
}
```

### Response Schema

```json
{
  "type": "object",
  "properties": {
    "replies": {"type": "array"},
    "documents": {"type": "array"},
    "metadata": {"type": "object"}
  }
}
```

## Obtaining YAML from Existing Pipelines

You can obtain YAML from existing Haystack pipelines:

```python
from haystack import Pipeline

# Create or load your pipeline
pipeline = Pipeline()
# ... add components and connections ...

# Get YAML representation
yaml_content = pipeline.dumps()

# Save to file
with open("pipeline.yml", "w") as f:
    f.write(yaml_content)
```

!!! note
    You'll need to manually add the `inputs` and `outputs` sections to the generated YAML.

## Limitations

!!! warning "YAML Pipeline Limitations"
    YAML-deployed pipelines have the following limitations:

    - :octicons-x-circle-16: **No OpenAI Compatibility**: Don't support OpenAI-compatible chat endpoints
    - :octicons-x-circle-16: **No Streaming**: Streaming responses are not supported
    - :octicons-x-circle-16: **No File Uploads**: File upload handling is not available
    - :material-lightning-bolt: **Async Only**: Pipelines are run as `AsyncPipeline` instances

!!! tip "Using PipelineWrapper for Advanced Features"
    For advanced features, use `PipelineWrapper` instead:

```python
# For OpenAI compatibility
class PipelineWrapper(BasePipelineWrapper):
    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> Union[str, Generator]:
        ...

# For file uploads
class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, files: Optional[list[UploadFile]] = None, query: str = "") -> str:
        ...

# For streaming
class PipelineWrapper(BasePipelineWrapper):
    def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        ...
```

## Example

For a complete working example of a YAML pipeline with proper `inputs` and `outputs`, see:

- [tests/test_files/yaml/inputs_outputs_pipeline.yml](https://github.com/deepset-ai/hayhooks/blob/main/tests/test_files/yaml/inputs_outputs_pipeline.yml)

This example demonstrates:

- Complete pipeline structure with components and connections
- Proper `inputs` mapping to component fields
- Proper `outputs` mapping from component results
- Real-world usage with `LinkContentFetcher`, `HTMLToDocument`, `PromptBuilder`, and `OpenAIGenerator`

## Next Steps

- [PipelineWrapper](pipeline-wrapper.md) - For advanced features like streaming and chat completion
- [Examples](../examples/overview.md) - See working examples
