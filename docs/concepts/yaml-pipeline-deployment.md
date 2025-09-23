# YAML Pipeline Deployment

Hayhooks supports deploying Haystack pipelines directly from YAML definitions. This approach builds request/response schemas automatically from the YAML-declared `inputs` and `outputs`.

## Overview

YAML pipeline deployment is ideal for:

- Simple pipelines with clear inputs and outputs
- Quick deployment without wrapper code
- Automatic schema generation
- CI/CD pipeline deployments

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

### CLI Deployment

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

### HTTP API Deployment

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
  temperature: llm.temperature
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
    "query": "What is Haystack?",
    "temperature": 0.7
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
    "query": {"type": "string"},
    "temperature": {
      "type": "number",
      "default": 0.7
    }
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

**Note:** You'll need to manually add the `inputs` and `outputs` sections to the generated YAML.

## Limitations

### Current Limitations

1. **No OpenAI Compatibility**: YAML-deployed pipelines don't support OpenAI-compatible chat endpoints
2. **No Streaming**: Streaming responses are not supported
3. **No File Uploads**: File upload handling is not available
4. **Async Only**: Pipelines are run as `AsyncPipeline` instances

### Workarounds

For advanced features, use `PipelineWrapper` instead:

```python
# For OpenAI compatibility
class PipelineWrapper(BasePipelineWrapper):
    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        ...

# For file uploads
class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
        ...

# For streaming
class PipelineWrapper(BasePipelineWrapper):
    def run_chat_completion_async(self, model: str, messages: List[dict], body: dict) -> AsyncGenerator:
        ...
```

## Best Practices

### 1. YAML Structure

```yaml
# Good structure
components:
  fetcher:
    type: haystack.components.fetchers.LinkContentFetcher
    init_parameters:
      timeout: 30

connections:
  - sender: fetcher.content
    receiver: prompt_builder.documents

inputs:
  urls: fetcher.urls
  query: prompt_builder.query

outputs:
  replies: llm.replies
```

### 2. Naming Conventions

```yaml
# Use descriptive names
inputs:
  website_urls: fetcher.urls      # Clear purpose
  user_question: prompt_builder.query  # Clear intent

outputs:
  ai_response: llm.replies        # Clear meaning
  source_documents: fetcher.content  # Clear source
```

### 3. Error Handling

While YAML pipelines don't allow custom error handling, you can:

- Use component-level error handling
- Set proper default values
- Validate inputs in the pipeline

### 4. Performance Considerations

```yaml
# Use efficient components
components:
  llm:
    type: haystack.components.generators.OpenAIChatGenerator
    init_parameters:
      model: gpt-3.5-turbo  # Faster and cheaper
      max_tokens: 1000     # Reasonable limit
```

## Migration from PipelineWrapper

To migrate from `PipelineWrapper` to YAML deployment:

1. **Extract Pipeline Logic**:
   ```python
   # From PipelineWrapper.setup()
   pipeline_yaml = (Path(__file__).parent / "pipeline.yml").read_text()
   self.pipeline = Pipeline.loads(pipeline_yaml)
   ```

2. **Create YAML File**:
   ```yaml
   # pipeline.yml
   components:
     # ... your components ...

   connections:
     # ... your connections ...

   inputs:
     # Map run_api parameters to component inputs
     urls: fetcher.urls
     question: prompt_builder.query

   outputs:
     # Map component outputs to response fields
     replies: llm.replies
   ```

3. **Deploy with YAML**:
   ```bash
   hayhooks pipeline deploy-yaml -n my_pipeline pipeline.yml
   ```

## Examples

### Simple Q&A Pipeline

```yaml
components:
  prompt_builder:
    type: haystack.components.builders.PromptBuilder
    init_parameters:
      template: "Answer: {{query}}"
  llm:
    type: haystack.components.generators.OpenAIGenerator

connections:
  - sender: prompt_builder
    receiver: llm

inputs:
  query: prompt_builder.query

outputs:
  replies: llm.replies
```

### Web Content Analysis

```yaml
components:
  fetcher:
    type: haystack.components.fetchers.LinkContentFetcher
    init_parameters:
      timeout: 30
  converter:
    type: haystack.components.converters.HTMLToDocument
  prompt_builder:
    type: haystack.components.builders.PromptBuilder
    init_parameters:
      template: "Based on this content: {{documents}}\nAnswer: {{query}}"
  llm:
    type: haystack.components.generators.OpenAIGenerator

connections:
  - sender: fetcher.content
    receiver: converter
  - sender: converter.documents
    receiver: prompt_builder.documents
  - sender: prompt_builder
    receiver: llm

inputs:
  urls: fetcher.urls
  query: prompt_builder.query

outputs:
  replies: llm.replies
  documents: converter.documents
```

## Next Steps

- [PipelineWrapper](pipeline-wrapper.md) - For advanced features
- [Agent Deployment](agent-deployment.md) - Deploy Haystack agents
- [Examples](../examples/overview.md) - See working examples