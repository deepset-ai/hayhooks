# Pipeline Deployment

Hayhooks provides flexible options for deploying Haystack pipelines and agents. This section covers the core concepts of pipeline deployment.

## Deployment Methods

### 1. PipelineWrapper Deployment (Recommended)

The most flexible approach using a `PipelineWrapper` class that encapsulates your pipeline logic.

**Key Features:**

- Maximum flexibility for initialization
- Custom execution logic
- OpenAI-compatible endpoint support
- Streaming support
- File upload handling

See [PipelineWrapper Details](pipeline-wrapper.md) for complete implementation guide and examples.

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

For a complete YAML example and detailed requirements, see [YAML Pipeline Deployment](yaml-pipeline-deployment.md).

!!! warning "YAML Pipeline Limitations"
    YAML-deployed pipelines do not support OpenAI-compatible chat endpoints or streaming. For chat/streaming (e.g., Open WebUI), use a `PipelineWrapper` and implement `run_chat_completion`/`run_chat_completion_async` (see [OpenAI Compatibility](../features/openai-compatibility.md)).

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

- **`run_chat_completion()`**: OpenAI-compatible chat endpoint (see [OpenAI Compatibility](../features/openai-compatibility.md))
  - Enable Open WebUI integration
  - Support chat completion format

- **`run_chat_completion_async()`**: Async chat completion (see [OpenAI Compatibility](../features/openai-compatibility.md))
  - Streaming support for chat interfaces
  - Better performance for concurrent chat requests

### Input/Output Handling

Hayhooks automatically handles:

- **Request Validation**: Pydantic models for input validation
- **Response Serialization**: JSON serialization of responses
- **File Uploads**: Automatic handling of multipart/form-data requests (see [File Upload Support](../features/file-upload-support.md))
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

### MCP Integration

All deployed pipelines can be exposed as MCP tools:

- **Automatic Discovery**: Pipelines are automatically listed as available tools
- **Schema Generation**: Input schemas are generated from method signatures
- **Tool Execution**: Tools can be called from MCP clients (see [MCP Support](../features/mcp-support.md))

## Next Steps

- [PipelineWrapper Details](pipeline-wrapper.md) - Learn about PipelineWrapper implementation
- [YAML Pipeline Deployment](yaml-pipeline-deployment.md) - Deploy from YAML files
