# Pipeline Deployment

Hayhooks provides flexible options for deploying Haystack pipelines and agents. This section covers the core concepts of pipeline deployment.

## Deployment Methods

### 1. PipelineWrapper Deployment (Recommended)

The most flexible approach using a `PipelineWrapper` class that encapsulates your pipeline logic. This method provides maximum flexibility for initialization, custom execution logic, OpenAI-compatible endpoint support, streaming capabilities, and file upload handling.

See [PipelineWrapper Details](pipeline-wrapper.md) for complete implementation guide and examples.

### 2. YAML Pipeline Deployment

Deploy pipelines directly from YAML definitions with automatic schema generation. This approach offers simple deployment from YAML files with automatic request/response schema generation, no wrapper code required, making it perfect for straightforward pipelines. The YAML must include `inputs` and `outputs` sections with properly defined pipeline components.

For a complete YAML example and detailed requirements, see [YAML Pipeline Deployment](yaml-pipeline-deployment.md).

!!! warning "YAML Pipeline Limitations"
    YAML-deployed pipelines do not support OpenAI-compatible chat endpoints or streaming. For chat/streaming (e.g., Open WebUI), use a `PipelineWrapper` and implement `run_chat_completion`/`run_chat_completion_async` (see [OpenAI Compatibility](../features/openai-compatibility.md)).

## Core Components

### BasePipelineWrapper Class

All pipeline wrappers inherit from `BasePipelineWrapper`:

#### Required Methods

**`setup()`** is called once when the pipeline is deployed to initialize your pipeline instance and set up any required resources.

**`run_api()`** is called for each API request to define your custom execution logic and return the pipeline result.

#### Optional Methods

**`run_api_async()`** is the async version of `run_api()`, providing better performance for concurrent requests and supporting async pipeline execution.

**`run_chat_completion()`** enables OpenAI-compatible chat endpoints for Open WebUI integration with chat completion format support (see [OpenAI Compatibility](../features/openai-compatibility.md)).

**`run_chat_completion_async()`** provides async chat completion with streaming support for chat interfaces and better performance for concurrent chat requests (see [OpenAI Compatibility](../features/openai-compatibility.md)).

### Input/Output Handling

Hayhooks automatically handles request validation using Pydantic models, JSON serialization of responses, multipart/form-data requests for file uploads (see [File Upload Support](../features/file-upload-support.md)), and automatic type conversion between JSON and Python types.

## Lifecycle Management

### Pipeline Registration

When you deploy a pipeline, Hayhooks validates the wrapper implementation, creates the pipeline instance using `setup()`, registers the pipeline with the server, generates API endpoints and schemas, and creates OpenAI-compatible endpoints if implemented.

### Pipeline Execution

For each request, Hayhooks validates the request against the schema, calls the appropriate method (`run_api`, `run_chat_completion`, etc.), handles errors and exceptions, and returns the response in the correct format.

### Pipeline Undeployment

When you undeploy a pipeline, Hayhooks removes the pipeline from the registry, deletes the pipeline files if saved, unregisters all API endpoints, and cleans up resources.

### MCP Integration

All deployed pipelines can be exposed as MCP tools. Pipelines are automatically listed as available tools with input schemas generated from method signatures. Tools can be called from MCP clients (see [MCP Support](../features/mcp-support.md)).

## Next Steps

- [PipelineWrapper Details](pipeline-wrapper.md) - Learn about PipelineWrapper implementation
- [YAML Pipeline Deployment](yaml-pipeline-deployment.md) - Deploy from YAML files
