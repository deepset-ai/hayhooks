# Custom Tracing Classifier Example

This example classifies a Haystack pipeline YAML as **I/O-bound** or **CPU-bound** using a
direct OpenAI API call, and traces each stage in the dashboard.

## What this demonstrates

- A custom Haystack component (`PipelineBoundClassifier`) that performs direct LLM classification via the OpenAI SDK.
- Prompt template loaded from a separate file (`classification_prompt_template.txt`).
- Custom tracing spans around:
  - input resolution (inline YAML vs uploaded files)
  - prompt rendering
  - OpenAI API request

## Files

```text
pipeline_wrapper.py
classification_prompt_template.txt
README.md
```

## Setup

Install dependencies and set OpenAI credentials:

```bash
pip install "hayhooks[tracing]" openai
export OPENAI_API_KEY=your_key_here
# optional override
export OPENAI_MODEL=gpt-5.4-mini
```

Enable the dashboard (Haystack span mirroring is on by default):

```bash
hayhooks run --with-tracing-dashboard
```

Deploy the example:

```bash
cd examples/pipeline_wrappers/custom_tracing
hayhooks pipeline deploy-files -n custom_tracing .
```

## Example request

JSON body:

```bash
curl -X POST "http://localhost:1416/custom_tracing/run" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_yaml": "components:\n  router:\n    type: haystack.components.routers.ConditionalRouter\n  agent:\n    type: haystack.components.agents.Agent\n  mcp_tool_invoker:\n    type: haystack_integrations.components.tools.mcp.MCPToolInvoker\nconnections: []\nmetadata: {}"
  }'
```

Multipart upload (YAML file):

```bash
curl -X POST "http://localhost:1416/custom_tracing/run" \
  -F "files=@./my_pipeline.yml"
```

Example response:

```json
{
  "classification": "io_bound",
  "confidence": 0.94,
  "rationale": "The pipeline relies on an Agent and MCP tool invocations, which are network-heavy.",
  "model": "gpt-5.4-mini",
  "input_source": "file_upload",
  "input_filename": "my_pipeline.yml"
}
```

## Tracing output

Open `http://localhost:1416/dashboard` and trigger a request. You should see:

- `hayhooks.pipeline.run` (automatic endpoint span)
- `hayhooks.classifier.resolve_input`
- `haystack.component.run` (custom component span, tagged with `haystack.component.name: PipelineBoundClassifier`)
- `hayhooks.classifier.render_prompt`
- `hayhooks.classifier.openai_call`
