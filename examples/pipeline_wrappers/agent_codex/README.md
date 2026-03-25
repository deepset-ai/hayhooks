# Responses API with Codex Client-Side Tools (+ Server Weather Tool)

Demonstrates a **hybrid** pattern for the OpenAI Responses API (`/v1/responses`):

- The model emits function calls for client-side tools (e.g. `exec_command`)
- Codex executes those tools locally and sends `function_call_output` back
- Weather requests are enriched with a **server-side** Haystack Agent (`server_get_weather`) backed by Open-Meteo

## Module layout

- `pipeline_wrapper.py` — orchestration: message building, weather enrichment, LLM streaming
- `client_tools.py` — Responses tool schema normalization and generation kwargs
- `weather.py` — Open-Meteo weather tool + weather agent

## Requirements

```bash
export OPENAI_API_KEY="sk-..."
```

## Deploy

```bash
hayhooks run
hayhooks pipeline deploy-files -n agent_codex examples/pipeline_wrappers/agent_codex
```

## Codex CLI setup

Add a Hayhooks profile to your Codex config:

```toml
# ~/.codex/config.toml

[model_providers.hayhooks]
name = "Hayhooks"
base_url = "http://localhost:1416/v1"
wire_api = "responses"

[profiles.hayhooks]
model_provider = "hayhooks"
model = "agent_codex"
model_reasoning_effort = "medium"
```

Then launch Codex with:

```bash
codex --profile hayhooks
```

Client-side tools (e.g. `exec_command`) are executed locally by Codex.
Weather-related prompts (e.g. "What's the weather in Berlin?") trigger a server-side lookup via Open-Meteo before the main LLM call.
