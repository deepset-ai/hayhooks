# Responses API with Codex Client-Side Tools (+ Server Weather Tool)

Demonstrates a **hybrid tool-calling** pattern for the OpenAI Responses API (`/v1/responses`) where **Codex** acts as an agentic client and **Hayhooks** acts as the backend LLM server.

## How it works

1. **Codex sends a request** to Hayhooks' `/v1/responses` endpoint. The request contains the full conversation history (`input_items`) plus the schemas of Codex's own tools (e.g. `exec_command`, `write_stdin`, `spawn_agent`, …).
2. **Hayhooks receives the request** in `run_response_async`. It converts the Responses API `input_items` into Haystack `ChatMessage` objects, preserving the full conversation context — including any past `function_call` / `function_call_output` rounds from previous tool loops.
3. **(Optional) Server-side weather enrichment.** Before calling the LLM, Hayhooks checks whether the latest user prompt looks weather-related. If so, it runs a lightweight Haystack Agent with an Open-Meteo tool, then injects the real weather data as extra system context. This is invisible to Codex — it just sees a better answer.
4. **Hayhooks forwards the client tool schemas to the LLM** as `generation_kwargs`, so the model knows which tools Codex can execute. The LLM response is streamed back.
5. **If the LLM emits a `function_call`**, Codex receives it, **executes the tool locally** (e.g. runs a shell command), and sends the result back as a `function_call_output` in a new `/v1/responses` request — restarting the loop from step 1.

In short: **Codex owns tool execution**, **Hayhooks owns the LLM call** (and can optionally enrich it with server-side tools the client doesn't know about).

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
