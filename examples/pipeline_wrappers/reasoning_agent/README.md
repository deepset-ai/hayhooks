# Reasoning Agent (GPT-5.4 mini)

This example shows how to stream reasoning content to Open WebUI using:

- `OpenAIResponsesChatGenerator`
- backend model `gpt-5.4-mini`
- Hayhooks OpenAI-compatible endpoints (`/v1/chat/completions` and `/v1/responses`)

## Why this example exists

`async_hybrid_streaming` uses the legacy `OpenAIGenerator`, which does not expose
reasoning chunks in Haystack `StreamingChunk` objects. As a result, Open WebUI
cannot render "Thinking" blocks from that example.

This wrapper uses `OpenAIResponsesChatGenerator`, which emits reasoning summary
chunks that Hayhooks forwards to Open WebUI.

## Run

```bash
export OPENAI_API_KEY=your_api_key_here
hayhooks deploy examples/pipeline_wrappers/reasoning_agent
```

The deployed model name is `reasoning_agent`.

## Quick verification (Chat Completions stream)

```bash
curl -N -X POST http://localhost:1416/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reasoning_agent",
    "messages": [{"role": "user", "content": "Solve 234*567 step by step"}],
    "stream": true
  }'
```

In the SSE stream, reasoning chunks appear as `reasoning_content` deltas.

## Open WebUI setup

1. Add an OpenAI-compatible connection pointing to `http://localhost:1416/v1`.
2. Select model `reasoning_agent`.
3. Ask a reasoning-heavy question (math, planning, debugging).
4. Open WebUI should render collapsible **Thinking** sections.

## Notes

- This example enables reasoning summaries by default with:
  `{"reasoning": {"effort": "high", "summary": "auto"}}`
- Reasoning effort is fixed in this example. Edit `DEFAULT_GENERATION_KWARGS` in `pipeline_wrapper.py` if you want a different value.
