# Open WebUI Events Example

Send status updates and UI events to Open WebUI during streaming, and optionally intercept tool calls for richer feedback.

## Where is the code?

- Event examples: [open_webui_agent_events](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/open_webui_agent_events), [open_webui_agent_on_tool_calls](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/open_webui_agent_on_tool_calls)
- See the main docs → Open WebUI integration and event hooks

## Deploy (example)

```bash
hayhooks pipeline deploy-files -n agent_events examples/pipeline_wrappers/open_webui_agent_events
```

## Run

- OpenAI-compatible chat (events stream to Open WebUI):

```bash
curl -X POST http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "agent_events",
    "messages": [{"role": "user", "content": "Tell me about machine learning"}]
  }'
```

!!! tip "Working with Events"
    - Use helpers from `hayhooks.open_webui`: `create_status_event`, `create_message_event`, `create_replace_event`, `create_source_event`, `create_notification_event`, `create_details_tag`
    - Intercept tool calls via `on_tool_call_start`/`on_tool_call_end` with `streaming_generator`/`async_streaming_generator`
    - For recommended Open WebUI settings, see the [Open WebUI Integration](../features/openwebui-integration.md) guide

## Related

- General guide: [Main docs](../index.md)
- Examples index: [Examples Overview](overview.md)
