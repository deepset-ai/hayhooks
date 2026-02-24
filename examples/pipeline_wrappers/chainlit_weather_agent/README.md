# Chainlit Weather Agent Example

This example demonstrates how to build a Haystack Agent that fetches **real weather data** and renders a **custom Chainlit widget** (WeatherCard) in the embedded Chainlit UI.

The heart of the example is `pipeline_wrapper.py`, which

1. Instantiates a Haystack `Agent` backed by an `OpenAIChatGenerator`.
2. Registers a `get_weather` tool that calls the free [Open-Meteo API](https://open-meteo.com/) to fetch current weather conditions.
3. Uses `on_tool_call_start` / `on_tool_call_end` callbacks to emit streaming events:
   - **status** events for progress indication
   - **notification** events for user-facing toasts
   - **custom_element** events that trigger the `WeatherCard` JSX component in the Chainlit UI
4. Implements `run_chat_completion_async` with `async_streaming_generator` for real-time token streaming.

When you invoke the pipeline from the Chainlit UI you will see:

- A progress step while the weather API is called.
- A notification confirming the location lookup.
- A **WeatherCard widget** showing temperature, humidity, wind speed, and weather conditions.
- The agent's friendly text summary streamed token by token.

## Folder structure

```text
pipeline_wrapper.py        # Pipeline implementation
README.md                  # You are reading it
```

The matching `WeatherCard.jsx` custom element lives in `src/hayhooks/server/chainlit_app/public/elements/WeatherCard.jsx` and is shipped with the `hayhooks[ui]` package.

## Prerequisites

- An `OPENAI_API_KEY` environment variable (the agent uses `gpt-4o-mini`)
- No weather API key required, Open-Meteo is free and unauthenticated

## Quick start

Create and activate a virtual environment (optional but recommended):

```shell
python -m venv .venv
source .venv/bin/activate
```

Install Hayhooks with the UI extra:

```shell
pip install "hayhooks[ui]"
```

Launch Hayhooks with the Chainlit UI enabled:

```shell
hayhooks run --with-ui
```

Deploy the pipeline wrapper (from the repository root):

```shell
hayhooks pipeline deploy-files -n weather_agent examples/pipeline_wrappers/chainlit_weather_agent
```

Open `http://localhost:1416/chat` in your browser, select the **weather_agent** pipeline, and ask something like:

> What's the weather in Berlin?

## How custom elements work

The pipeline emits a `custom_element` event via `create_custom_element_event("WeatherCard", props)`. The Chainlit app receives this event over the SSE stream and renders a `cl.CustomElement(name="WeatherCard", props=...)`, which Chainlit maps to the `WeatherCard.jsx` file in `public/elements/`.

This mechanism lets pipelines push rich, interactive UI widgets to the Chainlit frontend without modifying the Chainlit app itself.

## Open WebUI compatibility

This pipeline also works with Open WebUI. The `custom_element` events are silently ignored by Open WebUI, while the `status` and `notification` events render normally. The agent's text response streams as usual.
