# Open WebUI Agent Events Example

This example demonstrates how to enhance the chat experience in [Open WebUI](https://github.com/open-webui/open-webui) by emitting **events** from a Haystack `Agent` pipeline executed through **Hayhooks**.

The heart of the example is `pipeline_wrapper.py`, which

1. Instantiates a Haystack `Agent` backed by an `OpenAIChatGenerator`.
2. Implements `run_chat_completion_async` that yields an **async generator** mixing
   streaming chunks with Open WebUI event objects.
3. Sends two `status` events – one when the pipeline starts and another when it
   finishes.
4. Emits an HTML-like `<details>` block with extra information once the run is
   complete.

When you invoke the pipeline from Open WebUI you will see:

* A "Running the pipeline!" banner while the agent thinks.
* The answer streamed token by token.
* A "Pipeline completed!" success banner.
* An expandable section labelled *Pipeline completed!* containing additional
  details.

## Folder structure

```text
pipeline_wrapper.py        # Pipeline implementation
README.md                  # You are reading it
```

No additional dependencies are required besides `hayhooks` (which already pulls
Haystack).

## Quick start

Create and activate a virtual environment (optional but recommended):

```shell
python -m venv .venv
source .venv/bin/activate
```

Install Hayhooks:

```shell
pip install hayhooks
```

Launch Hayhooks:

```shell
hayhooks run
# The server will listen on http://localhost:1416
```

Deploy the pipeline wrapper:

```shell
hayhooks pipeline deploy-files -n agent_events .
```

Open *Open WebUI* → *Settings* → *Connections* and add a **Remote Hayhooks
server** pointing to `http://localhost:1416` (no API key is needed).

Start a new chat, select the *agent_events* pipeline and ask any question.

## Supported Open WebUI events

Hayhooks exposes convenience helpers in `hayhooks.open_webui` to build all event
kinds supported by Open WebUI. You can yield these objects (or plain strings)
from your pipeline generator to control the UI in real time.

| Helper function | `type` value | Purpose / UI effect |
|-----------------|--------------|------------------------------------------------|
| `create_status_event` | `status` | Show progress indicator (`description`, `done`, `hidden`). |
| `create_chat_completion_event` | `chat:completion` | Send a full chat-completion payload (usually not needed manually). |
| `create_message_event` | `message` | Append content to the current assistant message. |
| `create_replace_event` | `replace` | Replace the entire assistant message content. |
| `create_source_event` | `source` | Attach citations, code results or other rich blocks. |
| `create_notification_event` | `notification` | Display toast notifications (`info`, `success`, `warning`, `error`). |
| `create_details_tag`* | *(string)* | Render an expandable `<details>` block (not a real event). |

\* `create_details_tag` returns raw markup – Open WebUI recognises it and adds a
collapsible section to the chat.

For a deeper explanation of every event see the official
[Open WebUI event documentation](https://docs.openwebui.com/features/plugin/events).

---

Feel free to extend the wrapper and experiment with other event types!
