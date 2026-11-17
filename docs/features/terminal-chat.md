# Chat with a deployed Agent

Use terminal chat to try a deployed Agent, follow its streamed response, and inspect its tool arguments and results.
Your PipelineWrapper owns the Agent, model, prompts, and tools. The terminal connects to its `/v1/chat/completions` endpoint.

Install the client and select a deployment:

```bash
pip install 'hayhooks[chat]'
hayhooks chat my-agent
```

The deployment must implement `run_chat_completion` or `run_chat_completion_async`.
Text-only chat wrappers work with the same command. Model credentials belong on the server.

## Connect to your development server

Use the existing Hayhooks settings to select the server:

```bash
HAYHOOKS_HOST=127.0.0.1 HAYHOOKS_PORT=1417 hayhooks chat my-agent
```

`HAYHOOKS_USE_HTTPS` and `HAYHOOKS_DISABLE_SSL` retain their CLI meanings.
Terminal chat currently targets development servers reachable through these settings.
The chat command does not configure authentication headers or proxy path prefixes.
For a protected deployment, use a client configured for that server's authentication requirements.

The header identifies the deployment and server. File tools operate on the server's filesystem.
Connecting does not grant the deployment access to your terminal's local directory.

## Send, inspect, and stop

Press **Enter** or **Send** to send. **Ctrl+J** inserts a new line.
**Shift+Enter** also inserts a new line when your terminal distinguishes it from Enter.
You can draft another message while a response arrives.

Tool cards show **Running**, **Success**, **Error**, or **Interrupted**.
Click a card or focus its title with Tab and press Enter to inspect arguments and results.
Errors expand automatically. Tool results remain literal text, and long results support keyboard scrolling.

Selecting transcript text, tool details, or composer text copies it automatically.
**Ctrl+C** copies an existing selection. Empty selections and streamed updates leave the clipboard unchanged.
Scroll up to read without being pulled back by new output. **Ctrl+End** returns to the latest output.
**Ctrl+T** switches between dark and light themes.

**Escape** or **Stop** closes the current HTTP stream and stops receiving its response.
Server work may still be running. Stopping does not undo completed actions.
Received text and tool results remain visible. A tool with no received result has an unknown outcome.
Check the affected file or external system before repeating a write.

**Ctrl+Q** stops reception and exits. **Ctrl+C** also quits when no text is selected.

## Recover from a failed response

A response is complete only after the server sends its completion marker.
A lost connection or unfinished stream leaves the response marked incomplete.
The terminal reports the known failure and a next action. It never automatically resends a request.
If the server provides no cause, the terminal reports that uncertainty.

A model output limit or Agent step limit also leaves the response incomplete.
Narrow the request or adjust the deployed model's output limit or the Agent's `max_agent_steps`.
Check the received tool results and affected files before repeating work.

A tool error does not necessarily fail the turn. The Agent can recover and complete its answer while the error card remains visible.

Conversation history lasts for the current process. Subsequent requests include completed user prompts and final assistant answers.
Tool cards and side messages are display content and are not sent as conversation history.
Failed and stopped turns stay visible but are omitted from subsequent requests.
Include the relevant context in your next message and ask the Agent to verify the current state before acting again.
There is no saved chat or playback command.

## Try workspace tools

The [terminal chat Agent example](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/terminal_chat_agent)
provides `list_files`, `read_file`, and `write_file` as ordinary Haystack tools.
One Agent selects the tools and writes the answer.

In the server terminal, choose a dedicated demo directory:

```bash
mkdir -p ./chat-workspace
printf 'A note for the Agent.\n' > ./chat-workspace/note.txt
export EXAMPLE_CHAT_WORKSPACE="$(pwd)/chat-workspace"
export OPENAI_API_KEY='your-openai-key'
hayhooks run
```

In another terminal, deploy the wrapper and connect:

```bash
hayhooks pipeline deploy-files examples/pipeline_wrappers/terminal_chat_agent --name workspace-agent
hayhooks chat workspace-agent
```

Ask the Agent to list its files, read `note.txt`, write a summary to `summary.txt`, and read the summary back.
Inspect the tool cards and the actual file in the server directory.
Set `EXAMPLE_AGENT_MODEL` on the server to select the model. The example defaults to `gpt-4.1-mini`.

Tools accept relative paths beneath `EXAMPLE_CHAT_WORKSPACE` and reject symlinks.
Reads and write input are limited to 64 KiB of UTF-8 bytes. Listings show at most 200 immediate entries and report truncation.
Writes replace a file's text and require an existing parent directory.
The example uses containment checks, not an OS sandbox. Use an operator-owned demo directory.
Concurrent writes can overwrite each other.

## Stream your configured Agent

Configure the Agent in your wrapper and adapt its stream with `stream_agent_chat`:

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from hayhooks import BasePipelineWrapper, stream_agent_chat

class PipelineWrapper(BasePipelineWrapper):
	def setup(self):
		self.pipeline = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"))

	async def run_chat_completion_async(self, model, messages, body):
		return stream_agent_chat(
			self.pipeline, [ChatMessage.from_openai_dict_format(message) for message in messages]
		)
```

Register tools or Toolsets when you construct the Agent.
Pass additional state inputs as keyword arguments to `stream_agent_chat`.
Run-specific `tools` retain Haystack's semantics. The helper owns `streaming_callback` and rejects a conflicting override.
It preserves the configured Agent and correlates streamed tool arguments and results by call ID.

## Emit tool cards from another wrapper

Terminal chat reads response text from OpenAI content deltas.
Yield `PipelineEvent` objects through the same stream to display tool activity:

```python
from hayhooks import PipelineEvent

yield PipelineEvent(type="hayhooks.tool_call", data={
	"call_id": "actual-tool-call-id", "actor": "Agent",
	"name": "list_files", "arguments": "{}",
})
yield PipelineEvent(type="hayhooks.tool_result", data={
	"call_id": "actual-tool-call-id", "result": '["notes.md"]', "error": False,
})
```

Use matching call IDs and complete strings for `arguments` and `result`.
Arguments containing valid JSON are formatted for reading. Results remain literal.
A side message can use `hayhooks.message` with `message_id`, `actor`, and `text` fields.
It is displayed separately and does not become conversation history.

Only emit tool content intended for the client. Tool output is not a redacted diagnostic channel.
Keep credentials and internal server diagnostics out of tool results.
The terminal does not execute client-side tool requests or display model reasoning.
