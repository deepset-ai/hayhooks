"""
Default Chainlit app for Hayhooks.

This app provides a chat interface that communicates with Hayhooks'
OpenAI-compatible endpoints, allowing users to interact with deployed
Haystack pipelines through a conversational UI.

Features:
- Streaming responses
- Tool call visualization via cl.Step
- Session-based chat history (persists during browser session)
"""

import json
import os

import chainlit as cl
import httpx

# Configuration - these can be overridden via environment variables
HAYHOOKS_BASE_URL = os.getenv("HAYHOOKS_BASE_URL", "http://localhost:1416")
DEFAULT_MODEL = os.getenv("HAYHOOKS_DEFAULT_MODEL", "")


async def get_available_models() -> list[dict]:
    """Fetch available models (deployed pipelines) from Hayhooks."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HAYHOOKS_BASE_URL}/v1/models", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
    except Exception as e:
        cl.logger.warning(f"Failed to fetch models: {e}")
    return []


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Initialize empty history
    cl.user_session.set("history", [])

    # Fetch available models
    models = await get_available_models()

    if not models:
        await cl.Message(
            content="⚠️ No pipelines are currently deployed. "
            "Deploy a pipeline with `run_chat_completion` method to start chatting."
        ).send()
        return

    # If there's a default model set and it exists, use it
    model_ids = [m["id"] for m in models]
    if DEFAULT_MODEL and DEFAULT_MODEL in model_ids:
        cl.user_session.set("model", DEFAULT_MODEL)
        await cl.Message(content=f"🚀 Connected to **{DEFAULT_MODEL}**. How can I help you?").send()
    elif len(models) == 1:
        # Only one model available, use it automatically
        cl.user_session.set("model", models[0]["id"])
        await cl.Message(content=f"🚀 Connected to **{models[0]['id']}**. How can I help you?").send()
    else:
        # Multiple models - let user choose (limit to 10)
        actions = [cl.Action(name="select_model", payload={"model": m["id"]}, label=m["id"]) for m in models[:10]]
        await cl.Message(content="**Select a pipeline to chat with:**", actions=actions).send()


@cl.action_callback("select_model")
async def on_model_select(action: cl.Action):
    """Handle model selection."""
    model = action.payload["model"]
    cl.user_session.set("model", model)
    cl.user_session.set("history", [])  # Reset history on model switch
    await cl.Message(content=f"🚀 Connected to **{model}**. How can I help you?").send()


async def handle_open_webui_event(event: dict, state: dict) -> None:
    """
    Handle Open WebUI events from the streaming response.

    Converts Open WebUI events to Chainlit UI elements.

    Event types:
    - status: Progress updates (shown as Steps with spinner)
    - tool_result: Tool call results (shown in code block)
    - message: Append content
    - notification: Toast notifications
    - source: Citations/references
    """
    event_type = event.get("type", "")
    data = event.get("data", {})

    if event_type == "status":
        description = data.get("description", "Processing...")
        done = data.get("done", False)
        hidden = data.get("hidden", False)

        # Skip hidden events
        if hidden:
            # Just close any open step
            if done and (current_step := state.get("current_step")):
                await current_step.__aexit__(None, None, None)
                state["current_step"] = None
            return

        if not done:
            # Close previous step if exists
            if current_step := state.get("current_step"):
                await current_step.__aexit__(None, None, None)

            # Extract tool name from description if present (e.g., "Calling 'weather_tool' tool...")
            step_name = "Processing"
            step_type = "run"
            if "Calling '" in description and "' tool" in description:
                # Extract tool name
                start = description.find("Calling '") + 9
                end = description.find("' tool")
                if start > 8 and end > start:
                    step_name = f"🔧 {description[start:end]}"
                    step_type = "tool"

            # Create a new step with spinner (spinner shows while step is open)
            step = cl.Step(name=step_name, type=step_type, show_input=False)
            await step.__aenter__()
            step.output = f"⏳ {description}"
            state["current_step"] = step
            state["step_name"] = step_name

        elif current_step := state.get("current_step"):
            # Finish the step with success indicator
            current_step.output = f"✅ {description}"
            await current_step.__aexit__(None, None, None)
            state["current_step"] = None

    elif event_type == "tool_result":
        # Tool result with arguments and response - display in code block
        tool_name = data.get("tool", "unknown")
        arguments = data.get("arguments", {})
        result = data.get("result", "")

        # Format as code block
        args_str = json.dumps(arguments, indent=2) if isinstance(arguments, dict) else str(arguments)
        formatted_output = f"**Arguments:**\n```json\n{args_str}\n```\n\n**Result:**\n```\n{result}\n```"

        # Close the current step with the formatted result
        if current_step := state.get("current_step"):
            current_step.output = formatted_output
            await current_step.__aexit__(None, None, None)
            state["current_step"] = None

    elif event_type == "notification":
        notification_type = data.get("type", "info")
        content = data.get("content", "")
        # Map to visual indicators
        icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(notification_type, "")
        await cl.Message(content=f"{icon} {content}").send()


async def process_stream_line(line: str, state: dict) -> str | None:
    """
    Process a single SSE line from the streaming response.

    Handles both OpenAI-format content chunks and Open WebUI events.

    Returns the content delta if any.
    """
    if not line.startswith("data: "):
        return None

    data = line[6:]
    if data == "[DONE]":
        return None

    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return None

    # Check if this is an Open WebUI event (wrapped in "event" key)
    if "event" in chunk:
        event = chunk["event"]
        if "type" in event and "data" in event:
            await handle_open_webui_event(event, state)
        return None

    # Otherwise, it's an OpenAI-format chunk
    choices = chunk.get("choices", [])
    if not choices:
        return None

    delta = choices[0].get("delta", {})
    return delta.get("content", "")


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    model = cl.user_session.get("model")

    if not model:
        # No model selected - prompt user to select one
        models = await get_available_models()
        if not models:
            await cl.Message(content="⚠️ No pipelines are deployed. Please deploy a pipeline first.").send()
            return

        actions = [cl.Action(name="select_model", payload={"model": m["id"]}, label=m["id"]) for m in models[:10]]
        await cl.Message(content="**Please select a pipeline first:**", actions=actions).send()
        return

    # Get message history for context
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    # Create response message for streaming
    response_msg = cl.Message(content="")
    await response_msg.send()

    # State for tracking Open WebUI events (e.g., status steps)
    stream_state: dict = {"current_step": None}

    try:
        # Call Hayhooks OpenAI-compatible endpoint with streaming
        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                f"{HAYHOOKS_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": history,
                    "stream": True,
                },
                timeout=120.0,
            ) as response,
        ):
            if response.status_code != 200:
                error_text = await response.aread()
                await response_msg.stream_token(f"❌ Error: {error_text.decode()}")
                await response_msg.update()
                return

            full_response = ""
            async for line in response.aiter_lines():
                content = await process_stream_line(line, stream_state)
                if content:
                    full_response += content
                    await response_msg.stream_token(content)

            # Clean up any open steps
            if current_step := stream_state.get("current_step"):
                await current_step.__aexit__(None, None, None)

            # Update history with assistant response
            if full_response:
                history.append({"role": "assistant", "content": full_response})
                cl.user_session.set("history", history)

    except httpx.TimeoutException:
        await response_msg.stream_token("❌ Request timed out. Please try again.")
    except Exception as e:
        await response_msg.stream_token(f"❌ Error: {e!s}")

    await response_msg.update()


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """
    Resume a previous chat session.

    This is called when a user returns to a persisted chat thread.
    Requires data persistence to be configured.
    """
    # Restore history from thread messages
    history = []
    for message in thread.get("steps", []):
        msg_type = message.get("type")
        content = message.get("output", "")

        if msg_type == "user_message":
            history.append({"role": "user", "content": content})
        elif msg_type == "assistant_message":
            history.append({"role": "assistant", "content": content})

    cl.user_session.set("history", history)

    # Try to restore model selection
    metadata = thread.get("metadata", {})
    if model := metadata.get("model"):
        cl.user_session.set("model", model)


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Handle settings updates."""
    if "model" in settings:
        cl.user_session.set("model", settings["model"])
        await cl.Message(content=f"🔄 Switched to **{settings['model']}**").send()
