"""
Default Chainlit app for Hayhooks.

This app provides a chat interface that communicates with Hayhooks'
OpenAI-compatible endpoints, allowing users to interact with deployed
Haystack pipelines through a conversational UI.

Features:
- Streaming responses with real-time updates
- Tool call visualization via cl.Step
- Session-based chat history
- Open WebUI event handling (status, tool_result, notification)
- Automatic pipeline discovery and selection
- Health checks and robust error handling
"""

import json
import os
from typing import Any

import chainlit as cl
import httpx

# Configuration - override via environment variables
HAYHOOKS_BASE_URL = os.getenv("HAYHOOKS_BASE_URL", "http://localhost:1416")
DEFAULT_MODEL = os.getenv("HAYHOOKS_DEFAULT_MODEL", "")
REQUEST_TIMEOUT = 120.0
MODELS_FETCH_TIMEOUT = 10.0
HEALTH_CHECK_TIMEOUT = 5.0
MAX_PIPELINES_DISPLAY = 10

# HTTP Status Codes
HTTP_OK = 200

# Constants for tool name extraction
TOOL_CALL_PREFIX = "Calling '"
TOOL_CALL_SUFFIX = "' tool"
TOOL_CALL_PREFIX_LEN = len(TOOL_CALL_PREFIX) - 1

# Session keys
SESSION_HISTORY = "history"
SESSION_MODEL = "model"

# Event types
EVENT_STATUS = "status"
EVENT_TOOL_RESULT = "tool_result"
EVENT_NOTIFICATION = "notification"

# UI Messages
MSG_NO_PIPELINES = (
    "⚠️ No pipelines are currently deployed. "
    "Deploy a pipeline with `run_chat_completion` method to start chatting."
)
MSG_BACKEND_UNREACHABLE = (
    "❌ Could not connect to Hayhooks backend at {url}. "
    "Please ensure the server is running."
)
MSG_SELECT_PIPELINE = "**Select a pipeline to chat with:**"
MSG_CONNECTED = "🚀 Connected to **{model}**. How can I help you?"
MSG_TIMEOUT = "❌ Request timed out. Please try again."
MSG_GENERIC_ERROR = "❌ An error occurred. Please try again or contact support."


async def check_backend_health() -> bool:
    """Check if Hayhooks backend is reachable."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{HAYHOOKS_BASE_URL}/status",
                timeout=HEALTH_CHECK_TIMEOUT
            )
            return response.status_code == HTTP_OK
    except Exception:
        return False


async def get_available_models() -> list[dict[str, Any]]:
    """
    Fetch available models (deployed pipelines) from Hayhooks.

    Returns:
        List of model dictionaries with 'id' and metadata.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{HAYHOOKS_BASE_URL}/v1/models",
                timeout=MODELS_FETCH_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except httpx.HTTPStatusError as e:
        cl.logger.error(f"HTTP error fetching models: {e.response.status_code}")
    except httpx.RequestError as e:
        cl.logger.error(f"Request error fetching models: {e}")
    except Exception as e:
        cl.logger.error(f"Unexpected error fetching models: {e}")
    return []


async def send_message(content: str, actions: list[cl.Action] | None = None) -> None:
    """Helper to send a message with optional actions."""
    await cl.Message(content=content, actions=actions or []).send()


async def initialize_session() -> None:
    """Initialize user session with empty history."""
    cl.user_session.set(SESSION_HISTORY, [])
    cl.user_session.set(SESSION_MODEL, None)


async def select_model_automatically(models: list[dict[str, Any]]) -> str | None:
    """
    Automatically select a model based on defaults or availability.

    Returns:
        Model ID if auto-selected, None if user needs to choose.
    """
    model_ids = [m["id"] for m in models]

    # Use default model if specified and available
    if DEFAULT_MODEL and DEFAULT_MODEL in model_ids:
        return DEFAULT_MODEL

    # If only one model, use it
    if len(models) == 1:
        return models[0]["id"]

    return None


async def prompt_model_selection(models: list[dict[str, Any]]) -> None:
    """Show model selection UI to user."""
    actions = [
        cl.Action(name="select_model", payload={"model": m["id"]}, label=m["id"])
        for m in models[:MAX_PIPELINES_DISPLAY]
    ]
    await send_message(MSG_SELECT_PIPELINE, actions)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the chat session with health checks and model selection."""
    await initialize_session()

    # Check backend health
    if not await check_backend_health():
        await send_message(MSG_BACKEND_UNREACHABLE.format(url=HAYHOOKS_BASE_URL))
        cl.logger.error(f"Backend unreachable at {HAYHOOKS_BASE_URL}")
        return

    # Fetch available models
    models = await get_available_models()

    if not models:
        await send_message(MSG_NO_PIPELINES)
        return

    # Try to auto-select a model
    selected_model = await select_model_automatically(models)

    if selected_model:
        cl.user_session.set(SESSION_MODEL, selected_model)
        await send_message(MSG_CONNECTED.format(model=selected_model))
    else:
        # Multiple models available - let user choose
        await prompt_model_selection(models)


@cl.action_callback("select_model")
async def on_model_select(action: cl.Action) -> None:
    """Handle model selection and reset session."""
    model = action.payload.get("model")
    if not model:
        cl.logger.error("Model selection action missing model payload")
        return

    cl.user_session.set(SESSION_MODEL, model)
    cl.user_session.set(SESSION_HISTORY, [])  # Reset history on model switch
    await send_message(MSG_CONNECTED.format(model=model))


async def close_current_step(state: dict[str, Any]) -> None:
    """Close any open step in the state."""
    if current_step := state.get("current_step"):
        await current_step.__aexit__(None, None, None)
        state["current_step"] = None


def extract_tool_name(description: str) -> tuple[str, str]:
    """
    Extract tool name from status description.

    Returns:
        Tuple of (step_name, step_type)
    """
    if TOOL_CALL_PREFIX in description and TOOL_CALL_SUFFIX in description:
        start = description.find(TOOL_CALL_PREFIX) + len(TOOL_CALL_PREFIX)
        end = description.find(TOOL_CALL_SUFFIX)
        if start > TOOL_CALL_PREFIX_LEN and end > start:
            tool_name = description[start:end]
            return f"🔧 {tool_name}", "tool"
    return "Processing", "run"


async def handle_status_event(data: dict[str, Any], state: dict[str, Any]) -> None:
    """Handle status event (progress updates)."""
    description = data.get("description", "Processing...")
    done = data.get("done", False)
    hidden = data.get("hidden", False)

    # Handle hidden events
    if hidden:
        if done:
            await close_current_step(state)
        return

    # Start new step
    if not done:
        await close_current_step(state)

        step_name, step_type = extract_tool_name(description)
        step = cl.Step(name=step_name, type=step_type, show_input=False)
        await step.__aenter__()
        step.output = f"⏳ {description}"
        state["current_step"] = step

    # Complete step
    elif current_step := state.get("current_step"):
        current_step.output = f"✅ {description}"
        await close_current_step(state)


async def handle_tool_result_event(data: dict[str, Any], state: dict[str, Any]) -> None:
    """Handle tool_result event (tool execution results)."""
    arguments = data.get("arguments", {})
    result = data.get("result", "")

    # Format arguments as JSON
    args_str = json.dumps(arguments, indent=2) if isinstance(arguments, dict) else str(arguments)
    formatted_output = (
        f"**Arguments:**\n```json\n{args_str}\n```\n\n"
        f"**Result:**\n```\n{result}\n```"
    )

    # Close current step with formatted result
    if current_step := state.get("current_step"):
        current_step.output = formatted_output
        await close_current_step(state)


async def handle_notification_event(data: dict[str, Any]) -> None:
    """Handle notification event (user notifications)."""
    notification_type = data.get("type", "info")
    content = data.get("content", "")

    icon_map = {
        "info": "\u2139\ufe0f",  # Information symbol
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    icon = icon_map.get(notification_type, "\u2139\ufe0f")
    await send_message(f"{icon} {content}")


async def handle_open_webui_event(event: dict[str, Any], state: dict[str, Any]) -> None:
    """
    Handle Open WebUI events from the streaming response.

    Converts Open WebUI events to Chainlit UI elements.

    Event types:
    - status: Progress updates (shown as Steps with spinner)
    - tool_result: Tool call results (shown in code block)
    - notification: Toast notifications
    """
    event_type = event.get("type", "")
    data = event.get("data", {})

    handlers = {
        EVENT_STATUS: lambda: handle_status_event(data, state),
        EVENT_TOOL_RESULT: lambda: handle_tool_result_event(data, state),
        EVENT_NOTIFICATION: lambda: handle_notification_event(data),
    }

    handler = handlers.get(event_type)
    if handler:
        await handler()
    else:
        cl.logger.debug(f"Unhandled event type: {event_type}")


async def process_stream_line(line: str, state: dict[str, Any]) -> str | None:
    """
    Process a single SSE line from the streaming response.

    Handles both OpenAI-format content chunks and Open WebUI events.

    Returns:
        Content delta if any, None otherwise.
    """
    if not line.startswith("data: "):
        return None

    data = line[6:]
    if data == "[DONE]":
        return None

    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        cl.logger.debug(f"Failed to parse SSE chunk: {data}")
        return None

    # Handle Open WebUI event
    if "event" in chunk:
        event = chunk.get("event", {})
        if "type" in event and "data" in event:
            await handle_open_webui_event(event, state)
        return None

    # Handle OpenAI-format chunk
    choices = chunk.get("choices", [])
    if not choices:
        return None

    delta = choices[0].get("delta", {})
    return delta.get("content", "")


async def ensure_model_selected() -> str | None:
    """
    Ensure a model is selected. Prompt user if not.

    Returns:
        Model ID if selected, None otherwise.
    """
    model = cl.user_session.get(SESSION_MODEL)

    if model:
        return model

    # No model selected - prompt user
    models = await get_available_models()

    if not models:
        await send_message(MSG_NO_PIPELINES)
        return None

    await prompt_model_selection(models)
    return None


def build_chat_request(model: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the chat completion request payload."""
    return {
        "model": model,
        "messages": history,
        "stream": True,
    }


async def stream_chat_completion(
    model: str,
    history: list[dict[str, Any]],
    response_msg: cl.Message,
) -> str:
    """
    Stream chat completion from Hayhooks backend.

    Returns:
        Full response text.

    Raises:
        httpx.HTTPStatusError: If response status is not 200.
        httpx.TimeoutException: If request times out.
    """
    stream_state: dict[str, Any] = {"current_step": None}
    full_response = ""

    async with httpx.AsyncClient() as client, client.stream(
        "POST",
        f"{HAYHOOKS_BASE_URL}/v1/chat/completions",
        json=build_chat_request(model, history),
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            content = await process_stream_line(line, stream_state)
            if content:
                full_response += content
                await response_msg.stream_token(content)

        # Cleanup any open steps
        await close_current_step(stream_state)

    return full_response


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming user messages with streaming response."""
    # Ensure model is selected
    model = await ensure_model_selected()
    if not model:
        return

    # Build message history
    history = cl.user_session.get(SESSION_HISTORY, [])
    history.append({"role": "user", "content": message.content})

    # Create response message for streaming
    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        # Stream response from backend
        full_response = await stream_chat_completion(model, history, response_msg)

        # Update history with assistant response
        if full_response:
            history.append({"role": "assistant", "content": full_response})
            cl.user_session.set(SESSION_HISTORY, history)

    except httpx.HTTPStatusError as e:
        error_msg = f"Server error ({e.response.status_code})"
        cl.logger.error(f"HTTP error during chat completion: {e}")
        await response_msg.stream_token(f"❌ {error_msg}")

    except httpx.TimeoutException:
        cl.logger.error("Request timed out")
        await response_msg.stream_token(MSG_TIMEOUT)

    except Exception as e:
        cl.logger.error(f"Unexpected error during chat: {e}", exc_info=True)
        await response_msg.stream_token(MSG_GENERIC_ERROR)

    finally:
        await response_msg.update()
