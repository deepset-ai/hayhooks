import os
from typing import Any, TypedDict

import chainlit as cl
import httpx

from hayhooks.server.chainlit_app import utils


class StreamState(TypedDict):
    """State maintained during streaming response."""

    current_step: cl.Step | None


# Configuration
DEFAULT_MODEL = os.getenv("HAYHOOKS_DEFAULT_MODEL", "")
MAX_PIPELINES_DISPLAY = 10

# Session keys
SESSION_HISTORY = "history"
SESSION_MODEL = "model"

# UI Messages
MSG_NO_PIPELINES = (
    "⚠️ No pipelines are currently deployed. Deploy a pipeline with `run_chat_completion` method to start chatting."
)
MSG_BACKEND_UNREACHABLE = "❌ Could not connect to Hayhooks backend at {url}. Please ensure the server is running."
MSG_SELECT_PIPELINE = "**Select a pipeline to chat with:**"
MSG_CONNECTED = "🚀 Connected to **{model}**. How can I help you?"
MSG_TIMEOUT = "❌ Request timed out. Please try again."
MSG_GENERIC_ERROR = "❌ An error occurred. Please try again or contact support."


async def send_message(content: str, actions: list[cl.Action] | None = None) -> None:
    """Helper to send a message with optional actions."""
    await cl.Message(content=content, actions=actions or []).send()


async def initialize_session() -> None:
    """Initialize user session with empty history."""
    cl.user_session.set(SESSION_HISTORY, [])
    cl.user_session.set(SESSION_MODEL, None)


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
    if not await utils.check_backend_health():
        await send_message(MSG_BACKEND_UNREACHABLE.format(url=utils.HAYHOOKS_BASE_URL))
        cl.logger.error(f"Backend unreachable at {utils.HAYHOOKS_BASE_URL}")
        return

    # Fetch available models
    models = await utils.get_available_models()

    if not models:
        await send_message(MSG_NO_PIPELINES)
        return

    # Try to auto-select a model
    selected_model = utils.select_model_automatically(models, DEFAULT_MODEL)

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


async def close_current_step(state: StreamState) -> None:
    """Close any open step in the state."""
    if current_step := state.get("current_step"):
        await current_step.__aexit__(None, None, None)
        state["current_step"] = None


async def handle_status_event(data: dict[str, Any], state: StreamState) -> None:
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

        step_name, step_type = utils.extract_tool_name(description)
        step = cl.Step(name=step_name, type=step_type, show_input=False)
        await step.__aenter__()
        step.output = f"⏳ {description}"
        state["current_step"] = step

    # Complete step
    elif current_step := state.get("current_step"):
        current_step.output = f"✅ {description}"
        await close_current_step(state)


async def handle_tool_result_event(data: dict[str, Any], state: StreamState) -> None:
    """Handle tool_result event (tool execution results)."""
    arguments = data.get("arguments", {})
    result = data.get("result", "")

    # Format tool result
    formatted_output = utils.format_tool_result(arguments, result)

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
        "error": "❌",
    }
    icon = icon_map.get(notification_type, "\u2139\ufe0f")
    await send_message(f"{icon} {content}")


async def handle_open_webui_event(event: dict[str, Any], state: StreamState) -> None:
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
        utils.EVENT_STATUS: lambda: handle_status_event(data, state),
        utils.EVENT_TOOL_RESULT: lambda: handle_tool_result_event(data, state),
        utils.EVENT_NOTIFICATION: lambda: handle_notification_event(data),
    }

    handler = handlers.get(event_type)
    if handler:
        await handler()
    else:
        cl.logger.debug(f"Unhandled event type: {event_type}")


async def process_stream_line(line: str, state: StreamState) -> str | None:
    """
    Process a single SSE line from the streaming response.

    Handles both OpenAI-format content chunks and Open WebUI events.

    Returns:
        Content delta if any, None otherwise.
    """
    return await utils.process_sse_chunk(
        line,
        event_handler=lambda event: handle_open_webui_event(event, state),
    )


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
    models = await utils.get_available_models()

    if not models:
        await send_message(MSG_NO_PIPELINES)
        return None

    await prompt_model_selection(models)
    return None


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
    stream_state: StreamState = {"current_step": None}
    full_response = ""

    async with (
        httpx.AsyncClient() as client,
        client.stream(
            "POST",
            f"{utils.HAYHOOKS_BASE_URL}/v1/chat/completions",
            json=utils.build_chat_request(model, history),
            timeout=utils.REQUEST_TIMEOUT,
        ) as response,
    ):
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
        cl.logger.error(f"HTTP error during chat completion: {e}")
        response_msg.content = f"❌ Server error ({e.response.status_code})"

    except httpx.TimeoutException:
        cl.logger.error("Request timed out")
        response_msg.content = MSG_TIMEOUT

    except Exception as e:
        cl.logger.error(f"Unexpected error during chat: {e}", exc_info=True)
        response_msg.content = MSG_GENERIC_ERROR

    finally:
        await response_msg.update()
