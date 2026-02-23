import json
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

# Configuration - override via environment variables
HAYHOOKS_BASE_URL = os.getenv("HAYHOOKS_BASE_URL", "http://localhost:1416")
REQUEST_TIMEOUT = 120.0
MODELS_FETCH_TIMEOUT = 10.0
HEALTH_CHECK_TIMEOUT = 5.0

# HTTP Status Codes
HTTP_OK = 200

# Pattern for extracting tool names from status descriptions like "Calling 'weather_tool' tool..."
_TOOL_CALL_RE = re.compile(r"Calling '(.+?)' tool")

# Event types
EVENT_STATUS = "status"
EVENT_TOOL_RESULT = "tool_result"
EVENT_NOTIFICATION = "notification"


async def check_backend_health() -> bool:
    """
    Check if Hayhooks backend is reachable.

    Returns:
        True if backend is healthy, False otherwise.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HAYHOOKS_BASE_URL}/status", timeout=HEALTH_CHECK_TIMEOUT)
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
            response = await client.get(f"{HAYHOOKS_BASE_URL}/v1/models", timeout=MODELS_FETCH_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return []
            return data.get("data", [])
    except Exception:
        # Log errors are handled by caller
        return []


def select_model_automatically(models: list[dict[str, Any]], default_model: str = "") -> str | None:
    """
    Automatically select a model based on defaults or availability.

    Args:
        models: List of available models.
        default_model: Default model to prefer if available.

    Returns:
        Model ID if auto-selected, None if user needs to choose.
    """
    model_ids = [m.get("id") for m in models if m.get("id")]

    # Use default model if specified and available
    if default_model and default_model in model_ids:
        return default_model

    # If only one model, use it
    if len(model_ids) == 1:
        return model_ids[0]

    return None


def extract_tool_name(description: str) -> tuple[str, str]:
    """
    Extract tool name from status description.

    Args:
        description: Status description text.

    Returns:
        Tuple of (step_name, step_type)
    """
    if match := _TOOL_CALL_RE.search(description):
        return f"🔧 {match.group(1)}", "tool"
    return "Processing", "run"


def format_tool_result(arguments: dict[str, Any], result: str) -> str:
    """
    Format tool execution result for display.

    Args:
        arguments: Tool arguments.
        result: Tool execution result.

    Returns:
        Formatted markdown string.
    """
    args_str = json.dumps(arguments, indent=2) if isinstance(arguments, dict) else str(arguments)
    return f"**Arguments:**\n```json\n{args_str}\n```\n\n**Result:**\n```\n{result}\n```"


def build_chat_request(model: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build the chat completion request payload.

    Args:
        model: Model/pipeline name.
        history: Message history.

    Returns:
        Request payload dictionary.
    """
    return {
        "model": model,
        "messages": history,
        "stream": True,
    }


async def process_sse_chunk(
    line: str,
    event_handler: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> str | None:
    """
    Process a single SSE line from the streaming response.

    Handles both OpenAI-format content chunks and Open WebUI events.

    Args:
        line: SSE line to process.
        event_handler: Optional async callback to handle Open WebUI events.

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
        return None

    # Handle Open WebUI event
    if "event" in chunk:
        event = chunk.get("event", {})
        if "type" in event and "data" in event and event_handler:
            await event_handler(event)
        return None

    # Handle OpenAI-format chunk
    choices = chunk.get("choices", [])
    if not choices:
        return None

    delta = choices[0].get("delta") or {}
    content = delta.get("content") if isinstance(delta, dict) else None
    return content if content else None
