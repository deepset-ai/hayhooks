import json
from uuid import uuid4

from haystack.dataclasses import ChatMessage, ToolCall

from hayhooks import log


def _content_to_text(content: object) -> str:
    """Flatten a Responses API content field into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)


def _input_item_to_text(item: dict) -> str:
    return _content_to_text(item.get("content"))


def last_user_input_text(input_items: list[dict]) -> str | None:
    """Return last user text from Responses input items."""
    for item in reversed(input_items):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        if item.get("type") not in ("message", None):
            continue
        text = _input_item_to_text(item).strip()
        if text:
            return text
    return None


def is_tool_followup_request(input_items: list[dict]) -> bool:
    """Return True when the latest meaningful input item is a tool follow-up."""
    for item in reversed(input_items):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("function_call", "function_call_output"):
            return True
        if item_type in ("message", None):
            return False
    return False


def _parse_tool_arguments(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_call_id(item: dict, *, generate_if_missing: bool) -> str | None:
    for key in ("call_id", "id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    if generate_if_missing:
        return f"call_{uuid4().hex[:24]}"
    return None


def _tool_name_from_item(item: dict) -> str:
    value = item.get("name")
    if isinstance(value, str) and value:
        return value
    return "function"


def _serialize_tool_output(output: object) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output)
    except TypeError:
        return str(output)


def input_items_to_chat_messages(input_items: list[dict]) -> list[ChatMessage]:
    """Convert Responses API input items to Haystack ChatMessage objects."""
    messages: list[ChatMessage] = []
    tool_calls_by_id: dict[str, ToolCall] = {}

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "function_call":
            arguments = _parse_tool_arguments(item.get("arguments", "{}"))
            call_id = _extract_call_id(item, generate_if_missing=True)
            tool_name = _tool_name_from_item(item)
            log.opt(colors=True).debug("<blue>[client-tool]</blue> History item: call '{}' (call_id={})", tool_name, call_id)

            tool_call = ToolCall(tool_name=tool_name, arguments=arguments, id=call_id)
            tool_calls_by_id[call_id] = tool_call
            messages.append(ChatMessage.from_assistant(tool_calls=[tool_call]))
            continue

        if item_type == "function_call_output":
            call_id = _extract_call_id(item, generate_if_missing=False)
            output = _serialize_tool_output(item.get("output", ""))
            log.opt(colors=True).debug("<blue>[client-tool]</blue> History item: output (call_id={})", call_id or "unknown")

            if isinstance(call_id, str) and call_id in tool_calls_by_id:
                origin = tool_calls_by_id[call_id]
            else:
                origin = ToolCall(tool_name="function", arguments={}, id=str(call_id) if call_id else None)
                log.opt(colors=True).debug("<blue>[client-tool]</blue> Unknown call_id '{}'; using fallback origin", call_id)

            messages.append(ChatMessage.from_tool(tool_result=output, origin=origin))
            continue

        role = item.get("role")
        text = _input_item_to_text(item)
        if not text:
            continue
        if role == "user":
            messages.append(ChatMessage.from_user(text))
        elif role in ("system", "developer"):
            messages.append(ChatMessage.from_system(text))
        elif role == "assistant":
            messages.append(ChatMessage.from_assistant(text))

    return messages
