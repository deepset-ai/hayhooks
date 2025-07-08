from typing import Union
from haystack.dataclasses import ToolCallDelta, ToolCallResult
from hayhooks.open_webui import (
    create_status_event,
    OpenWebUIEvent,
    create_details_tag,
)


def default_on_tool_call_start(tool_call: ToolCallDelta) -> Union[OpenWebUIEvent, None]:
    """
    Default callback when a tool call starts.

    Returns a status event to indicate that the tool is being called.
    If the tool name is not present, returns None.
    """
    if tool_call.tool_name:
        return create_status_event(
            description=f"Calling '{tool_call.tool_name}' tool...",
            done=False,
        )
    return None


def default_on_tool_call_end(tool_call: ToolCallResult) -> str:
    """
    Default callback when a tool call ends.

    Returns a detailed HTML block with the tool call's arguments and response.
    This is designed to be rendered by Open WebUI.
    """
    summary = f"Tool call result for '{tool_call.origin.tool_name}'"
    content = (
        f"```\n"
        f"Arguments:\n"
        f"{tool_call.origin.arguments}\n"
        f"\nResponse:\n"
        f"{tool_call.result}\n"
        "```"
    )
    return create_details_tag(
        tool_name=tool_call.origin.tool_name,
        summary=summary,
        content=content,
    )
