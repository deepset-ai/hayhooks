from typing import Union, Optional, Dict, Any
from hayhooks.open_webui import (
    create_status_event,
    OpenWebUIEvent,
    create_details_tag,
)


def default_on_tool_call_start(
    tool_name: str, arguments: Optional[str], tool_call_id: Optional[str]
) -> Union[OpenWebUIEvent, None]:
    """
    Default callback when a tool call starts.

    Returns a status event to indicate that the tool is being called.
    If the tool name is not present, returns None.
    """
    if tool_name:
        return create_status_event(
            description=f"Calling '{tool_name}' tool...",
            done=False,
        )
    return None


def default_on_tool_call_end(tool_name: str, arguments: Dict[str, Any], result: str, error: bool) -> str:
    """
    Default callback when a tool call ends.

    Returns a detailed HTML block with the tool call's arguments and response.
    This is designed to be rendered by Open WebUI.
    """
    summary = f"Tool call result for '{tool_name}'"
    content = f"```\n" f"Arguments:\n" f"{arguments}\n" f"\nResponse:\n" f"{result}\n" "```"
    return create_details_tag(
        tool_name=tool_name,
        summary=summary,
        content=content,
    )
