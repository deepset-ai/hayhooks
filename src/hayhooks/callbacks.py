from typing import Any, Optional, Union

from hayhooks.open_webui import OpenWebUIEvent, create_details_tag, create_notification_event, create_status_event


def default_on_tool_call_start(
    tool_name: str,
    arguments: Optional[str],  # noqa: ARG001
    tool_call_id: Optional[str],  # noqa: ARG001
) -> Union[OpenWebUIEvent, None]:
    """
    Default callback function when a tool call starts.

    This callback creates a status event to indicate that a tool is being invoked.
    It provides real-time feedback to users about ongoing tool execution in the
    Open WebUI interface.

    Args:
        tool_name (str): The name of the tool being called. If empty or falsy,
            no event will be created.
        arguments (Optional[str]): The stringified arguments passed to the tool.
        tool_call_id (Optional[str]): A unique identifier for the tool call.

    Returns:
        Union[OpenWebUIEvent, None]: A status event object that can be rendered
            by Open WebUI to show tool execution progress. Returns None if the
            tool_name is empty or falsy.
    """
    if tool_name:
        return create_status_event(
            description=f"Calling '{tool_name}' tool...",
            done=False,
        )
    return None


def default_on_tool_call_end(
    tool_name: str, arguments: dict[str, Any], result: str, error: bool
) -> list[Union[OpenWebUIEvent, str]]:
    """
    Default callback function when a tool call ends.

    This callback creates appropriate events based on whether the tool call succeeded or failed.
    For successful calls, it generates a completion status event and a detailed summary.
    For failed calls, it creates an error notification.

    Args:
        tool_name (str): The name of the tool that was called.
        arguments (Dict[str, Any]): The arguments that were passed to the tool.
        result (str): The result or response from the tool execution.
        error (bool): Whether the tool call resulted in an error.

    Returns:
        List[Union[OpenWebUIEvent, str]]: A list of events to be processed by Open WebUI.
            For successful calls, returns a status event and a details tag with the tool's arguments and response.
            For failed calls, returns a hidden status event and an error notification.
            The list can contain both OpenWebUIEvent and str objects.
    """
    if error:
        return [
            create_status_event(
                description=f"Called '{tool_name}' tool",
                done=True,
                hidden=True,
            ),
            create_notification_event(
                content=f"Error calling '{tool_name}' tool",
                notification_type="error",
            ),
        ]

    return [
        create_status_event(
            description=f"Called '{tool_name}' tool",
            done=True,
        ),
        create_details_tag(
            tool_name=tool_name,
            summary=f"Tool call result for '{tool_name}'",
            content=(f"```\nArguments:\n{arguments}\n\nResponse:\n{result}\n```"),
        ),
    ]
