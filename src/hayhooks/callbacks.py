from typing import Any

from haystack.tracing.utils import coerce_tag_value

from hayhooks.events import PipelineEvent, create_notification_event, create_status_event
from hayhooks.open_webui import create_details_tag


def default_on_tool_call_start(
    tool_name: str,
    arguments: str | None,  # noqa: ARG001
    tool_call_id: str | None,  # noqa: ARG001
) -> PipelineEvent | None:
    """
    Default callback function when a tool call starts.

    This callback creates a status event to indicate that a tool is being invoked.
    It provides real-time feedback to users about ongoing tool execution in the UI.

    Args:
        tool_name (str): The name of the tool being called. If empty or falsy,
            no event will be created.
        arguments (str | None): The stringified arguments passed to the tool.
        tool_call_id (str | None): A unique identifier for the tool call.

    Returns:
        PipelineEvent | None: A status event object that can be rendered
            by the UI to show tool execution progress. Returns None if the
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
) -> list[PipelineEvent | str]:
    """
    Default callback function when a tool call ends.

    This callback creates appropriate events based on whether the tool call succeeded or failed.
    For successful calls, it generates a completion status event and a detailed summary.
    For failed calls, it creates an error notification.

    Args:
        tool_name (str): The name of the tool that was called.
        arguments (dict[str, Any]): The arguments that were passed to the tool.
        result (str): The result or response from the tool execution.
        error (bool): Whether the tool call resulted in an error.

    Returns:
        list[PipelineEvent | str]: A list of events to be processed by the UI.
            For successful calls, returns a status event and a details tag with the tool's arguments and response.
            For failed calls, returns a hidden status event and an error notification.
            The list can contain both PipelineEvent and str objects.
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


def default_on_pipeline_end(result: Any) -> str:
    """
    Default callback function when a pipeline run ends.

    This callback coerces the pipeline result to a string format suitable for display
    in the Open WebUI interface.

    Args:
        result (Any): The result produced by the pipeline run.

    Returns:
        str: The coerced string representation of the pipeline result.
    """
    return str(coerce_tag_value(result))
