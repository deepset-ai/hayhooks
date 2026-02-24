from typing import Any

from hayhooks.events import PipelineEvent


class ChainlitEvent(PipelineEvent):
    """
    Event specific to the embedded Chainlit UI.

    These events are ignored by Open WebUI and only processed by the
    Chainlit frontend mounted inside Hayhooks.
    """

    data: dict[str, Any]


def create_custom_element_event(name: str, props: dict[str, Any]) -> ChainlitEvent:
    """
    Create a custom element event for rendering rich UI widgets in the Chainlit frontend.

    This event is specific to the embedded Chainlit UI and will be ignored by Open WebUI.
    The Chainlit app renders the element using a matching JSX file in public/elements/.

    Args:
        name: Name of the custom element (must match a JSX file, e.g. "WeatherCard").
        props: Dictionary of props to pass to the JSX component.
    """
    return ChainlitEvent(type="custom_element", data={"name": name, "props": props})
