"""One authoritative classification of durable wrapper authoring modes."""

from __future__ import annotations

from enum import Enum

from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


class DurableAuthoringMode(str, Enum):
    """How a wrapper participates in the durable runtime."""

    NONE = "none"
    WRAPPER = "wrapper"
    MANAGED_AGENT = "managed_agent"


def durable_authoring_mode(wrapper: BasePipelineWrapper) -> DurableAuthoringMode:
    """Classify a wrapper once, with explicit wrapper methods taking precedence."""
    if getattr(wrapper, "_is_run_durable_implemented", False) or getattr(
        wrapper, "_is_run_durable_async_implemented", False
    ):
        return DurableAuthoringMode.WRAPPER
    if getattr(wrapper, "durable", False) and wrapper.pipeline is not None:
        return DurableAuthoringMode.MANAGED_AGENT
    return DurableAuthoringMode.NONE


__all__ = ["DurableAuthoringMode", "durable_authoring_mode"]
