"""Compatibility import path for durable Haystack adapters."""

from hayhooks.durable.adapters import (
    HaystackDurableAdapter,
    _checkpoint_agent_state,
    _checkpoint_data,
    _restore_agent_state,
    definition_revision,
    execution_kind,
    require_haystack_v3,
)

__all__ = [
    "HaystackDurableAdapter",
    "_checkpoint_agent_state",
    "_checkpoint_data",
    "_restore_agent_state",
    "definition_revision",
    "execution_kind",
    "require_haystack_v3",
]
