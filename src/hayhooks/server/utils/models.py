from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


@dataclass(slots=True)
class PreparedPipeline:
    """Result of a prepare-only deploy step (no registry/route mutations yet)."""

    name: str
    wrapper: BasePipelineWrapper
    extra_metadata: dict[str, Any] | None = field(default=None)
