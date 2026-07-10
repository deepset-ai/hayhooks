"""Compatibility helpers for supporting both Haystack v2 and v3 in a single codebase."""

from typing import Any

import haystack
from haystack import Pipeline as HaystackPipeline

# Haystack v2 ships a separate AsyncPipeline class; v3 merged it into Pipeline (which gained
# run_async natively). Resolve it dynamically so a single code path supports both versions.
# Annotated as type[Any] because on v2 AsyncPipeline is NOT a Pipeline subclass (both derive
# from PipelineBase).
Pipeline: type[Any] = getattr(haystack, "AsyncPipeline", HaystackPipeline)
AsyncPipeline = Pipeline

__all__ = ["AsyncPipeline", "Pipeline"]
