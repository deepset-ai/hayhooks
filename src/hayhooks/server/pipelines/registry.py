from typing import Any

from hayhooks.server.exceptions import PipelineNotFoundError
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


class _PipelineRegistry:
    """
    Registry for pipeline wrappers.

    All pipelines are stored as BasePipelineWrapper instances (or subclasses like YAMLPipelineWrapper).
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, BasePipelineWrapper] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add(
        self, name: str, pipeline_wrapper: BasePipelineWrapper, metadata: dict[str, Any] | None = None
    ) -> BasePipelineWrapper:
        """
        Add a pipeline wrapper to the registry.

        Args:
            name: Unique name for the pipeline.
            pipeline_wrapper: A BasePipelineWrapper instance (or subclass).
            metadata: Optional metadata to associate with the pipeline.

        Returns:
            The registered pipeline wrapper.

        Raises:
            ValueError: If a pipeline with the same name already exists.
            TypeError: If pipeline_wrapper is not a BasePipelineWrapper instance.
        """
        if metadata is None:
            metadata = {}

        if name in self._pipelines:
            msg = f"A pipeline with name {name} is already in the registry."
            raise ValueError(msg)

        if not isinstance(pipeline_wrapper, BasePipelineWrapper):
            msg = f"Expected BasePipelineWrapper instance, got {type(pipeline_wrapper).__name__}"
            raise TypeError(msg)

        self._pipelines[name] = pipeline_wrapper
        self._metadata[name] = metadata

        return pipeline_wrapper

    def remove(self, name: str) -> None:
        if name in self._pipelines:
            del self._pipelines[name]
            del self._metadata[name]

    def get(self, name: str) -> BasePipelineWrapper | None:
        return self._pipelines.get(name)

    def get_metadata(self, name: str) -> dict[str, Any] | None:
        return self._metadata.get(name)

    def update_metadata(self, name: str, metadata: dict[str, Any]) -> None:
        if name not in self._metadata:
            msg = f"Pipeline {name} not found in registry."
            raise PipelineNotFoundError(msg)
        self._metadata[name].update(metadata)

    def get_names(self) -> list[str]:
        return list(self._pipelines.keys())

    def clear(self) -> None:
        self._pipelines.clear()
        self._metadata.clear()


registry = _PipelineRegistry()
