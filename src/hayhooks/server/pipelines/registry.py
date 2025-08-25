from typing import Any, Optional, Union

from haystack import AsyncPipeline, DeserializationError, Pipeline
from haystack.core.errors import PipelineError

from hayhooks.server.exceptions import PipelineNotFoundError
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

PipelineType = Union[Pipeline, AsyncPipeline, BasePipelineWrapper]


class _PipelineRegistry:
    def __init__(self) -> None:
        self._pipelines: dict[str, PipelineType] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add(
        self, name: str, source_or_pipeline: Union[str, PipelineType], metadata: Optional[dict[str, Any]] = None
    ) -> PipelineType:
        if metadata is None:
            metadata = {}
        if name in self._pipelines:
            msg = f"A pipeline with name {name} is already in the registry."
            raise ValueError(msg)

        if isinstance(source_or_pipeline, (Pipeline, AsyncPipeline, BasePipelineWrapper)):
            pipeline = source_or_pipeline
        else:
            try:
                pipeline = Pipeline.loads(source_or_pipeline)
            except (PipelineError, DeserializationError) as e:
                msg = f"Unable to parse Haystack Pipeline {name}: {e}"
                raise ValueError(msg) from e

        self._pipelines[name] = pipeline
        self._metadata[name] = metadata

        return pipeline

    def remove(self, name: str) -> None:
        if name in self._pipelines:
            del self._pipelines[name]
            del self._metadata[name]

    def get(self, name: str) -> Union[PipelineType, None]:
        return self._pipelines.get(name)

    def get_metadata(self, name: str) -> Optional[dict[str, Any]]:
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
