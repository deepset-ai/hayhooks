from typing import Any, Dict, Optional, Union
from haystack import Pipeline
from haystack.core.errors import PipelineError
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

PipelineType = Union[Pipeline, BasePipelineWrapper]


class _PipelineRegistry:
    def __init__(self) -> None:
        self._pipelines: Dict[str, PipelineType] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add(
        self, name: str, source_or_pipeline: Union[str, PipelineType], metadata: Dict[str, Any] = {}
    ) -> PipelineType:
        if name in self._pipelines:
            msg = f"A pipeline with name {name} is already in the registry."
            raise ValueError(msg)

        if isinstance(source_or_pipeline, (Pipeline, BasePipelineWrapper)):
            pipeline = source_or_pipeline
        else:
            try:
                pipeline = Pipeline.loads(source_or_pipeline)
            except PipelineError as e:
                msg = f"Unable to parse Haystack Pipeline {name}: {e}"
                raise ValueError(msg) from e

        self._pipelines[name] = pipeline
        self._metadata[name] = metadata

        return pipeline

    def remove(self, name: str):
        if name in self._pipelines:
            del self._pipelines[name]
            del self._metadata[name]

    def get(self, name: str, with_metadata: bool = False) -> Optional[PipelineType]:
        if with_metadata:
            return self._pipelines.get(name), self._metadata.get(name)
        else:
            return self._pipelines.get(name)

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        return self._metadata.get(name)

    def get_names(self) -> list[str]:
        return list(self._pipelines.keys())

    def clear(self):
        self._pipelines.clear()
        self._metadata.clear()


registry = _PipelineRegistry()
