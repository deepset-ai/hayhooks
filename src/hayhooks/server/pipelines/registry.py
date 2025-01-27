from typing import Optional, Union

from haystack import Pipeline
from haystack.core.errors import PipelineError
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper

PipelineType = Union[Pipeline, BasePipelineWrapper]


class _PipelineRegistry:
    def __init__(self) -> None:
        self._pipelines: dict[str, PipelineType] = {}

    def add(self, name: str, source_or_pipeline: Union[str, PipelineType]) -> PipelineType:
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
        return pipeline

    def remove(self, name: str):
        if name in self._pipelines:
            del self._pipelines[name]

    def get(self, name: str) -> Optional[PipelineType]:
        return self._pipelines.get(name)

    def get_names(self) -> list[str]:
        return list(self._pipelines.keys())

    def clear(self):
        self._pipelines.clear()


registry = _PipelineRegistry()
