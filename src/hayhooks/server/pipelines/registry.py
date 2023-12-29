from typing import Any

from haystack.pipeline import Pipeline, Optional
from haystack.core.errors import PipelineError


class _PipelineRegistry:
    def __init__(self) -> None:
        self._pipelines: dict[str, Pipeline] = {}

    def add(self, name: str, source: str) -> Pipeline:
        if name in self._pipelines:
            msg = f"A pipeline with name {name} is already in the registry."
            raise ValueError(msg)

        try:
            self._pipelines[name] = Pipeline.loads(source)
        except PipelineError as e:
            msg = f"Unable to parse Haystack Pipeline {name}: {e}"
            raise ValueError(msg) from e

        return self._pipelines[name]

    def remove(self, name: str):
        if name in self._pipelines:
            del self._pipelines[name]

    def get(self, name: str) -> Optional[Pipeline]:
        return self._pipelines.get(name)

    def get_names(self) -> list[str]:
        return list(self._pipelines.keys())


registry = _PipelineRegistry()
