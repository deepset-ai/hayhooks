"""Pipeline wrapper with __init__.py in the package."""

from haystack import Pipeline
from haystack.core.component import component

from hayhooks import BasePipelineWrapper

# Import from __init__.py
from . import PACKAGE_VERSION

# Import from sibling module
from .helpers import double


@component
class Doubler:
    @component.output_types(result=int, version=str)
    def run(self, value: int):
        return {"result": double(value), "version": PACKAGE_VERSION}


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.pipeline.add_component("doubler", Doubler())

    def run_api(self, value: int) -> dict:
        result = self.pipeline.run({"doubler": {"value": value}})
        return result["doubler"]
