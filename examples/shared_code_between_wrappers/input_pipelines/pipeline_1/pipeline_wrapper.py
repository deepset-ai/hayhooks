from haystack import Pipeline
from haystack.core.component import component

# This import assumes that './common' is in the Python path.
from my_custom_lib import sum_two_numbers

from hayhooks import BasePipelineWrapper


@component
class Summer:
    @component.output_types(sum=int)
    def run(self, a: int, b: int):
        return {"sum": sum_two_numbers(a=a, b=b)}


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.pipeline.add_component("summer", Summer())

    def run_api(self, a: int, b: int) -> int:
        result = self.pipeline.run({"summer": {"a": a, "b": b}})
        return result["summer"]["sum"]
