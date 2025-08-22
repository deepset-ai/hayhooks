from haystack import Pipeline
from haystack.core.component import component

# This import assumes that './common' is in the Python path.
from my_custom_lib import subtract_two_numbers

from hayhooks import BasePipelineWrapper


@component
class Subtractor:
    @component.output_types(difference=int)
    def run(self, a: int, b: int):
        return {"difference": subtract_two_numbers(a=a, b=b)}


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.pipeline.add_component("subtractor", Subtractor())

    def run_api(self, a: int, b: int) -> int:
        result = self.pipeline.run({"subtractor": {"a": a, "b": b}})
        return result["subtractor"]["difference"]
