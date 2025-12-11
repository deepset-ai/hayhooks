"""Pipeline wrapper that uses relative imports from a sibling module in the same folder."""

from haystack import Pipeline
from haystack.core.component import component

from hayhooks import BasePipelineWrapper

# Relative import from sibling module in the same folder
from .helper import greet, multiply


@component
class Calculator:
    """A component that uses helper functions from a sibling module."""

    @component.output_types(greeting=str, product=int)
    def run(self, name: str, a: int, b: int):
        return {"greeting": greet(name), "product": multiply(a, b)}


class PipelineWrapper(BasePipelineWrapper):
    """A pipeline wrapper that tests importing from sibling modules."""

    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.pipeline.add_component("calculator", Calculator())

    def run_api(self, name: str, a: int, b: int) -> dict:
        """
        Run the pipeline with a name and two numbers.

        Args:
            name: Name to greet
            a: First number to multiply
            b: Second number to multiply

        Returns:
            Dictionary with greeting and multiplication result
        """
        result = self.pipeline.run({"calculator": {"name": name, "a": a, "b": b}})
        return {
            "greeting": result["calculator"]["greeting"],
            "multiply_result": result["calculator"]["product"],
        }
