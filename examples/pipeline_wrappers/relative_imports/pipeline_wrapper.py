"""
Pipeline wrapper demonstrating relative imports.

This example shows how to organize your pipeline wrapper code into multiple files
using Python's relative import syntax.
"""

# ruff: noqa: TID252

from haystack import Pipeline
from haystack.core.component import component

from hayhooks import BasePipelineWrapper

# Relative imports from sibling modules in the same folder
from .utils import calculate_average, calculate_sum, greet


@component
class Calculator:
    """A simple component that performs calculations using imported utilities."""

    @component.output_types(greeting=str, sum=int, average=float)
    def run(self, name: str, numbers: list[int]) -> dict:
        return {
            "greeting": greet(name),
            "sum": calculate_sum(numbers),
            "average": calculate_average(numbers),
        }


class PipelineWrapper(BasePipelineWrapper):
    """
    A pipeline wrapper that demonstrates relative imports.

    The helper functions (greet, calculate_sum, calculate_average) are defined
    in utils.py and imported using relative imports.
    """

    def setup(self) -> None:
        """Initialize the pipeline with a Calculator component."""
        self.pipeline = Pipeline()
        self.pipeline.add_component("calculator", Calculator())

    def run_api(self, name: str, numbers: list[int]) -> dict:
        """
        Run the calculator pipeline.

        Args:
            name: Name to greet
            numbers: List of numbers to calculate sum and average

        Returns:
            Dictionary with greeting, sum, and average
        """
        result = self.pipeline.run({"calculator": {"name": name, "numbers": numbers}})
        return result["calculator"]
