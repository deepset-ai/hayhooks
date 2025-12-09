"""
Utility functions for the pipeline wrapper.

This module demonstrates how to organize helper functions in a separate file
and import them using relative imports.
"""


def greet(name: str) -> str:
    """Generate a greeting message."""
    return f"Hello, {name}!"


def calculate_sum(numbers: list[int]) -> int:
    """Calculate the sum of a list of numbers."""
    return sum(numbers)


def calculate_average(numbers: list[int]) -> float:
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
