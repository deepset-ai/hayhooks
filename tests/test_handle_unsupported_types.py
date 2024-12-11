from typing import Optional, List
from hayhooks.server.utils.create_valid_type import handle_unsupported_types


def test_handle_simple_type():
    result = handle_unsupported_types(int, {})
    assert result == int


def test_handle_generic_type():
    result = handle_unsupported_types(List[int], {})
    assert result == list[int]


def test_handle_recursive_type():
    class Node:
        def __init__(self, value: int, next: Optional['Node'] = None):
            self.value = value
            self.next = next

    result = handle_unsupported_types(Node, {})
    assert result == Node


def test_handle_circular_reference():
    class A:
        def __init__(self, b: 'B'):
            self.b = b

    class B:
        def __init__(self, a: 'A'):
            self.a = a

    result = handle_unsupported_types(A, {})
    assert result == A  # Adjust assertion based on expected behavior


def test_handle_nested_generics():
    nested_type = dict[str, list[Optional[int]]]
    result = handle_unsupported_types(nested_type, {})
    assert result == nested_type
