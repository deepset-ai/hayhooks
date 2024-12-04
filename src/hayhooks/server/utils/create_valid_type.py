from collections.abc import Callable as CallableABC
from inspect import isclass
from types import GenericAlias
from typing import Callable, Dict, Optional, Union, get_args, get_origin, get_type_hints


def is_callable_type(t):
    """Check if a type is any form of callable"""
    if t in (Callable, CallableABC):
        return True

    # Check origin type
    origin = get_origin(t)
    if origin in (Callable, CallableABC):
        return True

    # Handle Optional/Union types
    if origin in (Union, type(Optional[int])):  # type(Optional[int]) handles runtime Optional type
        args = get_args(t)
        return any(is_callable_type(arg) for arg in args)

    return False


def handle_unsupported_types(
    type_: type, types_mapping: Dict[type, type], skip_callables: bool = True, visited_types: set = None
) -> Union[GenericAlias, type, None]:
    if visited_types is None:
        visited_types = set()

    if type_ in visited_types:
        return type_

    visited_types.add(type_)

    def handle_generics(t_) -> Union[GenericAlias, None]:
        if is_callable_type(t_) and skip_callables:
            return None

        if t_ in visited_types:
            return t_

        child_typing = []
        for t in get_args(t_):
            if t in types_mapping:
                result = types_mapping[t]
            elif isclass(t):
                result = handle_unsupported_types(t, types_mapping, skip_callables, visited_types)
            else:
                result = t
            child_typing.append(result)

        if len(child_typing) == 2 and child_typing[1] is type(None):
            return Optional[child_typing[0]]
        return GenericAlias(get_origin(t_), tuple(child_typing))

    if is_callable_type(type_) and skip_callables:
        return None

    if isclass(type_):
        new_type = {}
        for arg_name, arg_type in get_type_hints(type_).items():
            if get_args(arg_type):
                new_type[arg_name] = handle_generics(arg_type)
            else:
                new_type[arg_name] = arg_type
        return type_
    return handle_generics(type_)
