from collections.abc import Callable as CallableABC
from inspect import isclass
from types import GenericAlias
from typing import Dict, Optional, Union, get_args, get_origin, get_type_hints, Callable


def handle_unsupported_types(type_: type, types_mapping: Dict[type, type]) -> Union[GenericAlias, type]:
    """
    Recursively handle types that are not supported by Pydantic by replacing them with the given types mapping.
    """

    def is_callable_type(t):
        """Check if a type is any form of callable"""
        origin = get_origin(t)
        return (
            t is Callable
            or origin is Callable
            or origin is CallableABC
            or (origin is not None and isinstance(origin, type) and issubclass(origin, CallableABC))
            or (isinstance(t, type) and issubclass(t, CallableABC))
        )

    def handle_generics(t_) -> GenericAlias:
        """Handle generics recursively"""
        if is_callable_type(t_):
            return types_mapping[Callable]

        child_typing = []
        for t in get_args(t_):
            if t in types_mapping:
                result = types_mapping[t]
            elif is_callable_type(t):
                result = types_mapping[Callable]
            elif isclass(t):
                result = handle_unsupported_types(t, types_mapping)
            else:
                result = t
            child_typing.append(result)

        if len(child_typing) == 2 and child_typing[1] is type(None):
            return Optional[child_typing[0]]
        else:
            return GenericAlias(get_origin(t_), tuple(child_typing))

    if is_callable_type(type_):
        return types_mapping[Callable]

    if isclass(type_):
        new_type = {}
        for arg_name, arg_type in get_type_hints(type_).items():
            if get_args(arg_type):
                new_type[arg_name] = handle_generics(arg_type)
            else:
                new_type[arg_name] = arg_type
        return type_
    return handle_generics(type_)
