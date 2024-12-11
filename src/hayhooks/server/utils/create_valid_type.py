from collections.abc import Callable as CallableABC
from types import GenericAlias
from typing import Callable, Optional, Union, get_args, get_origin
from loguru import logger


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
    type_: type, types_mapping: dict, skip_callables: bool = True
) -> Union[GenericAlias, type, None]:
    logger.debug(f"Handling unsupported type: {type_}")

    if skip_callables and is_callable_type(type_):
        logger.warning(f"Skipping callable type: {type_}")
        return None

    # Handle generic types (like List, Optional, etc.)
    origin = get_origin(type_)
    if origin is not None:
        args = get_args(type_)
        # Map the inner types using the same mapping
        mapped_args = tuple(handle_unsupported_types(arg, types_mapping, skip_callables) or arg for arg in args)
        # Reconstruct the generic type with mapped arguments
        return origin[mapped_args]

    if type_ in types_mapping:
        logger.debug(f"Mapping type: {type_} to {types_mapping[type_]}")
        return types_mapping[type_]

    logger.debug(f"Returning original type: {type_}")
    return type_
