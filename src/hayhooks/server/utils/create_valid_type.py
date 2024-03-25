from typing import get_type_hints, Dict, get_origin, get_args
from typing_extensions import TypedDict
from types import GenericAlias
from inspect import isclass

def create_valid_type(typed_object:type, invalid_types:Dict[type, type]):
    """ 
        Returns a new type specification, replacing invalid_types in typed_object.
        example: replace_invalid_types(ExtractedAnswer, {DataFrame: List}]) returns 
        a TypedDict with DataFrame types replaced with List
    """
    def validate_type(v):
        child_typing = []
        for t in get_args(v):
            if t in invalid_types:
                result = invalid_types[t]
            elif isclass(t):
                result = create_valid_type(t, invalid_types)
            else: result = t
            child_typing.append(result)
        return GenericAlias(get_origin(v), tuple(child_typing))
    if isclass(typed_object):
        new_typing = {}
        for k, v in get_type_hints(typed_object).items():
            if(get_args(v) != ()):
                new_typing[k] = validate_type(v)
            else: new_typing[k] = v
        if new_typing == {}:
            return typed_object
        else: return TypedDict(typed_object.__name__, new_typing)
    else:
        return validate_type(typed_object)