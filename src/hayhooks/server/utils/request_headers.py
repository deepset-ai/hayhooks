import inspect
from collections.abc import Callable
from types import SimpleNamespace, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from starlette.datastructures import Headers

from hayhooks.server.exceptions import PipelineWrapperError


def accepts_request_headers(method: Callable[..., Any]) -> bool:
    """Recognize and validate the explicit ``headers: Headers | None = None`` opt-in."""
    try:
        parameter = inspect.signature(method).parameters.get("headers")
    except (TypeError, ValueError):
        return False
    if parameter is None or parameter.annotation is inspect.Parameter.empty:
        return False

    # Resolve only this annotation; unrelated forward references need not be importable at runtime.
    try:
        annotation = get_type_hints(
            SimpleNamespace(__annotations__={"headers": parameter.annotation}),
            globalns=getattr(inspect.unwrap(method), "__globals__", {}),
        )["headers"]
    except NameError:
        # An unresolved application type does not opt in to HTTP metadata.
        return False
    if annotation is not Headers and not (
        get_origin(annotation) in (Union, UnionType) and Headers in get_args(annotation)
    ):
        return False

    if (
        annotation != Headers | None
        or parameter.default is not None
        or parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ):
        name = getattr(method, "__name__", type(method).__name__)
        msg = f"{name}: request headers must be declared as a keyword parameter 'headers: Headers | None = None'"
        raise PipelineWrapperError(msg)
    return True
