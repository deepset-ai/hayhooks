import os
import sys
import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from loguru import logger as log

F = TypeVar("F", bound=Callable)


def formatter(record):
    if record["extra"]:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:"
            "<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level> - <magenta>{extra}</magenta>\n"
        )

    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>"
        "{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>\n"
    )


log.remove()
log.add(sys.stderr, level=os.getenv("LOG", "INFO").upper(), format=formatter)


def log_elapsed(level: str = "DEBUG") -> Callable[[F], F]:
    """Decorator that logs wall-clock time of a function call."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            result = func(*args, **kwargs)
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.opt(depth=1).log(level, "{}() completed in {:.0f}ms", func.__name__, elapsed_ms)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
