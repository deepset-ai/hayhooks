import logging
import os
import sys
import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from loguru import logger as log

from hayhooks.cli.theme import BRAND_COLOR, ERROR_COLOR, MUTED_COLOR, SUCCESS_COLOR

F = TypeVar("F", bound=Callable)

_TS = f"<fg {MUTED_COLOR}>{{time:YYYY-MM-DD HH:mm:ss}}</fg {MUTED_COLOR}>"
_LOC = (
    f"<fg {BRAND_COLOR}>{{name}}</fg {BRAND_COLOR}>"
    f":<fg {BRAND_COLOR}>{{function}}</fg {BRAND_COLOR}>"
    f":<fg {BRAND_COLOR}>{{line}}</fg {BRAND_COLOR}>"
)
_EXTRA = f"<fg {MUTED_COLOR}>{{extra}}</fg {MUTED_COLOR}>"

_MSG_STYLE = {
    "SUCCESS": ("<bold>", "</bold>"),
    "ERROR": (f"<fg {ERROR_COLOR}><bold>", f"</bold></fg {ERROR_COLOR}>"),
    "CRITICAL": (f"<fg {ERROR_COLOR}><bold>", f"</bold></fg {ERROR_COLOR}>"),
}

_VERBOSE = os.getenv("HAYHOOKS_LOG_FORMAT", "default").lower() == "verbose"


def formatter(record):
    parts = [_TS, " | ", "<level>{level: <8}</level>", " | "]
    if _VERBOSE:
        parts += [_LOC, " | "]

    style = _MSG_STYLE.get(record["level"].name)
    if style:
        parts += [style[0], "{message}", style[1]]
    else:
        parts.append("{message}")

    if record["extra"]:
        parts += [" - ", _EXTRA]
    parts.append("\n")
    return "".join(parts)


log.remove()

log.level("DEBUG", color="<fg #BBBBBB>")
log.level("INFO", color=f"<fg {BRAND_COLOR}>")
log.level("SUCCESS", color=f"<bold><fg {SUCCESS_COLOR}>")
log.level("WARNING", color="<yellow><bold>")
log.level("ERROR", color=f"<fg {ERROR_COLOR}><bold>")
log.level("CRITICAL", color=f"<fg {ERROR_COLOR}><bold>")

log.add(sys.stderr, level=os.getenv("LOG", "INFO").upper(), format=formatter)


class _InterceptHandler(logging.Handler):
    """Route standard-library logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = log.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def intercept_stdlib_logging() -> None:
    """
    Install loguru as the root handler so all stdlib logging flows through it.

    This must be called *after* uvicorn configures its loggers (i.e. inside
    a lifespan/startup hook or right before ``uvicorn.run``).  We patch the
    root logger so that any logger -- including ones uvicorn creates after
    our call -- inherits the intercept handler via propagation.
    """
    logging.root.handlers = [_InterceptHandler()]
    logging.root.setLevel(logging.DEBUG)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers.clear()
        stdlib_logger.propagate = True


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

        return wrapper  # ty: ignore[invalid-return-type]

    return decorator
