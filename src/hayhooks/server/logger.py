import logging
import os
import sys
import time
import uuid
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from loguru import logger as log

from hayhooks.colors import BRAND_COLOR, ERROR_COLOR, MUTED_COLOR, REQ_ID_COLOR, SUCCESS_COLOR

F = TypeVar("F", bound=Callable)

_TS = f"<fg {MUTED_COLOR}>{{time:YYYY-MM-DD HH:mm:ss}}</fg {MUTED_COLOR}>"
_LOC = (
    f"<fg {BRAND_COLOR}>{{name}}</fg {BRAND_COLOR}>"
    f":<fg {BRAND_COLOR}>{{function}}</fg {BRAND_COLOR}>"
    f":<fg {BRAND_COLOR}>{{line}}</fg {BRAND_COLOR}>"
)
_REQ_ID = f"<fg {REQ_ID_COLOR}>{{extra[request_id]}}</fg {REQ_ID_COLOR}>"

_MSG_STYLE = {
    "SUCCESS": ("<bold>", "</bold>"),
    "ERROR": (f"<fg {ERROR_COLOR}><bold>", f"</bold></fg {ERROR_COLOR}>"),
    "CRITICAL": (f"<fg {ERROR_COLOR}><bold>", f"</bold></fg {ERROR_COLOR}>"),
}

_VERBOSE = os.getenv("HAYHOOKS_LOG_FORMAT", "default").lower() == "verbose"


def formatter(record):
    parts = [_TS, " | ", "<level>{level: <8}</level>", " | "]

    if record["extra"].get("request_id"):
        parts += [_REQ_ID, " | "]

    if _VERBOSE:
        parts += [_LOC, " | "]

    style = _MSG_STYLE.get(record["level"].name)
    if style:
        parts += [style[0], "{message}", style[1]]
    else:
        parts.append("{message}")

    extra = {k: v for k, v in record["extra"].items() if k != "request_id"}
    if extra:
        flat = ", ".join(f"{k}={v!r}" for k, v in extra.items())
        escaped = flat.replace("{", "{{").replace("}", "}}")
        parts.append(f" <fg {MUTED_COLOR}>| {escaped}</fg {MUTED_COLOR}>")
    parts.append("\n")
    return "".join(parts)


log.remove()

log.level("DEBUG", color="<fg #BBBBBB>")
log.level("INFO", color=f"<fg {BRAND_COLOR}>")
log.level("SUCCESS", color=f"<bold><fg {SUCCESS_COLOR}>")
log.level("WARNING", color="<yellow><bold>")
log.level("ERROR", color=f"<fg {ERROR_COLOR}><bold>")
log.level("CRITICAL", color=f"<fg {ERROR_COLOR}><bold>")

_log_level = os.getenv("HAYHOOKS_LOG_LEVEL") or os.getenv("LOG", "INFO")
log.add(sys.stderr, level=_log_level.upper(), format=formatter)


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

        if "color_message" in record.__dict__ and record.args:
            msg = str(record.__dict__["color_message"]) % record.args
        else:
            msg = record.getMessage()
        log.opt(depth=depth, exception=record.exc_info).log(level, msg)


def generate_request_id() -> str:
    return uuid.uuid4().hex[:8]


class RequestIdMiddleware:
    """ASGI middleware that assigns a request ID and binds it to loguru context."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        request_id = generate_request_id()
        scope["state"] = {**scope.get("state", {}), "request_id": request_id}

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        with log.contextualize(request_id=request_id):
            await self.app(scope, receive, send_with_request_id)


_DEFAULT_INTERCEPTED_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi")


def intercept_stdlib_logging(loggers: list[str] | tuple[str, ...] | None = None) -> None:
    """
    Replace handlers on the given loggers so they flow through loguru.

    Args:
        loggers: Logger names to intercept. Falls back to
            ``_DEFAULT_INTERCEPTED_LOGGERS`` when *None*.
    """
    handler = _InterceptHandler()
    for name in loggers or _DEFAULT_INTERCEPTED_LOGGERS:
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [handler]
        stdlib_logger.setLevel(logging.DEBUG)
        stdlib_logger.propagate = False


def log_elapsed(level: str = "DEBUG") -> Callable[[F], F]:
    """Decorator that logs wall-clock time of a function call."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            result = func(*args, **kwargs)
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.opt(depth=1, colors=True).log(
                level, "<bold>{}</bold>() completed in <bold>{:.0f}ms</bold>", func.__name__, elapsed_ms
            )
            return result

        return wrapper  # ty: ignore[invalid-return-type]

    return decorator
