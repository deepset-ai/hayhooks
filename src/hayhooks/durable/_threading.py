"""Small daemon-thread bridge for synchronous durable work."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import Future as ThreadFuture
from contextlib import suppress
from contextvars import copy_context
from threading import Thread
from typing import TypeVar

_T = TypeVar("_T")
# Never let these reach the event loop: SystemExit and KeyboardInterrupt stop it, and Python 3.10/3.11 cannot set
# StopIteration on an asyncio future. Other BaseExceptions, including the runtime's signals, pass through.
_CONVERTED = (SystemExit, KeyboardInterrupt, GeneratorExit, StopIteration, StopAsyncIteration)


def is_async_callable(function: object) -> bool:
    """Whether calling ``function`` returns a coroutine, including objects with an async ``__call__``."""
    return inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(type(function).__call__)


def start_daemon_thread(function: Callable[[], _T], *, name: str) -> tuple[asyncio.Future[_T], asyncio.Future[None]]:
    """
    Start context-aware work without making interpreter shutdown wait for it.

    Returns the work's result and a future that resolves once the thread has exited, which
    outlives cancellation of the result.
    """
    loop = asyncio.get_running_loop()
    exited: asyncio.Future[None] = loop.create_future()
    result: ThreadFuture[_T] = ThreadFuture()
    result.set_running_or_notify_cancel()
    active_context = copy_context()

    def run() -> None:
        try:
            result.set_result(function())
        except _CONVERTED as error:
            converted = RuntimeError(f"durable work raised {type(error).__name__}")
            converted.__cause__ = error
            result.set_exception(converted)
        except BaseException as error:
            result.set_exception(error)
        finally:
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(exited.set_result, None)

    Thread(target=active_context.run, args=(run,), name=name, daemon=True).start()
    return asyncio.wrap_future(result), exited
