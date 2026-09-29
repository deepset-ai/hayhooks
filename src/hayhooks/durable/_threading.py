"""Small daemon-thread bridge for synchronous durable work."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future as ThreadFuture
from contextlib import suppress
from contextvars import copy_context
from threading import Thread
from typing import TypeVar

_T = TypeVar("_T")


def start_daemon_thread(function: Callable[[], _T], *, name: str) -> tuple[asyncio.Future[_T], asyncio.Event]:
    """Start context-aware work without making interpreter shutdown wait for it."""
    loop = asyncio.get_running_loop()
    done = asyncio.Event()
    result: ThreadFuture[_T] = ThreadFuture()
    result.set_running_or_notify_cancel()
    active_context = copy_context()

    def run() -> None:
        try:
            result.set_result(function())
        except BaseException as error:
            result.set_exception(error)
        finally:
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(done.set)

    Thread(target=active_context.run, args=(run,), name=name, daemon=True).start()
    return asyncio.wrap_future(result), done
