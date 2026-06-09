"""
Fan-out of live trace-buffer changes to SSE subscribers.

Span recording happens synchronously (often on worker threads); SSE consumers
run on the asyncio event loop. This module bridges the two: a sync ``notify()``
called from the recording hot path wakes each async subscriber via
``loop.call_soon_threadsafe``.

Each subscriber queue is a *coalescing wake-signal* (``maxsize=1``), so a burst
of span events collapses into a single delta read on the consumer side. The
payload is irrelevant — the consumer re-reads the buffer from its own cursor
when woken.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from threading import Lock

from hayhooks.server.logger import log
from hayhooks.server.utils.live_trace_buffer import register_change_listener


class _TraceStreamBroadcaster:
    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: set[asyncio.Queue[int]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def clear_loop(self) -> None:
        with self._lock:
            self._loop = None

    def subscribe(self) -> asyncio.Queue[int]:
        queue: asyncio.Queue[int] = asyncio.Queue(maxsize=1)
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[int]) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def notify(self) -> None:
        """Wake all subscribers. Safe to call from any thread; never raises."""
        with self._lock:
            loop = self._loop
            subscribers = list(self._subscribers)
        if loop is None or not subscribers:
            return
        for queue in subscribers:
            # RuntimeError means the loop is closed/closing — drop the signal.
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self._wake, queue)

    @staticmethod
    def _wake(queue: asyncio.Queue[int]) -> None:
        # Runs on the loop thread; coalesce by skipping when already signalled.
        if queue.full():
            return
        with suppress(asyncio.QueueFull):
            queue.put_nowait(1)


_BROADCASTER = _TraceStreamBroadcaster()


def get_trace_stream_broadcaster() -> _TraceStreamBroadcaster:
    return _BROADCASTER


def notify_trace_changed() -> None:
    _BROADCASTER.notify()


# Wire the buffer's change hook to our notify as soon as this module is imported
# (it is imported when the dashboard router loads, before any traces flow).
register_change_listener(notify_trace_changed)
log.debug("Live trace stream broadcaster registered as buffer change listener")
