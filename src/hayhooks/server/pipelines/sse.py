from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any, Union

StreamType = Union[Generator[Any, None, None], AsyncGenerator[Any, None]]


@dataclass(slots=True)
class SSEStream:
    """
    Declarative container that flags a generator (sync or async) for SSE streaming.
    """

    stream: StreamType
