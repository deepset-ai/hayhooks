from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any

StreamType = Generator[Any, None, None] | AsyncGenerator[Any, None]


@dataclass
class SSEStream:
    """
    Declarative container that flags a generator (sync or async) for SSE streaming.
    """

    stream: StreamType
