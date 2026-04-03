import json
from collections.abc import AsyncGenerator as AsyncGeneratorABC
from collections.abc import Generator as GeneratorABC
from typing import Any

from fastapi.responses import Response, StreamingResponse
from haystack.dataclasses import StreamingChunk

from hayhooks.events import PipelineEvent
from hayhooks.server.logger import log
from hayhooks.server.pipelines.sse import SSEStream


def _format_run_stream_chunk(stream_item: Any) -> str | bytes | None:
    if isinstance(stream_item, PipelineEvent):
        log.warning("PipelineEvent emitted during /run streaming; skipping. Use OpenAI chat endpoints for UI events.")
        return None

    if isinstance(stream_item, StreamingChunk):
        if stream_item.content:
            return stream_item.content
        reasoning = getattr(stream_item, "reasoning", None)
        return (getattr(reasoning, "reasoning_text", "") or "") if reasoning is not None else ""

    if isinstance(stream_item, (str, bytes)):
        return stream_item

    if stream_item is None:
        return ""

    try:
        return json.dumps(stream_item)
    except TypeError:
        return str(stream_item)


def _format_sse_chunk(formatted: str | bytes) -> str:
    text = formatted.decode("utf-8", errors="replace") if isinstance(formatted, bytes) else str(formatted)

    if text == "":
        return "data:\n\n"

    lines = text.splitlines()
    if not lines:
        return "data:\n\n"

    data_lines = "".join(f"data: {line}\n" for line in lines)
    return f"{data_lines}\n"


def _streaming_response_from_async_gen(async_gen: Any, media_type: str = "text/plain") -> Response:
    is_sse = media_type == "text/event-stream"

    async def async_stream():
        try:
            async for item in async_gen:
                formatted = _format_run_stream_chunk(item)
                if formatted is None:
                    continue
                if is_sse:
                    formatted = _format_sse_chunk(formatted)
                yield formatted
        finally:
            aclose = getattr(async_gen, "aclose", None)
            if callable(aclose):
                await aclose()

    return StreamingResponse(async_stream(), media_type=media_type)


def _streaming_response_from_gen(gen: Any, media_type: str = "text/plain") -> Response:
    is_sse = media_type == "text/event-stream"

    def sync_stream():
        try:
            for item in gen:
                formatted = _format_run_stream_chunk(item)
                if formatted is None:
                    continue
                if is_sse:
                    formatted = _format_sse_chunk(formatted)
                yield formatted
        finally:
            close = getattr(gen, "close", None)
            if callable(close):
                close()

    return StreamingResponse(sync_stream(), media_type=media_type)


def _streaming_response_from_result(result: Any) -> Response | None:
    # If the result is a SSEStream, return a StreamingResponse with the appropriate media type
    if isinstance(result, SSEStream):
        # Get the stream from the SSEStream
        stream = result.stream

        # If the stream is an async generator, return a StreamingResponse with the appropriate media type
        if isinstance(stream, AsyncGeneratorABC):
            return _streaming_response_from_async_gen(stream, media_type="text/event-stream")

        # If the stream is a generator, return a StreamingResponse with the appropriate media type
        if isinstance(stream, GeneratorABC):
            return _streaming_response_from_gen(stream, media_type="text/event-stream")

        # If the stream is not a generator or async generator, raise a TypeError
        msg = f"SSEStream.stream must be a generator or async generator (got type {type(stream)!r})"
        raise TypeError(msg)

    # If the result is a Response, return the result
    if isinstance(result, Response):
        return result

    # Following generic cases are for non-SSE streaming responses (plain text)
    if isinstance(result, AsyncGeneratorABC):
        return _streaming_response_from_async_gen(result)
    if isinstance(result, GeneratorABC):
        return _streaming_response_from_gen(result)
    return None

