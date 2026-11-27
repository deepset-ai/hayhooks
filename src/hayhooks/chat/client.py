from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import TypeAdapter, ValidationError

from hayhooks.chat.events import ChatEvent, ReasoningSummary, ResponseDelta, SideMessage, ToolFinished, ToolStarted

_EVENT_TYPES = {
    "hayhooks.reasoning_summary": TypeAdapter(ReasoningSummary),
    "hayhooks.tool_call": TypeAdapter(ToolStarted),
    "hayhooks.tool_result": TypeAdapter(ToolFinished),
    "hayhooks.message": TypeAdapter(SideMessage),
}


_FailureKind = Literal["unavailable", "access", "deployment", "provider", "server", "protocol", "incomplete"]
_HTTP_FAILURES: dict[int, _FailureKind] = {401: "access", 403: "access", 404: "deployment"}
_PROVIDER_CODES = {"insufficient_quota", "rate_limit_exceeded", "invalid_api_key"}


class ChatError(Exception):
    """The deployment request could not finish."""

    def __init__(self, message: str, *, kind: _FailureKind = "server") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class _Frame:
    event: ChatEvent | None = None
    text: str | None = None
    complete: bool = False
    failure: str | None = None


def _safe_detail(value: str) -> str:
    if "Traceback (most recent call last)" in value:
        value = value.strip().splitlines()[-1]
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value)
    value = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [redacted]", value)
    value = re.sub(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]+", "[redacted]", value)
    value = re.sub(
        r"(?i)\b(api[_-]?key|token|password|secret|authorization)[\"']?\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        value,
    )
    value = re.sub(r"https?://[^\s/@]+:[^\s/@]+@", "[redacted]@", value)
    value = re.sub(r"(?<!\w)(?:[A-Za-z]:\\|/)[^\s,;]+", "[path]", value)
    return " ".join(value.split())[:240]


def _diagnostic(payload: object) -> tuple[str, str | None]:
    if not isinstance(payload, dict):
        return "", None
    detail = payload.get("detail", payload.get("error", payload))
    if isinstance(detail, str):
        return _safe_detail(detail), None
    if not isinstance(detail, dict):
        return "", None
    parts = [
        _safe_detail(value)
        for key in ("message", "code", "type", "request_id", "correlation_id")
        if isinstance(value := detail.get(key), str) and value
    ]
    code = detail.get("code")
    return "; ".join(parts)[:360], code if isinstance(code, str) else None


def _invalid_data() -> ChatError:
    return ChatError("The server sent invalid chat data", kind="protocol")


async def _sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Yield SSE lines immediately on CR or LF, including when CRLF spans two chunks."""
    response.encoding = "utf-8-sig"
    pending: list[str] = []
    skip_lf = False
    async for chunk in response.aiter_text():
        if not chunk:
            continue
        text = chunk.removeprefix("\n") if skip_lf else chunk
        skip_lf = chunk.endswith("\r")
        lines = re.split(r"\r\n|\r|\n", text)
        tail = lines.pop()
        for line in lines:
            pending.append(line)
            yield "".join(pending)
            pending.clear()
        if tail:
            pending.append(tail)
    if pending:
        yield "".join(pending)


async def _sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data: list[str] = []
    async for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


def _custom_event(event: dict, progress: Callable[[str], None]) -> ChatEvent | None:
    kind = event.get("type")
    if kind == "status":
        status = event.get("data")
        if not isinstance(status, dict):
            raise _invalid_data()
        description = status.get("description")
        if description is not None and not isinstance(description, str):
            raise _invalid_data()
        if description:
            progress(_safe_detail(description))
        return None
    adapter = _EVENT_TYPES.get(kind)
    if adapter is None:
        return None
    try:
        parsed = adapter.validate_python(event.get("data"))
    except ValidationError as error:
        raise _invalid_data() from error
    if isinstance(parsed, ToolStarted):
        try:
            arguments = json.dumps(json.loads(parsed.arguments), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            arguments = parsed.arguments
        return ToolStarted(parsed.call_id, parsed.actor, parsed.name, arguments)
    return parsed


def _choice_frame(payload: dict) -> _Frame:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise _invalid_data()
    if not choices:
        return _Frame()
    choice = choices[0]
    if not isinstance(choice, dict):
        raise _invalid_data()
    delta = choice.get("delta", {})
    if delta is None:
        delta = {}
    if not isinstance(delta, dict):
        raise _invalid_data()
    content = delta.get("content")
    finish = choice.get("finish_reason")
    if (content is not None and not isinstance(content, str)) or (finish is not None and not isinstance(finish, str)):
        raise _invalid_data()
    failure = {
        "length": (
            "The model reached its output limit or the Agent reached its step limit. "
            "Shorten the request or increase the deployment's output limit or Agent max_agent_steps. "
            "Check work already performed by tools before retrying"
        ),
        "content_filter": "The model filtered the response. Revise the request or check the deployment's policy",
    }.get(finish)
    return _Frame(text=content, complete=finish == "stop", failure=failure)


def _decode(data: str, progress: Callable[[str], None]) -> _Frame:
    if data == "[DONE]":
        return _Frame(complete=True)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise _invalid_data() from error
    if not isinstance(payload, dict):
        raise _invalid_data()
    if "error" in payload:
        detail, code = _diagnostic(payload)
        kind = "provider" if code in _PROVIDER_CODES else "server"
        message = f"Deployment error: {detail or 'The server supplied no cause'}"
        raise ChatError(message, kind=kind)
    if "event" not in payload:
        return _choice_frame(payload)
    event = payload["event"]
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise _invalid_data()
    return _Frame(event=_custom_event(event, progress))


class ChatClient:
    def __init__(self, server_url: str, pipeline: str, *, verify_ssl: bool = True) -> None:
        self.server_url = server_url.rstrip("/")
        self.pipeline = pipeline
        self.verify_ssl = verify_ssl
        self.history: list[dict[str, str]] = []

    def _error(self, kind: _FailureKind, cause: str) -> ChatError:
        action = {
            "unavailable": "Check that the Hayhooks server is running and the server address is correct",
            "access": "Check access to the Hayhooks server",
            "deployment": "Check the deployment name and deploy the wrapper before sending again",
            "provider": "Check the deployment's model credentials, quota, and server logs before sending again",
            "server": "Check the Hayhooks server logs for the cause before sending again",
            "protocol": "Check the wrapper's chat stream and the Hayhooks server logs",
            "incomplete": "Completion is unknown and tools may have run. Check the server before sending again",
        }[kind]
        message = f"Deployment '{_safe_detail(self.pipeline)}': {cause}. {action}."
        return ChatError(message, kind=kind)

    async def _check_response(self, response: httpx.Response) -> None:
        if response.is_error:
            await response.aread()
            try:
                detail, code = _diagnostic(response.json())
            except ValueError:
                detail, code = "", None
            kind = _HTTP_FAILURES.get(response.status_code, "server")
            if code in _PROVIDER_CODES:
                kind = "provider"
            cause = f"Server returned HTTP {response.status_code}: {detail or 'The server supplied no cause'}"
            request_id = response.headers.get("x-request-id")
            if request_id:
                cause += f" (request {_safe_detail(request_id)})"
            raise self._error(kind, cause)
        if "text/event-stream" not in response.headers.get("content-type", ""):
            message = "The deployment did not return a chat event stream"
            failure = self._error("protocol", message)
            raise failure

    async def run(self, prompt: str, progress: Callable[[str], None], *, emit: Callable[[ChatEvent], None]) -> str:
        messages = [*self.history, {"role": "user", "content": prompt}]
        segments: list[list[str]] = [[]]
        complete = False
        timeout = httpx.Timeout(10.0, read=None)
        progress("Connecting")
        try:
            async with (
                httpx.AsyncClient(verify=self.verify_ssl, timeout=timeout) as client,
                client.stream(
                    "POST",
                    f"{self.server_url}/v1/chat/completions",
                    json={"model": self.pipeline, "messages": messages, "stream": True},
                ) as response,
            ):
                await self._check_response(response)
                progress(f"Connected · {self.pipeline}")
                async for data in _sse_data(_sse_lines(response)):
                    try:
                        frame = _decode(data, progress)
                    except ChatError as error:
                        raise self._error(error.kind, str(error)) from error
                    if isinstance(frame.event, ToolStarted) and segments[-1]:
                        segments.append([])
                    if frame.event is not None:
                        emit(frame.event)
                    if frame.text:
                        segments[-1].append(frame.text)
                        emit(ResponseDelta(len(segments) - 1, frame.text))
                    if frame.failure:
                        failure = self._error("provider", frame.failure)
                        raise failure
                    complete = complete or frame.complete
        except httpx.HTTPError as error:
            unavailable = isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
            kind = "unavailable" if unavailable else "incomplete"
            cause = "Could not connect to the Hayhooks server" if unavailable else "The chat connection was lost"
            raise self._error(kind, cause) from error
        if not complete:
            failure = self._error("incomplete", "The chat stream ended without a completion marker")
            raise failure
        answer = "\n\n".join("".join(segment) for segment in segments if segment)
        self.history = [*messages, {"role": "assistant", "content": answer}]
        return answer
