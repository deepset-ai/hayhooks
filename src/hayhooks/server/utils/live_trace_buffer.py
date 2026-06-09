from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from threading import RLock
from time import time
from typing import Any, TypedDict

from typing_extensions import NotRequired

from hayhooks.server.logger import log
from hayhooks.settings import settings

_TRACE_TAG_PRIORITY = (
    "hayhooks.pipeline.name",
    "hayhooks.transport",
    "hayhooks.openai.operation",
    "hayhooks.openai.stream_requested",
    "hayhooks.openai.execution_mode",
    "hayhooks.deploy.strategy",
    "hayhooks.response.stream_type",
    "hayhooks.response.streaming",
    "hayhooks.success",
    "hayhooks.error.type",
    "hayhooks.error.message",
    "hayhooks.error.stack",
    "hayhooks.http.status_code",
)
_IGNORED_TRACE_TAG_KEYS = {"hayhooks.elapsed_ms"}
_MAX_TRACE_TAGS = 32
_MAX_SPAN_TAGS = 8
_MAX_SPANS_PER_TRACE = 500
_MAX_TAG_VALUE_LENGTH = 500


class TraceTagDict(TypedDict):
    key: str
    value: str


class TraceSpanNodeDict(TypedDict):
    span_id: str
    name: str
    start_time_ms: int
    duration_ms: int
    running: bool
    tags: list[TraceTagDict]
    children: list[TraceSpanNodeDict]


class TraceSummaryDict(TypedDict):
    trace_id: str
    start_time_ms: int
    duration_ms: int
    entrypoint: str | None
    tags: list[TraceTagDict]
    span_count: int
    root_span: TraceSpanNodeDict
    _cursor_seq: NotRequired[int]


class _SpanState(TypedDict):
    span_id: str
    parent_span_id: str | None
    name: str
    start_time_ms: int
    duration_ms: int
    running: bool
    tags: dict[str, Any]


class _TraceState(TypedDict):
    trace_id: str
    entrypoint: str | None
    spans: dict[str, _SpanState]
    start_time_ms: int
    updated_at_ms: int
    cursor_seq: int


def _new_trace_state(*, trace_id: str, start_time_ms: int, cursor_seq: int) -> _TraceState:
    return {
        "trace_id": trace_id,
        "entrypoint": None,
        "spans": {},
        "start_time_ms": start_time_ms,
        "updated_at_ms": start_time_ms,
        "cursor_seq": cursor_seq,
    }


def _new_span_state(
    *,
    span_id: str,
    parent_span_id: str | None,
    operation_name: str,
    start_time_ms: int,
    tags: dict[str, Any],
) -> _SpanState:
    return {
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": operation_name,
        "start_time_ms": start_time_ms,
        "duration_ms": 0,
        "running": True,
        "tags": tags,
    }


def _max_buffered_traces() -> int:
    return settings.dashboard_trace_buffer_capacity


def _truncate_tag_text(value: str) -> str:
    return value[:_MAX_TAG_VALUE_LENGTH] if len(value) > _MAX_TAG_VALUE_LENGTH else value


def _truncate_tag_values(tags: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in tags.items():
        if isinstance(value, str):
            result[key] = _truncate_tag_text(value)
        elif isinstance(value, Mapping | list | tuple | set):
            try:
                result[key] = _truncate_tag_text(json.dumps(value, sort_keys=True, default=str))
            except TypeError:
                result[key] = _truncate_tag_text(str(value))
        else:
            result[key] = value
    return result


def _stringify_tag_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping | list | tuple | set):
        try:
            text = json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            text = str(value)
    else:
        text = str(value)
    text = text.strip()
    return _truncate_tag_text(text) if text else None


def _collect_span_tags(span_tags: dict[str, Any]) -> list[TraceTagDict]:
    """Collect a small curated set of tags for a single span."""
    tags: list[TraceTagDict] = []
    seen: set[str] = set()
    for key in _TRACE_TAG_PRIORITY:
        if key in span_tags:
            value_text = _stringify_tag_value(span_tags[key])
            if value_text is not None:
                tags.append({"key": key, "value": value_text})
                seen.add(key)
        if len(tags) >= _MAX_SPAN_TAGS:
            return tags
    for key, value in span_tags.items():
        if key in seen or key in _IGNORED_TRACE_TAG_KEYS:
            continue
        value_text = _stringify_tag_value(value)
        if value_text is not None:
            tags.append({"key": key, "value": value_text})
        if len(tags) >= _MAX_SPAN_TAGS:
            break
    return tags


def _collect_trace_tags(span_tag_maps: list[dict[str, Any]]) -> list[TraceTagDict]:  # noqa: C901
    tags: list[TraceTagDict] = []
    seen_keys: set[str] = set()

    def add_tag(key: str, value: Any) -> None:
        if key in seen_keys or key in _IGNORED_TRACE_TAG_KEYS:
            return
        value_text = _stringify_tag_value(value)
        if value_text is None:
            return
        tags.append({"key": key, "value": value_text})
        seen_keys.add(key)

    for key in _TRACE_TAG_PRIORITY:
        for span_tags in span_tag_maps:
            if key in span_tags:
                add_tag(key, span_tags[key])
                break

    for span_tags in span_tag_maps:
        for key, value in span_tags.items():
            if len(tags) >= _MAX_TRACE_TAGS:
                break
            if key not in seen_keys:
                add_tag(key, value)
        if len(tags) >= _MAX_TRACE_TAGS:
            break

    return tags[:_MAX_TRACE_TAGS]


class _LiveTraceBuffer:
    def __init__(self) -> None:
        self._lock = RLock()
        self._traces: dict[str, _TraceState] = {}
        self._next_cursor_seq = 1

    def _issue_cursor_seq(self) -> int:
        current = self._next_cursor_seq
        self._next_cursor_seq += 1
        return current

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()
            self._next_cursor_seq = 1

    def record_span_start(  # noqa: PLR0913
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        operation_name: str,
        start_time_ms: int,
        tags: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            trace_state = self._traces.get(trace_id)
            if trace_state is None:
                trace_state = _new_trace_state(trace_id=trace_id, start_time_ms=start_time_ms, cursor_seq=0)
                self._traces[trace_id] = trace_state

            if len(trace_state["spans"]) >= _MAX_SPANS_PER_TRACE:
                return

            trace_state["updated_at_ms"] = start_time_ms
            trace_state["cursor_seq"] = self._issue_cursor_seq()
            trace_state["start_time_ms"] = min(trace_state["start_time_ms"], start_time_ms)
            span_tags: dict[str, Any] = _truncate_tag_values(dict(tags or {}))
            if (
                trace_state["entrypoint"] is None
                and (entrypoint := span_tags.get("hayhooks.pipeline.name")) is not None
            ):
                trace_state["entrypoint"] = str(entrypoint)

            trace_state["spans"][span_id] = _new_span_state(
                span_id=span_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time_ms=start_time_ms,
                tags=span_tags,
            )

            self._evict_old_traces()

    def record_span_finish(
        self,
        *,
        trace_id: str,
        span_id: str,
        duration_ms: int,
        completed_at_ms: int,
        tags: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            trace_state = self._traces.get(trace_id)
            if trace_state is None:
                return

            span = trace_state["spans"].get(span_id)
            if span is None:
                return

            span["duration_ms"] = max(0, duration_ms)
            span["running"] = False
            if tags:
                span["tags"].update(_truncate_tag_values(dict(tags)))
                if (
                    trace_state["entrypoint"] is None
                    and (entrypoint := span["tags"].get("hayhooks.pipeline.name")) is not None
                ):
                    trace_state["entrypoint"] = str(entrypoint)
            trace_state["updated_at_ms"] = completed_at_ms
            trace_state["cursor_seq"] = self._issue_cursor_seq()
            self._evict_old_traces()

    def get_recent_traces(
        self,
        *,
        since_ms: int | None,
        limit: int,
        after_seq: int | None = None,
    ) -> list[TraceSummaryDict]:
        with self._lock:
            if after_seq is not None:
                traces = [trace for trace in self._traces.values() if trace["cursor_seq"] > after_seq]
                traces.sort(key=lambda trace: trace["cursor_seq"])
            else:
                traces = sorted(self._traces.values(), key=lambda trace: trace["updated_at_ms"], reverse=True)
            if since_ms is not None:
                traces = [trace for trace in traces if trace["start_time_ms"] >= since_ms]

            trace_copies: list[_TraceState] = []
            for trace in traces:
                spans_copy: dict[str, _SpanState] = {}
                for span_id, span in trace["spans"].items():
                    tags_copy: dict[str, Any] = dict(span["tags"])
                    spans_copy[span_id] = {
                        "span_id": span["span_id"],
                        "parent_span_id": span["parent_span_id"],
                        "name": span["name"],
                        "start_time_ms": span["start_time_ms"],
                        "duration_ms": span["duration_ms"],
                        "running": span["running"],
                        "tags": tags_copy,
                    }
                trace_copies.append(
                    {
                        "trace_id": trace["trace_id"],
                        "entrypoint": trace["entrypoint"],
                        "spans": spans_copy,
                        "start_time_ms": trace["start_time_ms"],
                        "updated_at_ms": trace["updated_at_ms"],
                        "cursor_seq": trace["cursor_seq"],
                    }
                )
                if len(trace_copies) >= limit:
                    break

        result: list[TraceSummaryDict] = []
        for trace in trace_copies:
            normalized = self._normalize_trace(trace)
            if normalized is not None:
                result.append(normalized)
        return result

    def get_cursor_head(self) -> int:
        with self._lock:
            return self._next_cursor_seq - 1

    def _normalize_trace(self, trace: _TraceState) -> TraceSummaryDict | None:
        spans_by_id = trace["spans"]
        if not spans_by_id:
            return None

        children_by_parent: dict[str, list[str]] = {}
        for span_id, span in spans_by_id.items():
            parent_id = span["parent_span_id"]
            if parent_id is not None and parent_id in spans_by_id:
                children_by_parent.setdefault(parent_id, []).append(span_id)

        root_candidates = [
            span
            for span in spans_by_id.values()
            if span["parent_span_id"] is None or span["parent_span_id"] not in spans_by_id
        ]
        if not root_candidates:
            root_candidates = list(spans_by_id.values())
        root_span = min(root_candidates, key=lambda span: span["start_time_ms"])

        sorted_children: dict[str, list[str]] = {
            parent: sorted(child_ids, key=lambda cid: spans_by_id[cid]["start_time_ms"])
            for parent, child_ids in children_by_parent.items()
        }

        def build_span_tree(span_id: str) -> TraceSpanNodeDict:
            span = spans_by_id[span_id]
            child_ids = sorted_children.get(span_id, [])
            return {
                "span_id": span["span_id"],
                "name": span["name"],
                "start_time_ms": span["start_time_ms"],
                "duration_ms": span["duration_ms"],
                "running": span["running"],
                "tags": _collect_span_tags(span.get("tags", {})),
                "children": [build_span_tree(child_id) for child_id in child_ids],
            }

        start_time_ms = min(span["start_time_ms"] for span in spans_by_id.values())
        duration_ms = max(
            (span["start_time_ms"] + span["duration_ms"] - start_time_ms for span in spans_by_id.values()),
            default=0,
        )
        trace_tags = _collect_trace_tags([span.get("tags", {}) for span in spans_by_id.values()])

        return {
            "trace_id": trace["trace_id"],
            "start_time_ms": start_time_ms,
            "duration_ms": duration_ms,
            "entrypoint": trace["entrypoint"],
            "tags": trace_tags,
            "span_count": len(spans_by_id),
            "root_span": build_span_tree(root_span["span_id"]),
            "_cursor_seq": trace["cursor_seq"],
        }

    def _evict_old_traces(self) -> None:
        max_buffered_traces = _max_buffered_traces()
        if len(self._traces) <= max_buffered_traces:
            return

        sorted_ids = sorted(self._traces, key=lambda trace_id: self._traces[trace_id]["updated_at_ms"], reverse=True)
        keep_ids = set(sorted_ids[:max_buffered_traces])
        for trace_id in list(self._traces):
            if trace_id not in keep_ids:
                del self._traces[trace_id]


_TRACE_BUFFER = _LiveTraceBuffer()

_change_listeners: list[Callable[[], None]] = []


def register_change_listener(listener: Callable[[], None]) -> None:
    """
    Register a callback fired after any span start/finish is recorded.

    Used by the SSE layer to wake streaming subscribers. Kept dependency-free
    (no asyncio import here) so the buffer stays a plain in-memory store. The
    callback must be cheap and non-blocking; it must never raise. Registration
    is idempotent.
    """
    if listener not in _change_listeners:
        _change_listeners.append(listener)


def _emit_change() -> None:
    for listener in _change_listeners:
        try:
            listener()
        except Exception as exc:  # never let notification break span recording
            log.debug("Trace change listener raised: {}", exc)


def record_live_span_start(  # noqa: PLR0913
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    operation_name: str,
    start_time_ms: int,
    tags: Mapping[str, Any] | None = None,
) -> None:
    _TRACE_BUFFER.record_span_start(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        operation_name=operation_name,
        start_time_ms=start_time_ms,
        tags=tags,
    )
    _emit_change()


def record_live_span_finish(
    *,
    trace_id: str,
    span_id: str,
    duration_ms: int,
    tags: Mapping[str, Any] | None = None,
) -> None:
    _TRACE_BUFFER.record_span_finish(
        trace_id=trace_id,
        span_id=span_id,
        duration_ms=duration_ms,
        completed_at_ms=int(time() * 1000),
        tags=tags,
    )
    _emit_change()


def get_recent_traces(
    *,
    since_ms: int | None,
    limit: int,
    after_seq: int | None = None,
) -> list[TraceSummaryDict]:
    return _TRACE_BUFFER.get_recent_traces(since_ms=since_ms, limit=limit, after_seq=after_seq)


def get_trace_cursor_head() -> int:
    return _TRACE_BUFFER.get_cursor_head()


def clear_live_traces() -> None:
    _TRACE_BUFFER.clear()
