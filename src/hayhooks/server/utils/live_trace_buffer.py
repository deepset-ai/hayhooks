from __future__ import annotations

from collections.abc import Mapping
import json
from threading import RLock
from time import time
from typing import Any

_MAX_BUFFERED_TRACES = 200
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
    "hayhooks.http.status_code",
)
_IGNORED_TRACE_TAG_KEYS = {"hayhooks.elapsed_ms"}
_MAX_TRACE_TAGS = 32
_MAX_SPAN_TAGS = 8


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
    return text if text else None


def _collect_span_tags(span_tags: dict[str, Any]) -> list[dict[str, str]]:
    """Collect a small curated set of tags for a single span."""
    tags: list[dict[str, str]] = []
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


def _collect_trace_tags(span_tag_maps: list[dict[str, Any]]) -> list[dict[str, str]]:
    tags: list[dict[str, str]] = []
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
        self._traces: dict[str, dict[str, Any]] = {}

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()

    def record_span_start(
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
            trace = self._traces.get(trace_id)
            if trace is None:
                trace = {
                    "trace_id": trace_id,
                    "entrypoint": None,
                    "spans": {},
                    "start_time_ms": start_time_ms,
                    "updated_at_ms": start_time_ms,
                }
                self._traces[trace_id] = trace

            trace["updated_at_ms"] = start_time_ms
            trace["start_time_ms"] = min(trace["start_time_ms"], start_time_ms)
            span_tags = dict(tags or {})
            if trace["entrypoint"] is None and (entrypoint := span_tags.get("hayhooks.pipeline.name")) is not None:
                trace["entrypoint"] = str(entrypoint)

            trace["spans"][span_id] = {
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "name": operation_name,
                "start_time_ms": start_time_ms,
                "duration_ms": 0,
                "tags": span_tags,
            }

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
            trace = self._traces.get(trace_id)
            if trace is None:
                return

            span = trace["spans"].get(span_id)
            if span is None:
                return

            span["duration_ms"] = max(0, duration_ms)
            if tags:
                span["tags"].update(tags)
                if trace["entrypoint"] is None and (entrypoint := span["tags"].get("hayhooks.pipeline.name")) is not None:
                    trace["entrypoint"] = str(entrypoint)
            trace["updated_at_ms"] = completed_at_ms

    def get_recent_traces(self, *, since_ms: int | None, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            traces = sorted(self._traces.values(), key=lambda trace: trace["updated_at_ms"], reverse=True)
            if since_ms is not None:
                traces = [trace for trace in traces if trace["start_time_ms"] >= since_ms]

            result: list[dict[str, Any]] = []
            for trace in traces:
                normalized = self._normalize_trace(trace)
                if normalized is not None:
                    result.append(normalized)
                if len(result) >= limit:
                    break
            return result

    def _normalize_trace(self, trace: Mapping[str, Any]) -> dict[str, Any] | None:
        spans_by_id: dict[str, dict[str, Any]] = trace["spans"]
        if not spans_by_id:
            return None

        root_candidates = [
            span
            for span in spans_by_id.values()
            if span["parent_span_id"] is None or span["parent_span_id"] not in spans_by_id
        ]
        if not root_candidates:
            root_candidates = list(spans_by_id.values())
        root_span = min(root_candidates, key=lambda span: span["start_time_ms"])

        def build_span_tree(span_id: str) -> dict[str, Any]:
            span = spans_by_id[span_id]
            child_ids = sorted(
                [child["span_id"] for child in spans_by_id.values() if child["parent_span_id"] == span_id],
                key=lambda child_id: spans_by_id[child_id]["start_time_ms"],
            )
            return {
                "span_id": span["span_id"],
                "name": span["name"],
                "start_time_ms": span["start_time_ms"],
                "duration_ms": span["duration_ms"],
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
        }

    def _evict_old_traces(self) -> None:
        if len(self._traces) <= _MAX_BUFFERED_TRACES:
            return

        sorted_ids = sorted(self._traces, key=lambda trace_id: self._traces[trace_id]["updated_at_ms"], reverse=True)
        keep_ids = set(sorted_ids[:_MAX_BUFFERED_TRACES])
        for trace_id in list(self._traces):
            if trace_id not in keep_ids:
                del self._traces[trace_id]


_TRACE_BUFFER = _LiveTraceBuffer()


def record_live_span_start(
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


def get_recent_traces(*, since_ms: int | None, limit: int) -> list[dict[str, Any]]:
    return _TRACE_BUFFER.get_recent_traces(since_ms=since_ms, limit=limit)


def clear_live_traces() -> None:
    _TRACE_BUFFER.clear()
