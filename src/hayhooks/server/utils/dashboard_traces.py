from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
import json
from typing import Any

import requests
from requests import RequestException

_TAG_PRIORITY = (
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
    "service.name",
    "serviceName",
)
_IGNORED_TAG_KEYS = {"hayhooks.elapsed_ms"}
_MAX_TRACE_TAGS = 32
_MAX_SPAN_TAGS = 8
_MAX_TAG_VALUE_CHARS = 220
_SIGNOZ_CORE_FIELDS = {
    "traceID",
    "traceId",
    "spanID",
    "spanId",
    "parentSpanID",
    "parentSpanId",
    "name",
    "operation",
    "timestamp",
    "durationNano",
    "serviceName",
}


class TraceBackendError(RuntimeError):
    """Raised when trace backend data cannot be retrieved or parsed."""


def _tag_map(span: Mapping[str, Any]) -> dict[str, Any]:
    tags = span.get("tags", [])
    if not isinstance(tags, list):
        return {}
    result: dict[str, Any] = {}
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        key = tag.get("key")
        if key is None:
            continue
        result[str(key)] = tag.get("value")
    return result


def _parent_span_id(span: Mapping[str, Any]) -> str | None:
    references = span.get("references", [])
    if not isinstance(references, list):
        return None
    for reference in references:
        if not isinstance(reference, dict):
            continue
        if reference.get("refType") == "CHILD_OF" and reference.get("spanID") is not None:
            return str(reference["spanID"])
    return None


def _to_ms(value_us: Any) -> int:
    try:
        return int(int(value_us) / 1000)
    except (TypeError, ValueError):
        return 0


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
    if len(text) > _MAX_TAG_VALUE_CHARS:
        text = f"{text[: _MAX_TAG_VALUE_CHARS - 3]}..."
    return text if text else None


def _collect_span_tags(span_tags: dict[str, Any]) -> list[dict[str, str]]:
    """Collect a small curated set of tags for a single span."""
    tags: list[dict[str, str]] = []
    seen: set[str] = set()
    for key in _TAG_PRIORITY:
        if key in span_tags:
            value_text = _stringify_tag_value(span_tags[key])
            if value_text is not None:
                tags.append({"key": key, "value": value_text})
                seen.add(key)
        if len(tags) >= _MAX_SPAN_TAGS:
            return tags
    for key, value in span_tags.items():
        if key in seen or key in _IGNORED_TAG_KEYS:
            continue
        value_text = _stringify_tag_value(value)
        if value_text is not None:
            tags.append({"key": key, "value": value_text})
        if len(tags) >= _MAX_SPAN_TAGS:
            break
    return tags


def _collect_trace_tags_from_span_tags(span_tag_maps: list[dict[str, Any]]) -> list[dict[str, str]]:
    tags: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    def add_tag(key: str, value: Any) -> None:
        if key in seen_keys or key in _IGNORED_TAG_KEYS:
            return
        value_text = _stringify_tag_value(value)
        if value_text is None:
            return
        tags.append({"key": key, "value": value_text})
        seen_keys.add(key)

    for key in _TAG_PRIORITY:
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


def _extract_signoz_span_tags(row_data: Mapping[str, Any]) -> dict[str, Any]:
    span_tags: dict[str, Any] = {}
    for key, value in row_data.items():
        if key in _SIGNOZ_CORE_FIELDS or key.startswith("_"):
            continue
        if value in (None, ""):
            continue

        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                if nested_value in (None, ""):
                    continue
                span_tags[f"{key}.{nested_key}"] = nested_value
            continue

        span_tags[key] = value

    return span_tags


def _build_span_tree(
    *,
    span_id: str,
    spans_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str, list[str]],
) -> dict[str, Any]:
    span = spans_by_id[span_id]
    children = sorted(children_by_parent.get(span_id, []), key=lambda child_id: spans_by_id[child_id].get("startTime", 0))
    return {
        "span_id": span_id,
        "name": str(span.get("operationName", "")),
        "start_time_ms": _to_ms(span.get("startTime", 0)),
        "duration_ms": _to_ms(span.get("duration", 0)),
        "tags": _collect_span_tags(_tag_map(span)),
        "children": [
            _build_span_tree(span_id=child_id, spans_by_id=spans_by_id, children_by_parent=children_by_parent)
            for child_id in children
        ],
    }


def _find_root_span_id(spans_by_id: dict[str, dict[str, Any]]) -> str:
    roots = [span_id for span_id, span in spans_by_id.items() if _parent_span_id(span) is None]
    candidates = roots or list(spans_by_id)
    return min(candidates, key=lambda span_id: spans_by_id[span_id].get("startTime", 0))


def normalize_jaeger_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    spans = trace.get("spans", [])
    if not isinstance(spans, list) or len(spans) == 0:
        msg = "Jaeger trace payload does not contain spans"
        raise TraceBackendError(msg)

    spans_by_id: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    span_tag_maps: list[dict[str, Any]] = []
    entrypoint = None

    for span in spans:
        if not isinstance(span, dict):
            continue
        span_id = span.get("spanID")
        if span_id is None:
            continue
        span_id_str = str(span_id)
        spans_by_id[span_id_str] = span
        parent_span_id = _parent_span_id(span)
        if parent_span_id is not None:
            children_by_parent[parent_span_id].append(span_id_str)

        tags = _tag_map(span)
        span_tag_maps.append(tags)
        if entrypoint is None and (pipeline_name := tags.get("hayhooks.pipeline.name")) is not None:
            entrypoint = str(pipeline_name)

    if not spans_by_id:
        msg = "Jaeger trace payload does not contain valid spans"
        raise TraceBackendError(msg)

    root_span_id = _find_root_span_id(spans_by_id)
    root_span = spans_by_id[root_span_id]
    if entrypoint is None:
        entrypoint = _tag_map(root_span).get("hayhooks.pipeline.name")
        if entrypoint is not None:
            entrypoint = str(entrypoint)
    trace_tags = _collect_trace_tags_from_span_tags(span_tag_maps)

    return {
        "trace_id": str(trace.get("traceID", "")),
        "start_time_ms": _to_ms(root_span.get("startTime", 0)),
        "duration_ms": _to_ms(root_span.get("duration", 0)),
        "entrypoint": entrypoint,
        "tags": trace_tags,
        "span_count": len(spans_by_id),
        "root_span": _build_span_tree(
            span_id=root_span_id,
            spans_by_id=spans_by_id,
            children_by_parent=children_by_parent,
        ),
    }


def fetch_jaeger_traces(
    *,
    backend_url: str,
    service_name: str,
    start_time_us: int,
    end_time_us: int,
    limit: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    url = f"{backend_url.rstrip('/')}/api/traces"
    params = {
        "service": service_name,
        "start": start_time_us,
        "end": end_time_us,
        "limit": limit,
    }
    try:
        response = requests.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except (RequestException, ValueError) as exc:
        msg = f"Failed to fetch traces from Jaeger: {exc!s}"
        raise TraceBackendError(msg) from exc

    data = payload.get("data")
    if not isinstance(data, list):
        msg = "Jaeger response does not include a 'data' list"
        raise TraceBackendError(msg)

    return [trace for trace in data if isinstance(trace, dict)]


def _to_ms_from_nano(value_nano: Any) -> int:
    try:
        return int(int(value_nano) / 1_000_000)
    except (TypeError, ValueError):
        return 0


def _to_ms_from_signoz_timestamp(value: Any) -> int:
    if isinstance(value, int | float):
        if value > 1_000_000_000_000_000:
            return int(value / 1_000_000)
        if value > 1_000_000_000_000:
            return int(value)
        return int(value * 1000)

    if isinstance(value, str):
        parsed_value = value
        if parsed_value.endswith("Z"):
            parsed_value = parsed_value[:-1] + "+00:00"
        try:
            return int(datetime.fromisoformat(parsed_value).timestamp() * 1000)
        except ValueError:
            return 0

    return 0


def _extract_signoz_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    data = payload.get("data")
    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            rows.extend([row for row in data["rows"] if isinstance(row, dict)])

        if isinstance(data.get("results"), list):
            for result in data["results"]:
                if not isinstance(result, dict):
                    continue
                if isinstance(result.get("rows"), list):
                    rows.extend([row for row in result["rows"] if isinstance(row, dict)])

    elif isinstance(data, list):
        for result in data:
            if not isinstance(result, dict):
                continue
            if isinstance(result.get("rows"), list):
                rows.extend([row for row in result["rows"] if isinstance(row, dict)])

    return rows


def fetch_signoz_traces(
    *,
    backend_url: str,
    service_name: str,
    start_time_ms: int,
    end_time_ms: int,
    limit: int,
    timeout_seconds: float,
    api_key: str,
) -> list[dict[str, Any]]:
    url = f"{backend_url.rstrip('/')}/api/v5/query_range"
    payload = {
        "start": start_time_ms,
        "end": end_time_ms,
        "requestType": "raw",
        "variables": {},
        "compositeQuery": {
            "queries": [
                {
                    "type": "builder_query",
                    "spec": {
                        "name": "A",
                        "signal": "traces",
                        "filter": {"expression": f"serviceName = '{service_name}'"},
                        "selectFields": [
                            {"name": "traceID"},
                            {"name": "spanID"},
                            {"name": "parentSpanID"},
                            {"name": "name"},
                            {"name": "timestamp"},
                            {"name": "durationNano"},
                            {"name": "serviceName"},
                            {"name": "hayhooks.pipeline.name"},
                        ],
                        "order": [{"key": {"name": "timestamp"}, "direction": "desc"}],
                        "limit": limit,
                        "offset": 0,
                        "disabled": False,
                    },
                }
            ]
        },
    }
    headers = {"SIGNOZ-API-KEY": api_key} if api_key else {}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
        if response.status_code == 401:
            if api_key:
                msg = "Failed to fetch traces from SigNoz: API key is invalid or unauthorized"
            else:
                msg = (
                    "Failed to fetch traces from SigNoz: API key required. "
                    "Set HAYHOOKS_DASHBOARD_TRACE_SIGNOZ_API_KEY."
                )
            raise TraceBackendError(msg)

        response.raise_for_status()
        payload = response.json()
    except (RequestException, ValueError) as exc:
        msg = f"Failed to fetch traces from SigNoz: {exc!s}"
        raise TraceBackendError(msg) from exc

    if isinstance(payload, dict) and payload.get("status") == "error":
        error = payload.get("error", {})
        if isinstance(error, dict):
            error_message = error.get("message")
            if isinstance(error_message, str) and error_message:
                msg = f"Failed to fetch traces from SigNoz: {error_message}"
                raise TraceBackendError(msg)

    return _extract_signoz_rows(payload if isinstance(payload, Mapping) else {})


def normalize_signoz_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}

    for row in rows:
        row_data = row.get("data", row)
        if not isinstance(row_data, dict):
            continue

        trace_id = row_data.get("traceID") or row_data.get("traceId")
        span_id = row_data.get("spanID") or row_data.get("spanId")
        if trace_id is None or span_id is None:
            continue

        trace_id_str = str(trace_id)
        span_id_str = str(span_id)
        parent_span_raw = row_data.get("parentSpanID") or row_data.get("parentSpanId")
        parent_span_id = str(parent_span_raw) if parent_span_raw not in (None, "") else None
        start_time_ms = _to_ms_from_signoz_timestamp(row_data.get("timestamp", row.get("timestamp")))
        duration_ms = _to_ms_from_nano(row_data.get("durationNano"))
        operation_name = str(row_data.get("name", row_data.get("operation", "")))
        span_tags = _extract_signoz_span_tags(row_data)

        trace = traces.setdefault(
            trace_id_str,
            {
                "trace_id": trace_id_str,
                "entrypoint": None,
                "spans": {},
            },
        )

        if trace["entrypoint"] is None:
            trace["entrypoint"] = row_data.get("hayhooks.pipeline.name")

        trace["spans"][span_id_str] = {
            "span_id": span_id_str,
            "parent_span_id": parent_span_id,
            "name": operation_name,
            "start_time_ms": start_time_ms,
            "duration_ms": duration_ms,
            "tags": span_tags,
        }

    traces_data: list[dict[str, Any]] = []
    for trace in traces.values():
        spans_by_id: dict[str, dict[str, Any]] = trace["spans"]
        if not spans_by_id:
            continue

        root_candidates = [
            span
            for span in spans_by_id.values()
            if span["parent_span_id"] is None or span["parent_span_id"] not in spans_by_id
        ]
        if not root_candidates:
            root_candidates = list(spans_by_id.values())
        root_span = min(root_candidates, key=lambda span: span["start_time_ms"])

        def _build_signoz_tree(span_id: str) -> dict[str, Any]:
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
                "children": [_build_signoz_tree(child_id) for child_id in child_ids],
            }

        start_time_ms = min(span["start_time_ms"] for span in spans_by_id.values())
        duration_ms = max(
            (span["start_time_ms"] + span["duration_ms"] - start_time_ms for span in spans_by_id.values()),
            default=0,
        )
        trace_tags = _collect_trace_tags_from_span_tags([span.get("tags", {}) for span in spans_by_id.values()])

        traces_data.append(
            {
                "trace_id": trace["trace_id"],
                "start_time_ms": start_time_ms,
                "duration_ms": duration_ms,
                "entrypoint": trace["entrypoint"],
                "tags": trace_tags,
                "span_count": len(spans_by_id),
                "root_span": _build_signoz_tree(root_span["span_id"]),
            }
        )

    traces_data.sort(key=lambda trace: trace["start_time_ms"], reverse=True)
    return traces_data
