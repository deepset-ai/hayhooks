"""Immutable durable payload models and strict JSON codecs."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import to_jsonable_python
from typing_extensions import TypeAliasType

from hayhooks.durable.engine import (
    MAX_CONTROL_SCALAR_BYTES,
    ExecutionPayloadSizeError,
    ExecutionStatus,
    PayloadKind,
    normalize_cancellation_reason,
)
from hayhooks.durable.store import StoredExecution

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue = TypeAliasType("JsonValue", JsonScalar | list["JsonValue"] | dict[str, "JsonValue"])
DEFAULT_MAX_JSON_BYTES = 1_000_000
_SENSITIVE_NAME = (
    r"(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|authorization|bearer|password|passwd|secret|"
    r"client[_ -]?secret)"
)
_REDACTIONS = (
    (re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;&]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1<redacted>"),
    (re.compile(rf'(?i)(["\']{_SENSITIVE_NAME}["\']\s*:\s*)["\'][^"\']*["\']'), r'\1"<redacted>"'),
    (re.compile(rf"(?i)({_SENSITIVE_NAME})\s*[:=]\s*[^\s,;&]+"), r"\1=<redacted>"),
)


class ExecutionKind(str, Enum):
    PIPELINE = "pipeline"
    AGENT = "agent"


class CheckpointEnvelope(BaseModel):
    """Private recovery payload; lifecycle state remains in ``ExecutionControl``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    adapter_kind: ExecutionKind
    adapter_checkpoint: JsonValue
    application_state: dict[str, JsonValue] = Field(default_factory=dict)
    resume_input: JsonValue = None


class PersistedError(BaseModel):
    """Bounded, payload-safe failure metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str = Field(min_length=1)
    message: str
    retryable: bool = False
    code: str | None = None

    @field_validator("type", "code", mode="before")
    @classmethod
    def _bound_scalar(cls, value: object) -> str | None:
        if value is None:
            return None
        return str(value).encode()[:MAX_CONTROL_SCALAR_BYTES].decode(errors="ignore")

    @field_validator("message", mode="before")
    @classmethod
    def _bound_message(cls, value: object) -> str:
        message = str(value)
        for pattern, replacement in _REDACTIONS:
            message = pattern.sub(replacement, message)
        return normalize_cancellation_reason(message) or ""


class ExecutionProgress(BaseModel):
    """One immutable client-visible progress event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=1)
    kind: str = "progress"
    message: str
    timestamp: datetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class ExecutionResult(BaseModel):
    """Public execution projection with no private store or lease data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str
    status: ExecutionStatus
    attempt: int
    sequence: int
    progress: tuple[ExecutionProgress, ...] = ()
    result: JsonValue = None
    error: PersistedError | None = None
    waiting: dict[str, JsonValue] | None = None
    cancellation_requested_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    links: dict[str, str] = Field(default_factory=dict)


def encode_json(value: object, *, max_bytes: int, canonical: bool = False) -> bytes:
    """Encode a strict JSON value as bounded UTF-8."""
    _validate_json_value(value)
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            sort_keys=canonical,
        ).encode()
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("value must be valid JSON") from error
    _check_size(payload, max_bytes)
    return payload


def decode_json(payload: bytes, *, max_bytes: int) -> JsonValue:
    """Decode bounded UTF-8 JSON, rejecting non-finite constants."""
    _check_size(payload, max_bytes)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=float,
        )
        _validate_json_value(value)
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError, RecursionError) as error:
        raise ValueError("payload must contain valid UTF-8 JSON") from error
    return cast(JsonValue, value)


def operation_fingerprint(
    deployment: str,
    revision: str,
    owner_id: str | None,
    validated_input: object,
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> str:
    """Hash the operation scope and canonical validated input."""
    value = {
        "deployment": deployment,
        "revision": revision,
        "owner_id": owner_id,
        "input": _canonical_json(validated_input, max_bytes=max_bytes),
    }
    return hashlib.sha256(encode_json(value, max_bytes=max_bytes, canonical=True)).hexdigest()


def project_execution(
    stored: StoredExecution,
    *,
    links: Mapping[str, str] | None = None,
    max_payload_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> ExecutionResult:
    """Decode the public subset of one stored execution snapshot."""
    payloads = stored.payloads
    result = (
        decode_json(payloads[PayloadKind.RESULT], max_bytes=max_payload_bytes)
        if PayloadKind.RESULT in payloads
        else None
    )
    error_payload = payloads.get(PayloadKind.ERROR)
    error = (
        PersistedError.model_validate(decode_json(error_payload, max_bytes=max_payload_bytes))
        if error_payload is not None
        else None
    )
    wait = (
        decode_json(payloads[PayloadKind.WAIT], max_bytes=max_payload_bytes) if PayloadKind.WAIT in payloads else None
    )
    if wait is not None and not isinstance(wait, dict):
        raise ValueError("wait payload must be a JSON object")
    waiting = (
        {key: wait[key] for key in ("kind", "message", "expected_input_schema") if key in wait}
        if wait is not None
        else None
    )
    progress = []
    for event in stored.progress:
        value = decode_json(event.data, max_bytes=max_payload_bytes)
        if not isinstance(value, dict):
            raise ValueError("progress payload must be a JSON object")
        progress.append(ExecutionProgress.model_validate({**value, "sequence": event.sequence}))
    control = stored.control
    return ExecutionResult(
        execution_id=control.run_id,
        status=control.status,
        attempt=control.run_attempt,
        sequence=control.version,
        progress=tuple(progress),
        result=result,
        error=error,
        waiting=waiting,
        cancellation_requested_at=(
            datetime.fromtimestamp(control.cancel_requested_at_ms / 1_000, tz=timezone.utc)
            if control.cancel_requested_at_ms is not None
            else None
        ),
        created_at=datetime.fromtimestamp(control.created_at_ms / 1_000, tz=timezone.utc),
        updated_at=datetime.fromtimestamp(control.updated_at_ms / 1_000, tz=timezone.utc),
        links=dict(links or {}),
    )


def _canonical_json(value: object, *, max_bytes: int) -> JsonValue:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    elif isinstance(value, Enum):
        value = value.value
    if value is None or type(value) in (str, int, float, bool):
        _validate_json_value(value)
        return cast(JsonScalar, value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        mapping = cast(Mapping[str, object], value)
        return {key: _canonical_json(item, max_bytes=max_bytes) for key, item in mapping.items()}
    if isinstance(value, list | tuple):
        return [_canonical_json(item, max_bytes=max_bytes) for item in value]
    if isinstance(value, set | frozenset):
        items = [_canonical_json(item, max_bytes=max_bytes) for item in value]
        return sorted(items, key=lambda item: encode_json(item, max_bytes=max_bytes, canonical=True))
    try:
        serialized = to_jsonable_python(value)
    except (TypeError, ValueError):
        raise ValueError(f"{type(value).__name__} is not JSON serializable") from None
    if serialized is value:
        raise ValueError(f"{type(value).__name__} is not JSON serializable")
    return _canonical_json(serialized, max_bytes=max_bytes)


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError("JSON numbers must be finite")
    if type(value) not in (list, dict):
        raise ValueError(f"{type(value).__name__} is not a JSON value")
    if type(value) is dict and not all(type(key) is str for key in value):
        raise ValueError("JSON object keys must be strings")
    try:
        items = value if isinstance(value, list) else cast(dict[str, object], value).values()
        for item in items:
            _validate_json_value(item)
    except RecursionError as error:
        raise ValueError("value must not contain recursive JSON data") from error


def _check_size(payload: bytes, max_bytes: int) -> None:
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if len(payload) > max_bytes:
        raise ExecutionPayloadSizeError("JSON payload exceeds its configured byte limit")
