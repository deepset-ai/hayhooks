"""Strict codecs and public durable execution projections."""

from __future__ import annotations

from dataclasses import replace
from datetime import timezone

import pytest
from pydantic import BaseModel, HttpUrl

from hayhooks.durable.engine import ExecutionPayloadSizeError, ExecutionStatus, PayloadKind, ProgressEvent
from hayhooks.durable.models import (
    CheckpointEnvelope,
    ExecutionKind,
    ExecutionProgress,
    PersistedError,
    decode_json,
    encode_json,
    operation_fingerprint,
    project_execution,
)
from hayhooks.durable.store import StoredExecution
from tests.durable_store_contract import contract_control


def json_payload(value: object) -> bytes:
    return encode_json(value, max_bytes=4_096)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"question": "hello"}, id="input"),
        pytest.param(
            CheckpointEnvelope(
                adapter_kind=ExecutionKind.PIPELINE,
                adapter_checkpoint={"snapshot": [1, 2]},
                application_state={"step": 2},
                resume_input={"approved": True},
            ).model_dump(mode="json"),
            id="checkpoint",
        ),
        pytest.param({"answer": 42}, id="result"),
        pytest.param(PersistedError(type="ValueError", message="bad", code="E1").model_dump(mode="json"), id="error"),
        pytest.param(
            {"kind": "approval", "message": "Continue?", "expected_input_schema": {"type": "boolean"}},
            id="wait",
        ),
        pytest.param(
            ExecutionProgress(
                sequence=1,
                message="working",
                timestamp="2026-08-25T10:00:00Z",
                metadata={"percent": 50},
            ).model_dump(mode="json", exclude={"sequence"}),
            id="progress",
        ),
    ],
)
def test_payload_round_trip(value) -> None:
    assert decode_json(encode_json(value, max_bytes=4_096), max_bytes=4_096) == value


@pytest.mark.parametrize(
    "value",
    [object(), (1, 2), {1: "not-a-string-key"}, {"number": float("nan")}, {"number": float("inf")}],
)
def test_encoder_rejects_non_json_values(value) -> None:
    with pytest.raises(ValueError):
        encode_json(value, max_bytes=100)


@pytest.mark.parametrize(
    "payload",
    [b"\xff", b"{", b"NaN", b"1e999"],
)
def test_decoder_rejects_malformed_json(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_json(payload, max_bytes=100)


def test_codec_rejects_oversized_payloads() -> None:
    with pytest.raises(ExecutionPayloadSizeError):
        encode_json({"large": "value"}, max_bytes=4)
    with pytest.raises(ExecutionPayloadSizeError):
        decode_json(b'{"large":"value"}', max_bytes=4)


def test_operation_fingerprint_is_canonical_but_preserves_list_order() -> None:
    class SetInput(BaseModel):
        tags: set[str]
        steps: list[int]
        url: HttpUrl

    first = SetInput(tags={"zeta", "alpha"}, steps=[1, 2], url="https://example.com")
    second = SetInput(tags={"alpha", "zeta"}, steps=[1, 2], url="https://example.com")
    assert operation_fingerprint("jobs", "v1", "owner", first) == operation_fingerprint("jobs", "v1", "owner", second)
    assert operation_fingerprint("jobs", "v1", None, {"a": 1, "b": 2}) == operation_fingerprint(
        "jobs", "v1", None, {"b": 2, "a": 1}
    )
    assert operation_fingerprint("jobs", "v1", "owner", first) != operation_fingerprint(
        "jobs", "v1", "owner", SetInput(tags=first.tags, steps=[2, 1], url=first.url)
    )


def test_public_projection_excludes_private_data_and_allowlists_wait_fields() -> None:
    control = replace(
        contract_control("jobs"),
        status=ExecutionStatus.WAITING,
        version=4,
        run_attempt=2,
        cancel_requested_at_ms=1_500,
        created_at_ms=1_000,
        updated_at_ms=2_000,
    )
    stored = StoredExecution(
        control=control,
        payloads={
            PayloadKind.INPUT: json_payload({"private_input": "hidden"}),
            PayloadKind.CHECKPOINT: json_payload({"private_checkpoint": "hidden"}),
            PayloadKind.ERROR: json_payload(
                PersistedError(type="RetryError", message="retrying").model_dump(mode="json")
            ),
            PayloadKind.WAIT: json_payload(
                {
                    "kind": "approval",
                    "message": "Continue?",
                    "expected_input_schema": {"type": "boolean"},
                    "resume_payload": "hidden",
                }
            ),
        },
        progress=(
            ProgressEvent(
                7,
                json_payload(
                    {
                        "kind": "progress",
                        "message": "working",
                        "sequence": 999,
                        "timestamp": "2026-08-25T10:00:00Z",
                        "metadata": {"percent": 50},
                    }
                ),
            ),
        ),
    )

    public = project_execution(stored, links={"self": "/executions/run_1"})
    assert (public.execution_id, public.attempt, public.sequence) == ("run_1", 2, 4)
    assert public.waiting == {
        "kind": "approval",
        "message": "Continue?",
        "expected_input_schema": {"type": "boolean"},
    }
    assert public.progress[0].sequence == 7
    assert public.created_at.tzinfo is timezone.utc
    encoded = json_payload(public.model_dump(mode="json")).decode()
    for private in ("private_input", "private_checkpoint", "resume_payload", "owner", "lease", "fence"):
        assert private not in encoded


def test_projection_decodes_result_payload() -> None:
    control = replace(
        contract_control("jobs"),
        status=ExecutionStatus.COMPLETED,
        version=3,
        run_attempt=1,
        created_at_ms=1_000,
        updated_at_ms=2_000,
    )
    public = project_execution(
        StoredExecution(control, {PayloadKind.RESULT: json_payload({"answer": 42})}, ()),
    )
    assert public.result == {"answer": 42}


def test_error_messages_are_redacted_and_bounded() -> None:
    error = PersistedError(
        type="RuntimeError",
        message=(
            "Authorization: Bearer auth-token password=hunter2 "
            'https://example.test/?api_key=query-secret&token=other {"client_secret":"json-secret"}'
        ),
    )
    for secret in ("auth-token", "hunter2", "query-secret", "other", "json-secret"):
        assert secret not in error.message
    assert "<redacted>" in error.message
    assert PersistedError(type="Error", message="password=direct-secret").message == "password=<redacted>"

    bounded = PersistedError(type="RuntimeError", message="💥" * 2_000)
    assert len(bounded.message) <= 2_000
    assert len(bounded.message.encode()) <= 4_096
