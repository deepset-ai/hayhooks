"""Backend-neutral durable-store policy and contract."""
# ruff: noqa: EM101, EM102

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

from hayhooks.durable.engine import (
    Checkpoint,
    Complete,
    ExecutionCommand,
    ExecutionControl,
    ExecutionPayloadSizeError,
    Fail,
    Heartbeat,
    PayloadKind,
    RequestCancellation,
    Resume,
    ScheduleRetry,
    Suspend,
    TransitionPlan,
    validate_run_id,
)
from hayhooks.durable.models import ExecutionAdmissionError, ExecutionStoreError

MAINTENANCE_BATCH_SIZE = 100
DEFAULT_TRANSACTION_MAX_RETRIES = 8
DEFAULT_TRANSACTION_BACKOFF_MAX_MS = 25


class ExecutionStoreCorruptionError(ExecutionStoreError):
    """Persisted backend state cannot be safely decoded as durable control data."""


class ExecutionContentionError(ExecutionStoreError):
    """A bounded optimistic transaction could not obtain a stable snapshot."""


class ExecutionIdempotencyConflictError(RuntimeError):
    """A logical idempotency key was reused for a different request binding."""


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    created: bool
    control: ExecutionControl


@dataclass(frozen=True, slots=True)
class ExecutionStoreConfig:
    key_prefix: str = "hayhooks:durable"
    transaction_max_retries: int = DEFAULT_TRANSACTION_MAX_RETRIES
    transaction_backoff_max_ms: int = DEFAULT_TRANSACTION_BACKOFF_MAX_MS
    lease_commit_safety_ms: int = 50
    terminal_ttl_seconds: int = 604_800
    max_nonterminal_executions: int = 0
    max_input_bytes: int = 256_000
    max_checkpoint_bytes: int = 512_000
    max_result_bytes: int = 512_000
    max_error_bytes: int = 64_000
    max_wait_bytes: int = 64_000
    max_progress_events: int = 100
    max_progress_event_bytes: int = 8_192

    def __post_init__(self) -> None:
        for name in (
            "transaction_max_retries",
            "terminal_ttl_seconds",
            "max_input_bytes",
            "max_checkpoint_bytes",
            "max_result_bytes",
            "max_error_bytes",
            "max_wait_bytes",
            "max_progress_events",
            "max_progress_event_bytes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if min(self.transaction_backoff_max_ms, self.lease_commit_safety_ms, self.max_nonterminal_executions) < 0:
            raise ValueError("durable limits cannot be negative")


class ExecutionBackend(Protocol):
    """Internal operations needed by the durable adapter."""

    config: ExecutionStoreConfig
    deployment: str

    async def initialize(self) -> None: ...

    async def submit(
        self, control: ExecutionControl, input_payload: bytes, *, binding_digest: str
    ) -> SubmissionResult: ...

    async def get(self, run_id: str) -> ExecutionControl | None: ...

    async def read_payloads(self, run_id: str, kinds: tuple[PayloadKind, ...]) -> dict[PayloadKind, bytes | None]: ...

    async def read_progress(self, run_id: str) -> list[bytes]: ...

    async def transition(
        self, run_id: str, command: ExecutionCommand, *, candidate: bool = False
    ) -> TransitionPlan: ...

    async def read_candidate(self) -> str | None: ...

    async def maintain(self, command_factory: Callable[[int, int], ExecutionCommand]) -> int: ...

    async def operational_counts(self) -> dict[str, int]: ...


def parse_idempotency_binding(value: str) -> tuple[str, str]:
    """Decode the execution ID and request digest stored for an idempotency key."""
    run_id, separator, binding = value.partition("|")
    validate_run_id(run_id)
    if not separator or not binding:
        raise ValueError("idempotency binding is invalid")
    return run_id, binding


def parse_lease_member(value: str) -> tuple[str, int]:
    """Decode the execution ID and fence stored in the lease-expiry index."""
    run_id, separator, raw_fence = value.rpartition("|")
    validate_run_id(run_id)
    if not separator:
        raise ValueError("lease index member is invalid")
    fence = int(raw_fence)
    if fence < 0:
        raise ValueError("lease index member has a negative fence")
    return run_id, fence


def bind_command(command: ExecutionCommand, *, now_ms: int, lease_commit_safety_ms: int) -> ExecutionCommand:
    """Apply the backend clock and lease safety policy before reduction."""
    if isinstance(command, (Heartbeat, Checkpoint, ScheduleRetry, Suspend, Complete, Fail)):
        return replace(command, now_ms=now_ms, lease_commit_safety_ms=lease_commit_safety_ms)
    return replace(command, now_ms=now_ms)


def check_admission(raw: Mapping[Any, Any], config: ExecutionStoreConfig) -> None:
    """Validate the optional single-deployment nonterminal cap."""
    values = {key.decode("utf-8") if isinstance(key, bytes) else str(key): int(value) for key, value in raw.items()}
    nonterminal = values.get("nonterminal", 0)
    if nonterminal < 0:
        raise ExecutionStoreCorruptionError("capacity contains a negative counter")
    if config.max_nonterminal_executions and nonterminal >= config.max_nonterminal_executions:
        raise ExecutionAdmissionError("deployment_nonterminal")


def validate_command_payloads(command: ExecutionCommand, config: ExecutionStoreConfig) -> None:
    """Reject command payloads before either backend attempts a transition."""
    checks: tuple[tuple[str, bytes | None, int], ...] = ()
    if isinstance(command, Checkpoint):
        checks = (
            ("checkpoint", command.payload, config.max_checkpoint_bytes),
            *(("progress", event, config.max_progress_event_bytes) for event in command.progress_events),
        )
    elif isinstance(command, Suspend):
        checks = (
            ("checkpoint", command.checkpoint, config.max_checkpoint_bytes),
            ("wait", command.wait, config.max_wait_bytes),
            *(("progress", event, config.max_progress_event_bytes) for event in command.progress_events),
        )
    elif isinstance(command, Resume):
        checks = (
            ("checkpoint", command.checkpoint, config.max_checkpoint_bytes),
            *(("progress", event, config.max_progress_event_bytes) for event in command.progress_events),
        )
    elif isinstance(command, RequestCancellation):
        checks = tuple(("progress", event, config.max_progress_event_bytes) for event in command.progress_events)
    elif isinstance(command, Complete):
        checks = (
            ("result", command.result, config.max_result_bytes),
            *(("progress", event, config.max_progress_event_bytes) for event in command.progress_events),
        )
    elif isinstance(command, Fail):
        checks = (
            ("error", command.error, config.max_error_bytes),
            *(("progress", event, config.max_progress_event_bytes) for event in command.progress_events),
        )
    elif isinstance(command, ScheduleRetry):
        checks = (("error", command.error, config.max_error_bytes),)
    for label, payload, limit in checks:
        if payload is not None and len(payload) > limit:
            raise ExecutionPayloadSizeError(f"{label} payload exceeds its configured byte limit")


def bind_progress_sequences(plan: TransitionPlan, config: ExecutionStoreConfig) -> TransitionPlan:
    """Persist reducer-assigned progress sequence numbers inside event payloads."""
    if not plan.progress_events:
        return plan
    events = []
    for event in plan.progress_events:
        try:
            value = json.loads(event.data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ExecutionPayloadSizeError("progress payload is not valid JSON") from error
        if not isinstance(value, dict):
            raise ExecutionPayloadSizeError("progress payload must be a JSON object")
        value["sequence"] = event.sequence
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > config.max_progress_event_bytes:
            raise ExecutionPayloadSizeError("progress payload exceeds its configured byte limit")
        events.append(replace(event, data=encoded))
    return replace(plan, progress_events=tuple(events))
