"""Volatile in-memory durable store used by local development and tests."""

from __future__ import annotations

import asyncio
import copy
import uuid
from collections import deque
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from hayhooks.durable.context import RESUME_INPUT_KEY, ExecutionClaim
from hayhooks.durable.models import (
    DEFAULT_MAX_PROGRESS_EVENTS,
    ExecutionError,
    ExecutionLeaseLostError,
    ExecutionRecord,
    ExecutionStatus,
    JsonValue,
    utc_now,
    validate_json,
)


class InMemoryExecutionStore:
    """Volatile deterministic store for development and tests only."""

    volatile = True

    def __init__(
        self, *, max_progress_events: int = DEFAULT_MAX_PROGRESS_EVENTS, max_records: int | None = None
    ) -> None:
        self.max_progress_events = max(1, max_progress_events)
        self.max_records = max_records
        self._records: dict[str, ExecutionRecord] = {}
        self._queued: deque[str] = deque()
        self._queued_ids: set[str] = set()
        self._claims: dict[str, _InMemoryExecutionClaim] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        async with self._lock:
            self._initialized = True
            self._closed = False
            for execution_id, record in self._records.items():
                if (
                    not record.terminal
                    and record.status is not ExecutionStatus.WAITING
                    and execution_id not in self._claims
                ):
                    self._enqueue_locked(execution_id)

    async def submit(self, record: ExecutionRecord) -> bool:
        async with self._lock:
            if self._closed:
                msg = "durable execution store is closed"
                raise RuntimeError(msg)
            if record.execution_id in self._records:
                return False
            record.max_progress_events = self.max_progress_events
            record.to_json()
            self._records[record.execution_id] = _copy_record(record)
            self._enqueue_locked(record.execution_id)
            self._evict_locked()
            return True

    async def get(self, execution_id: str) -> ExecutionRecord | None:
        async with self._lock:
            record = self._records.get(execution_id)
            return _copy_record(record) if record else None

    async def get_many(self, execution_ids: list[str]) -> dict[str, ExecutionRecord | None]:
        async with self._lock:
            return {
                execution_id: (_copy_record(record) if (record := self._records.get(execution_id)) else None)
                for execution_id in execution_ids
            }

    async def get_changed(self, known_sequences: Mapping[str, int]) -> dict[str, ExecutionRecord | None]:
        async with self._lock:
            changed: dict[str, ExecutionRecord | None] = {}
            for execution_id, known_sequence in known_sequences.items():
                record = self._records.get(execution_id)
                if record is None or record.sequence != known_sequence:
                    changed[execution_id] = _copy_record(record) if record else None
            return changed

    async def claim_next(self, worker_name: str) -> ExecutionClaim | None:
        async with self._lock:
            if self._closed or not self._initialized:
                return None
            now = utc_now()
            attempts = len(self._queued)
            for _ in range(attempts):
                execution_id = self._queued.popleft()
                self._queued_ids.discard(execution_id)
                record = self._records.get(execution_id)
                if (
                    record is None
                    or record.terminal
                    or record.status is ExecutionStatus.WAITING
                    or execution_id in self._claims
                ):
                    continue
                if record.retry_at and record.retry_at > now:
                    self._enqueue_locked(execution_id)
                    continue
                record = _copy_record(record)
                record.status = ExecutionStatus.RUNNING
                record.attempt += 1
                if record.error is not None:
                    record.last_retry_error = record.error
                    record.error = None
                record.retry_at = None
                record.touch()
                claim = _InMemoryExecutionClaim(self, record, f"{worker_name}:{uuid.uuid4().hex}")
                self._claims[execution_id] = claim
                self._records[execution_id] = _copy_record(record)
                return claim
            return None

    async def request_cancel(self, execution_id: str, reason: str | None = None) -> bool:
        async with self._lock:
            persisted = self._records.get(execution_id)
            if persisted is None or persisted.terminal:
                return False
            record = _copy_record(persisted)
            record.request_cancellation(reason)
            if record.status is ExecutionStatus.WAITING:
                record.mark_canceled()
                record.to_json()
                self._records[execution_id] = _copy_record(record)
                self._queued_ids.discard(execution_id)
                return True
            record.to_json()
            self._records[execution_id] = _copy_record(record)
            return True

    async def resume(self, execution_id: str, update: JsonValue | None = None) -> bool:
        async with self._lock:
            persisted = self._records.get(execution_id)
            if persisted is None or persisted.status is not ExecutionStatus.WAITING:
                return False
            record = _copy_record(persisted)
            if record.cancel_requested_at is not None:
                record.mark_canceled()
                self._records[execution_id] = _copy_record(record)
                return False
            if update is not None:
                record.application_state[RESUME_INPUT_KEY] = validate_json(
                    update, limit=record.max_record_bytes, label="resume update"
                )
            record.wait = None
            record.status = ExecutionStatus.QUEUED
            record.append_progress("Execution resumed", kind="resumed")
            record.to_json()
            self._records[execution_id] = _copy_record(record)
            self._enqueue_locked(execution_id)
            return True

    async def retire_incompatible(self, definition_revision: str) -> int:
        """Fail unclaimed work that can no longer run on the published revision."""
        retired = 0
        async with self._lock:
            for execution_id, persisted in list(self._records.items()):
                if (
                    persisted.terminal
                    or persisted.definition_revision == definition_revision
                    or persisted.status is ExecutionStatus.RUNNING
                    or execution_id in self._claims
                ):
                    continue
                record = _copy_record(persisted)
                record.result = None
                record.retry_at = None
                record.wait = None
                if record.cancel_requested_at is not None:
                    record.mark_canceled(force=True)
                    record.append_progress(
                        "Execution canceled during deployment replacement",
                        kind="canceled",
                    )
                else:
                    record.mark_failed(
                        ExecutionError(
                            type="DefinitionRevisionConflictError",
                            message="Execution cannot continue after its deployment definition was replaced",
                            retryable=False,
                            code="definition_revision_conflict",
                        )
                    )
                    record.append_progress(
                        "Execution retired because its deployment definition was replaced",
                        kind="definition_revision_conflict",
                    )
                record.to_json()
                self._records[execution_id] = _copy_record(record)
                self._queued_ids.discard(execution_id)
                retired += 1
            self._evict_locked()
        return retired

    async def operational_counts(self) -> dict[str, int]:
        async with self._lock:
            counts = {status.value: 0 for status in ExecutionStatus}
            for record in self._records.values():
                counts[record.status.value] += 1
            counts["queue_deliveries"] = len(self._queued_ids)
            counts["active_claims"] = len(self._claims)
            counts["delayed"] = sum(
                record.status is ExecutionStatus.QUEUED and record.retry_at is not None and record.retry_at > utc_now()
                for record in self._records.values()
            )
            return counts

    async def close(self) -> None:
        async with self._lock:
            self._closed = True

    def _enqueue_locked(self, execution_id: str) -> None:
        if execution_id not in self._queued_ids:
            self._queued.append(execution_id)
            self._queued_ids.add(execution_id)

    def _evict_locked(self) -> None:
        if self.max_records is None or len(self._records) <= self.max_records:
            return
        for execution_id in list(self._records):
            if len(self._records) <= self.max_records:
                break
            if self._records[execution_id].terminal and execution_id not in self._claims:
                self._records.pop(execution_id, None)

    async def _checkpoint(self, claim: _InMemoryExecutionClaim) -> None:
        async with self._lock:
            self._assert_owner_locked(claim)
            claim.record.touch()
            persisted = self._records[claim.record.execution_id]
            if persisted.cancel_requested_at is not None:
                claim.record.cancel_requested_at = persisted.cancel_requested_at
                claim.record.cancel_reason = persisted.cancel_reason
            claim.record.to_json()
            self._records[claim.record.execution_id] = _copy_record(claim.record)

    async def _complete(self, claim: _InMemoryExecutionClaim) -> None:
        async with self._lock:
            self._assert_owner_locked(claim)
            record = claim.record
            persisted = self._records[record.execution_id]
            if persisted.cancel_requested_at is not None:
                # A cancellation accepted between the worker's final check and
                # this fenced terminal write wins over a completion candidate.
                record.cancel_requested_at = persisted.cancel_requested_at
                record.cancel_reason = persisted.cancel_reason
                record.mark_canceled(force=True)
            if not record.terminal:
                msg = f"Execution '{record.execution_id}' cannot complete while {record.status.value}"
                raise RuntimeError(msg)
            record.to_json()
            self._records[record.execution_id] = _copy_record(record)
            self._claims.pop(record.execution_id, None)
            claim.finished = True
            self._evict_locked()

    async def _suspend(self, claim: _InMemoryExecutionClaim) -> None:
        async with self._lock:
            self._assert_owner_locked(claim)
            persisted = self._records[claim.record.execution_id]
            if persisted.cancel_requested_at is not None:
                claim.record.cancel_requested_at = persisted.cancel_requested_at
                claim.record.cancel_reason = persisted.cancel_reason
                claim.record.mark_canceled(force=True)
            if (
                claim.record.status is not ExecutionStatus.WAITING
                and claim.record.status is not ExecutionStatus.CANCELED
            ):
                msg = "Execution claims may only suspend waiting records"
                raise RuntimeError(msg)
            claim.record.touch()
            claim.record.to_json()
            self._records[claim.record.execution_id] = _copy_record(claim.record)
            self._claims.pop(claim.record.execution_id, None)
            claim.finished = True

    async def _retry(self, claim: _InMemoryExecutionClaim, error: ExecutionError, delay: float) -> None:
        async with self._lock:
            self._assert_owner_locked(claim)
            record = claim.record
            persisted = self._records[record.execution_id]
            if persisted.cancel_requested_at is not None:
                record.cancel_requested_at = persisted.cancel_requested_at
                record.cancel_reason = persisted.cancel_reason
                record.mark_canceled(force=True)
                record.to_json()
                self._records[record.execution_id] = _copy_record(record)
                self._claims.pop(record.execution_id, None)
                claim.finished = True
                return
            record.status = ExecutionStatus.QUEUED
            record.error = error
            record.retry_at = utc_now() + timedelta(seconds=delay)
            record.append_progress("Execution retry scheduled", kind="retry")
            record.to_json()
            self._records[record.execution_id] = _copy_record(record)
            self._claims.pop(record.execution_id, None)
            claim.finished = True
            self._enqueue_locked(record.execution_id)

    async def _exit(self, claim: _InMemoryExecutionClaim) -> None:
        async with self._lock:
            if self._claims.get(claim.record.execution_id) is not claim:
                return
            self._claims.pop(claim.record.execution_id, None)
            persisted = self._records.get(claim.record.execution_id)
            if persisted is not None and not persisted.terminal and persisted.status is not ExecutionStatus.WAITING:
                persisted.status = ExecutionStatus.QUEUED
                persisted.touch()
                self._records[persisted.execution_id] = _copy_record(persisted)
                self._enqueue_locked(persisted.execution_id)

    async def _cancellation_requested(self, execution_id: str) -> bool:
        async with self._lock:
            record = self._records.get(execution_id)
            return bool(record and record.cancel_requested_at is not None)

    def _assert_owner_locked(self, claim: _InMemoryExecutionClaim) -> None:
        if self._claims.get(claim.record.execution_id) is not claim:
            msg = f"Execution claim for '{claim.record.execution_id}' is no longer owned"
            raise ExecutionLeaseLostError(msg)


class _InMemoryExecutionClaim:
    def __init__(self, store: InMemoryExecutionStore, record: ExecutionRecord, token: str) -> None:
        self.store = store
        self._record = record
        self.token = token
        self.finished = False

    @property
    def record(self) -> ExecutionRecord:
        return self._record

    async def checkpoint(self) -> None:
        await self.store._checkpoint(self)

    async def cancellation_requested(self) -> bool:
        return await self.store._cancellation_requested(self.record.execution_id)

    async def complete(self) -> None:
        await self.store._complete(self)

    async def suspend(self) -> None:
        await self.store._suspend(self)

    async def retry(self, error: ExecutionError, *, delay: float) -> None:
        await self.store._retry(self, error, delay)

    async def __aenter__(self) -> _InMemoryExecutionClaim:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.finished:
            await self.store._exit(self)


class InMemoryExecutionStoreProvider:
    """Create volatile namespaced stores; opt in explicitly for development."""

    def __init__(self, **store_kwargs: Any) -> None:
        self.store_kwargs = store_kwargs
        self.stores: dict[str, InMemoryExecutionStore] = {}

    def create_execution_store(self, deployment_name: str) -> InMemoryExecutionStore:
        return self.stores.setdefault(deployment_name, InMemoryExecutionStore(**self.store_kwargs))

    async def close(self) -> None:
        for store in self.stores.values():
            await store.close()


def _copy_record(record: ExecutionRecord | None) -> ExecutionRecord:
    if record is None:
        msg = "execution record is required"
        raise TypeError(msg)
    return copy.deepcopy(record)
