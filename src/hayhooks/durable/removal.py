"""Immutable removal evidence and bounded lease ownership scopes."""
# ruff: noqa: EM101

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

MAX_REMOVAL_LEASE_CONTROLS = 200
_FAMILY_WORKER = re.compile(r"hh1:([0-9a-f]{64}):[0-9a-f]{16}-[0-9]+\Z")
_INSTANCE_WORKER = re.compile(r"([0-9a-f]{16})-[0-9]+\Z")


@dataclass(frozen=True, slots=True)
class RemovalScope:
    """A current deployment instance or a trusted stable owner family within one deployment."""

    kind: Literal["deployment_instance", "owner_family"]
    identity: str

    def __post_init__(self) -> None:
        size = 64 if self.kind == "owner_family" else 16
        if self.kind not in ("deployment_instance", "owner_family") or not re.fullmatch(
            rf"[0-9a-f]{{{size}}}", self.identity
        ):
            raise ValueError("invalid removal ownership scope")

    def owns(self, worker_id: str) -> bool:
        return worker_scope(worker_id) == self


def worker_scope(worker_id: str) -> RemovalScope | None:
    if match := _FAMILY_WORKER.fullmatch(worker_id):
        return RemovalScope("owner_family", match[1])
    if match := _INSTANCE_WORKER.fullmatch(worker_id):
        return RemovalScope("deployment_instance", match[1])
    return None


@dataclass(frozen=True, slots=True)
class StoreRemovalSnapshot:
    """Atomic namespace counts and optional ownership evidence, never cached metrics."""

    observed_at_ms: int
    nonterminal: int
    active_leases: int
    owned_leases: int | None
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RemovalProof:
    """This instance's local drain plus durable scope evidence, never a receipt for other processes."""

    deployment: str
    revision: str
    scope: RemovalScope
    worker_identity: str
    complete: bool
    pending_writes: int
    worker_tasks: int
    local_active: int
    owned_leases: int | None
    store_snapshot: StoreRemovalSnapshot | None
    blockers: tuple[str, ...]


@runtime_checkable
class RemovalStore(Protocol):
    """Optional atomic claim-fencing and observation contract for removal-capable stores."""

    async def owner_scope_fenced(self, scope: RemovalScope) -> bool:
        """Read whether new claims for this scope are permanently prohibited."""
        ...

    async def fence_owner_scope(self, scope: RemovalScope) -> None:
        """Durably reject new claims for this scope, atomically with each claim commit."""
        ...

    async def removal_snapshot(self, *, scope: RemovalScope | None = None) -> StoreRemovalSnapshot:
        """Read atomic counts. Unprovable scope ownership returns None and blockers."""
        ...
