"""Configuration owned by the portable durable runtime."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class DurableSettings(BaseModel):
    """Durable storage, retention, retry, lease, and worker settings."""

    durable_store: Literal["memory", "redis"] = "redis"
    durable_redis_url: str = "redis://localhost:6379/0"
    durable_redis_key_prefix: str = "hayhooks:durable"
    durable_redis_socket_timeout: float = Field(default=5.0, gt=0.0, le=300.0)
    durable_redis_socket_connect_timeout: float = Field(default=5.0, gt=0.0, le=300.0)
    durable_redis_health_check_interval: int = Field(default=30, ge=0, le=3_600)
    durable_terminal_ttl_seconds: int = Field(default=604_800, ge=1)
    durable_max_progress_events: int = Field(default=100, ge=1, le=10_000)
    durable_max_record_bytes: int = Field(default=1_000_000, ge=1_024)
    durable_max_nonterminal_executions: int = Field(default=0, ge=0)
    durable_shutdown_grace_period: float = Field(default=5.0, ge=0.0)
    durable_max_attempts: int = Field(default=3, ge=1, le=1_000)
    durable_retry_base_delay: float = Field(default=1.0, ge=0.0, le=86_400.0)
    durable_retry_max_delay: float = Field(default=60.0, ge=0.0, le=604_800.0)
    durable_poll_interval: float = Field(default=1.0, ge=0.05, le=60.0)
    durable_lease_duration_ms: int = Field(default=30_000, ge=1, le=86_400_000)
    durable_lease_commit_safety_ms: int = Field(default=1_500, ge=0, le=86_400_000)
    durable_execution_concurrency: int = Field(default=1, ge=1, le=128)

    @model_validator(mode="after")
    def _validate_lease_margin(self) -> Self:
        if self.durable_lease_commit_safety_ms >= self.durable_lease_duration_ms:
            msg = "durable_lease_commit_safety_ms must be smaller than durable_lease_duration_ms"
            raise ValueError(msg)
        if self.durable_lease_duration_ms - self.durable_lease_commit_safety_ms <= max(
            10, self.durable_lease_duration_ms / 3
        ):
            msg = "durable lease duration minus commit safety must exceed the heartbeat interval"
            raise ValueError(msg)
        return self

    @classmethod
    def from_app_settings(cls, app_settings: Any) -> DurableSettings:
        """Copy durable fields from Hayhooks settings without retaining that dependency."""
        return cls(**{name: getattr(app_settings, name) for name in cls.model_fields})


__all__ = ["DurableSettings"]
