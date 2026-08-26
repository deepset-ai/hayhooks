"""Process-kill recovery smoke test for Redis-backed Haystack execution."""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path

import pytest
from pydantic import BaseModel
from redis.asyncio import Redis

from hayhooks.durable.engine import ExecutionStatus
from hayhooks.durable.haystack import HaystackDurableAdapter
from hayhooks.durable.redis import RedisExecutionStore
from hayhooks.durable.runtime import DurableDeployment, RuntimeConfig
from hayhooks.durable.store import StoreConfig

try:
    _HAYSTACK_VERSION = tuple(int(part) for part in version("haystack-ai").split(".", maxsplit=2)[:2])
except ValueError:
    _HAYSTACK_VERSION = (0, 0)
_HAYSTACK_V3 = (3, 1) <= _HAYSTACK_VERSION < (4, 0)

if _HAYSTACK_V3:
    from haystack import Pipeline, component

    @component
    class RecordingIncrement:
        def __init__(self, marker: str, label: str, delay: float = 0) -> None:
            self.marker = marker
            self.label = label
            self.delay = delay

        @component.output_types(value=int)
        def run(self, value: int) -> dict[str, int]:
            with Path(self.marker).open("a", encoding="utf-8") as stream:
                stream.write(f"{self.label}\n")
            time.sleep(self.delay)
            return {"value": value + 1}


class RecoveryRequest(BaseModel):
    value: int


async def _run_worker(mode: str, redis_url: str, prefix: str, marker: str, run_id: str = "") -> None:
    redis = Redis.from_url(redis_url, decode_responses=False)
    store = RedisExecutionStore(
        redis,
        "jobs",
        config=StoreConfig(lease_commit_safety_ms=10, terminal_ttl_seconds=60),
        key_prefix=prefix,
    )
    pipeline = Pipeline()
    pipeline.add_component("first", RecordingIncrement(marker, "first"))
    pipeline.add_component("second", RecordingIncrement(marker, "second", 30 if mode == "crash" else 0))
    pipeline.connect("first.value", "second.value")
    adapter = HaystackDurableAdapter(pipeline)

    def run(context, request: RecoveryRequest):
        return adapter.run_pipeline(context, {"first": {"value": request.value}}, checkpoint_at="second")

    deployment = DurableDeployment(
        "jobs",
        "v1",
        store,
        RecoveryRequest,
        run,
        config=RuntimeConfig(
            poll_interval_seconds=0.02,
            shutdown_grace_seconds=0.1,
            lease_duration_ms=300,
            operational_backoff_min_seconds=0.01,
            operational_backoff_max_seconds=0.1,
        ),
    )
    await deployment.start()
    try:
        if mode == "crash":
            submitted = await deployment.submit({"value": 1})
            sys.stdout.write(f"{submitted.control.run_id}\n")
            sys.stdout.flush()
            await asyncio.Event().wait()
        else:
            for _ in range(500):
                stored = await store.read(run_id)
                if stored is not None and stored.control.terminal:
                    sys.stdout.write(f"{stored.control.status.value}\n")
                    sys.stdout.flush()
                    return
                await asyncio.sleep(0.02)
            message = "recovered execution did not finish"
            raise TimeoutError(message)
    finally:
        await deployment.close()
        await redis.aclose()


@pytest.mark.integration
@pytest.mark.skipif(not _HAYSTACK_V3, reason="process recovery requires Haystack 3.1+")
async def test_killed_pipeline_resumes_without_repeating_completed_component(tmp_path: Path) -> None:
    redis_url = os.getenv("HAYHOOKS_TEST_REDIS_URL")
    if not redis_url:
        pytest.skip("set HAYHOOKS_TEST_REDIS_URL to run the real-Redis suite")
    prefix = f"hayhooks:test:{uuid.uuid4().hex}"
    marker = tmp_path / "calls"

    async def spawn(mode: str, run_id: str = ""):
        return await asyncio.create_subprocess_exec(
            sys.executable,
            __file__,
            mode,
            redis_url,
            prefix,
            str(marker),
            run_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    crashed = await spawn("crash")
    recovered = None
    try:
        assert crashed.stdout is not None
        run_id = (await asyncio.wait_for(crashed.stdout.readline(), timeout=10)).decode().strip()
        for _ in range(500):
            if marker.exists() and marker.read_text().splitlines().count("second") == 1:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("pipeline never reached the post-checkpoint component")
        crashed.kill()
        await crashed.wait()

        recovered = await spawn("recover", run_id)
        stdout, stderr = await asyncio.wait_for(recovered.communicate(), timeout=15)
        assert recovered.returncode == 0, stderr.decode()
        assert stdout.decode().strip() == ExecutionStatus.COMPLETED.value
        assert marker.read_text().splitlines() == ["first", "second", "second"]
    finally:
        if crashed.returncode is None:
            crashed.kill()
            await crashed.wait()
        if recovered is not None and recovered.returncode is None:
            recovered.kill()
            await recovered.wait()
        redis = Redis.from_url(redis_url, decode_responses=False)
        keys = [key async for key in redis.scan_iter(match=f"{prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(_run_worker(*sys.argv[1:]))
