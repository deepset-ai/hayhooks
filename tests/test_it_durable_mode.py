"""Durable mode against real Redis: managed shutdown ordering and recovery in a fresh interpreter."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hayhooks.durable.engine import ExecutionStatus
from hayhooks.server.app import create_app
from hayhooks.server.pipelines.loader import registry_module_name
from hayhooks.settings import settings
from tests.pipeline_sources import HAYSTACK_V3, write_tree

REDIS_URL = os.getenv("HAYHOOKS_TEST_REDIS_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REDIS_URL, reason="set HAYHOOKS_TEST_REDIS_URL to run the real-Redis suite"),
    pytest.mark.skipif(not HAYSTACK_V3, reason="durable adapters require Haystack 3.1+"),
]

# A synchronous runner that holds its claim until the test releases it, then checkpoints through Redis.
RETAINED_WRAPPER = """
import threading

from haystack import Pipeline
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext

started = threading.Event()
release = threading.Event()


class Request(BaseModel):
    value: int


class Result(BaseModel):
    value: int


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "v1"

    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_durable(self, context: DurableContext, request: Request) -> Result:
        started.set()
        release.wait(30)
        context.checkpoint_sync()
        return Result(value=request.value)
"""

# Checkpoints a Haystack State holding a wrapper-local type and handler, then hangs so the process can be killed.
NOTES_WRAPPER = """
import asyncio
import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.agents.state import State
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext


class Note(BaseModel):
    text: str


def keep_latest(_current, new):
    return new


class Request(BaseModel):
    text: str


class Result(BaseModel):
    text: str
    note_type: str
    handler_module: str
    attempt: int


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "notes-v1"

    def setup(self) -> None:
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
        if "notes" not in context.state:
            schema = {"note": {"type": Note, "handler": keep_latest}}
            state = State(schema=schema, data={"note": Note(text=request.text)})
            context.state["notes"] = state.to_dict()
            await context.checkpoint()
            with Path(os.environ["NOTES_MARKER"]).open("a") as marker:
                marker.write("checkpointed\\n")
            await asyncio.sleep(60)
        state = State.from_dict(context.state["notes"])
        note = state.get("note")
        return Result(
            text=note.text,
            note_type=f"{type(note).__module__}.{type(note).__qualname__}",
            handler_module=state.schema["note"]["handler"].__module__,
            attempt=context.attempt,
        )
"""

_events: list[str] = []


def _probe_redis_class():
    from redis.asyncio import Redis

    class ProbeRedis(Redis):
        """Records aclose() and rejects later commands, so a reconnecting client cannot hide an early close."""

        closed = False

        async def aclose(self, *args, **kwargs) -> None:
            self.closed = True
            _events.append("aclose")
            await super().aclose(*args, **kwargs)

        async def execute_command(self, *args, **options):
            self._require_open()
            return await super().execute_command(*args, **options)

        def pipeline(self, *args, **kwargs):
            self._require_open()
            return super().pipeline(*args, **kwargs)

        def _require_open(self) -> None:
            if self.closed:
                _events.append("used after close")
                msg = "Redis client used after aclose()"
                raise RuntimeError(msg)

    return ProbeRedis


@pytest.fixture
def redis_prefix() -> Iterator[str]:
    from redis.asyncio import Redis

    prefix = f"hayhooks:test:{uuid.uuid4().hex}"
    yield prefix

    async def cleanup() -> None:
        redis = Redis.from_url(REDIS_URL)
        keys = [key async for key in redis.scan_iter(match=f"{prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()

    asyncio.run(cleanup())


@pytest.fixture
def redis_durable_mode(durable_pipelines_dir: Path, redis_prefix: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    import redis.asyncio

    for name, value in {
        "durable_redis_url": REDIS_URL,
        "durable_redis_key_prefix": redis_prefix,
        "durable_shutdown_grace_seconds": 0.1,
        "durable_poll_interval_seconds": 0.02,
        "durable_maintenance_interval_seconds": 0.02,
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(redis.asyncio, "Redis", _probe_redis_class())
    _events.clear()
    return durable_pipelines_dir


class _Background(threading.Thread):
    def __init__(self, target) -> None:
        super().__init__(daemon=True)
        self._target_call = target
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self._target_call()
        except BaseException as error:
            self.error = error


async def _read_execution(prefix: str, run_id: str):
    from redis.asyncio import Redis

    from hayhooks.durable.redis import RedisExecutionStore

    redis = Redis.from_url(REDIS_URL)
    try:
        return await RedisExecutionStore(redis, "jobs", key_prefix=prefix).read(run_id)
    finally:
        await redis.aclose()


def test_retained_work_keeps_redis_until_drained_and_streams_end_first(redis_durable_mode: Path, redis_prefix: str):
    write_tree(redis_durable_mode, {"jobs/pipeline_wrapper.py": RETAINED_WRAPPER})
    app = create_app()
    wrapper_module = sys.modules[registry_module_name("jobs")]
    client = TestClient(app)
    client.__enter__()
    run_id = client.post("/jobs/run-durable", json={"value": 7}).json()["execution_id"]
    assert wrapper_module.started.wait(5)

    stream_path = f"/jobs/executions/{run_id}/stream"
    stream = _Background(lambda: _events.append(f"stream ended: {client.get(stream_path).text}"))
    stream.start()
    time.sleep(0.3)  # the stream is now blocked on the viewer client
    shutdown = _Background(lambda: client.__exit__(None, None, None))
    shutdown.start()

    stream.join(5)
    assert not stream.is_alive()
    # Past the shutdown grace, the retained thread still owns its claim, so Redis stays open.
    shutdown.join(1)
    assert shutdown.is_alive()
    assert len(_events) == 1
    assert _events[0].startswith("stream ended")
    assert "event: completed" not in _events[0]

    wrapper_module.release.set()
    shutdown.join(10)

    assert not shutdown.is_alive()
    assert shutdown.error is None
    assert _events[1:] == ["aclose", "aclose"]
    stored = asyncio.run(_read_execution(redis_prefix, run_id))
    assert stored is not None
    assert stored.control.status is ExecutionStatus.COMPLETED


@pytest.mark.parametrize("failure", ["start", "close", "drain", "drain-cancelled"])
def test_lifespan_cleanup_releases_redis_only_after_drainage(
    redis_durable_mode: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from tests.pipeline_sources import DURABLE_WRAPPER

    write_tree(redis_durable_mode, {f"{name}/pipeline_wrapper.py": DURABLE_WRAPPER for name in ("first", "second")})
    app = create_app()
    runtime = app.state.durable_runtime
    first, second = runtime._deployments.values()
    wait_drained = runtime.wait_drained

    async def record_drain() -> None:
        _events.append("drain before close" if first.accepting else "drain")
        if failure == "drain":
            raise RuntimeError(failure)
        if failure == "drain-cancelled":
            raise asyncio.CancelledError
        await wait_drained()

    async def fail(*_args) -> None:
        raise RuntimeError(failure)

    close_second = second.close

    async def failing_close() -> None:
        await close_second()
        raise RuntimeError(failure)

    monkeypatch.setattr(runtime, "wait_drained", record_drain)
    if failure == "start":
        monkeypatch.setattr(second.store, "initialize", fail)
    elif failure == "close":
        monkeypatch.setattr(second, "close", failing_close)

    with pytest.raises(BaseException) as error, TestClient(app):
        pass

    # The originating failure propagates; a cancelled drain surfaces as the portal's cancellation.
    if failure != "drain-cancelled":
        assert str(error.value) == failure
    # Every deployment closes, including the one that started before the failure.
    assert not first.accepting
    assert not second.accepting
    # An unresolved drain leaves the clients to the failed app until the process exits.
    assert _events == (["drain", "aclose", "aclose"] if failure in ("start", "close") else ["drain"])


def test_same_revision_recovers_typed_state_in_a_fresh_interpreter_after_relocation(
    tmp_path: Path, redis_prefix: str
) -> None:
    source = write_tree(tmp_path / "v1" / "pipelines", {"notes/pipeline_wrapper.py": NOTES_WRAPPER})
    marker = tmp_path / "marker"
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[1]),
        "HAYHOOKS_DURABLE_MODE": "true",
        "HAYHOOKS_DURABLE_REDIS_URL": REDIS_URL or "",
        "HAYHOOKS_DURABLE_REDIS_KEY_PREFIX": redis_prefix,
        "HAYHOOKS_DURABLE_LEASE_DURATION_MS": "600",
        "HAYHOOKS_DURABLE_LEASE_COMMIT_SAFETY_MS": "10",
        "HAYHOOKS_DURABLE_POLL_INTERVAL_SECONDS": "0.05",
        "HAYHOOKS_DURABLE_MAINTENANCE_INTERVAL_SECONDS": "0.05",
        # Allowlist entries name the stable module path, not the source location.
        "HAYSTACK_DESERIALIZATION_ALLOWLIST": registry_module_name("notes"),
        "NOTES_MARKER": str(marker),
    }
    submit = """
        import time

        from fastapi.testclient import TestClient

        from hayhooks.server.app import create_app

        with TestClient(create_app()) as client:
            print(client.post("/notes/run-durable", json={"text": "hello"}).json()["execution_id"], flush=True)
            time.sleep(120)
    """
    crashed = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(submit)],
        env={**env, "HAYHOOKS_PIPELINES_DIR": str(source)},
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert crashed.stdout is not None
        run_id = crashed.stdout.readline().strip()
        for _ in range(500):
            if marker.exists():
                break
            time.sleep(0.02)
        else:
            pytest.fail("the execution never checkpointed")
    finally:
        crashed.kill()
        crashed.wait()

    relocated = tmp_path / "v2" / "pipelines"
    shutil.move(source, relocated)
    recover = f"""
        import json
        import time

        from fastapi.testclient import TestClient

        from hayhooks.server.app import create_app

        with TestClient(create_app()) as client:
            for _ in range(500):
                execution = client.get("/notes/executions/{run_id}").json()
                if execution["status"] in ("completed", "failed"):
                    break
                time.sleep(0.02)
        print(json.dumps(execution))
    """
    recovered = subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(recover)],
        env={**env, "HAYHOOKS_PIPELINES_DIR": str(relocated)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert recovered.returncode == 0, recovered.stderr
    execution = json.loads(recovered.stdout.strip().splitlines()[-1])
    assert execution["status"] == "completed", execution
    assert execution["result"] == {
        "text": "hello",
        "note_type": f"{registry_module_name('notes')}.Note",
        "handler_module": registry_module_name("notes"),
        "attempt": 2,
    }
    assert marker.read_text() == "checkpointed\n"
