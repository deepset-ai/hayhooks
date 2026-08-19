"""One real-process crash/restart check for Redis-backed durable execution."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import uuid
from pathlib import Path

import pytest
import requests
from redis import Redis

from hayhooks.server.a2a.redis_task_store import RedisTaskStore
from tests.durable_helpers import cleanup_redis, server_error, start_server, stop_server, wait_for_server, wait_until

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not importlib.metadata.version("haystack-ai").startswith("3."), reason="durable execution requires Haystack 3"
    ),
]

_REDIS_URL_ENV = "HAYHOOKS_TEST_REDIS_URL"
_PROCESS_RECOVERY_ENV = "HAYHOOKS_TEST_PROCESS_RECOVERY"
_FIXTURE_DIR = Path(__file__).parent / "test_files/durable_process_recovery"
_A2A_FIXTURE_DIR = Path(__file__).parent / "test_files/durable_a2a_process_recovery"
_CRASH_AFTER_A2A_SUBMIT_ENV = "HAYHOOKS_TEST_CRASH_AFTER_A2A_SUBMIT"


def _wait(predicate, message: str):
    """Poll for up to ten seconds; every wait in this module shares that budget."""
    return wait_until(predicate, message, attempts=200, delay=0.05)


def _wait_for_file(path: Path) -> None:
    _wait(path.exists, "durable test wrapper did not reach its crash window")


def _wait_for_completion(execution_url: str) -> dict:
    def completed() -> dict | None:
        response = requests.get(execution_url, timeout=0.5)
        response.raise_for_status()
        execution = response.json()
        if execution["status"] in {"failed", "canceled"}:
            pytest.fail(f"durable execution ended as {execution['status']}: {execution}")
        return execution if execution["status"] == "completed" else None

    return _wait(completed, "durable execution did not recover to completion")


def create_a2a_recovery_app():
    """Build the process-test A2A app after loading its pipeline fixture."""
    from hayhooks.durable.runtime import DurableDeployment, DurableRuntime
    from hayhooks.server.a2a.app import create_a2a_app
    from hayhooks.server.utils.deploy_utils import deploy_pipelines
    from hayhooks.settings import settings

    durable_runtime = DurableRuntime(app_settings=settings)
    deploy_pipelines(durable_runtime=durable_runtime)
    if os.getenv(_CRASH_AFTER_A2A_SUBMIT_ENV) == "1":
        submit = DurableDeployment.submit

        async def submit_then_crash(self, *args, **kwargs):
            result = await submit(self, *args, **kwargs)
            os.kill(os.getpid(), signal.SIGKILL)
            return result

        DurableDeployment.submit = submit_then_crash
    return create_a2a_app(durable_runtime=durable_runtime)


def _a2a_rpc(base_url: str, method: str, params: dict, request_id: str) -> dict:
    response = requests.post(
        f"{base_url}/durable_agent/",
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        headers={"A2A-Version": "1.0"},
        timeout=2,
    )
    response.raise_for_status()
    payload = response.json()
    assert "error" not in payload, payload
    return payload["result"]


def _wait_for_task_state(base_url: str, task_id: str, state: str) -> dict:
    def ready() -> dict | None:
        task = _a2a_rpc(base_url, "GetTask", {"id": task_id}, f"get-{task_id}")
        return task if task["status"]["state"] == state else None

    return _wait(ready, f"A2A task '{task_id}' did not reach {state}")


def _a2a_task(result: dict) -> dict:
    return result.get("task", result)


def test_redis_durable_execution_survives_a_process_kill_and_restart(tmp_path: Path, unused_tcp_port: int) -> None:
    if os.getenv(_PROCESS_RECOVERY_ENV) != "1":
        pytest.skip(f"set {_PROCESS_RECOVERY_ENV}=1 to run the process-recovery smoke test")
    redis_url = os.getenv(_REDIS_URL_ENV)
    if not redis_url:
        pytest.skip(f"set {_REDIS_URL_ENV} to run the process-recovery smoke test")

    pipelines_dir = tmp_path / "pipelines"
    shutil.copytree(_FIXTURE_DIR, pipelines_dir / "recovery_job")
    database_path = tmp_path / "effects.sqlite3"
    ready_file = tmp_path / "ready"
    prefix = f"hayhooks:test:process-recovery:{uuid.uuid4().hex}"
    environment = os.environ | {
        "HAYHOOKS_PIPELINES_DIR": str(pipelines_dir),
        "HAYHOOKS_DURABLE_STORE": "redis",
        "HAYHOOKS_DURABLE_REDIS_URL": redis_url,
        "HAYHOOKS_DURABLE_REDIS_KEY_PREFIX": prefix,
        "HAYHOOKS_DURABLE_LEASE_DURATION_MS": "250",
        "HAYHOOKS_DURABLE_LEASE_COMMIT_SAFETY_MS": "25",
        "HAYHOOKS_DURABLE_MAX_ATTEMPTS": "2",
    }
    base_url = f"http://127.0.0.1:{unused_tcp_port}"
    request_body = {"database_path": str(database_path), "ready_file": str(ready_file)}
    headers = {"Idempotency-Key": "process-recovery"}
    servers: list[subprocess.Popen[str]] = []

    try:
        server = start_server(unused_tcp_port, environment)
        servers.append(server)
        wait_for_server(server, base_url)

        submitted = requests.post(f"{base_url}/recovery_job/run-durable", json=request_body, headers=headers, timeout=1)
        assert submitted.status_code == 202, submitted.text
        execution = submitted.json()
        execution_url = f"{base_url}{execution['links']['self']}"
        _wait_for_file(ready_file)

        server.kill()
        server.wait(timeout=3)

        server = start_server(unused_tcp_port, environment)
        servers.append(server)
        wait_for_server(server, base_url)
        completed = _wait_for_completion(execution_url)

        assert completed["attempt"] == 2
        assert completed["result"] == {"attempt": 2, "effect_applied": False}
        with sqlite3.connect(database_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM checkpoint_runs").fetchone() == (1,)
            assert connection.execute("SELECT COUNT(*) FROM effects").fetchone() == (1,)

        replay = requests.post(f"{base_url}/recovery_job/run-durable", json=request_body, headers=headers, timeout=1)
        assert replay.status_code == 200, replay.text
        assert replay.headers["Idempotent-Replay"] == "true"
        assert replay.json()["execution_id"] == execution["execution_id"]
    finally:
        for server in servers:
            stop_server(server)
        cleanup_redis(redis_url, prefix)


def test_redis_durable_a2a_tasks_read_through_after_restart(tmp_path: Path, unused_tcp_port: int) -> None:
    if os.getenv(_PROCESS_RECOVERY_ENV) != "1":
        pytest.skip(f"set {_PROCESS_RECOVERY_ENV}=1 to run the process-recovery smoke test")
    redis_url = os.getenv(_REDIS_URL_ENV)
    if not redis_url:
        pytest.skip(f"set {_REDIS_URL_ENV} to run the process-recovery smoke test")

    pipelines_dir = tmp_path / "pipelines"
    shutil.copytree(_A2A_FIXTURE_DIR, pipelines_dir / "durable_agent")
    prefix = f"hayhooks:test:a2a-process-recovery:{uuid.uuid4().hex}"
    environment = os.environ | {
        "HAYHOOKS_PIPELINES_DIR": str(pipelines_dir),
        "HAYHOOKS_A2A_EXTERNAL_URL": f"http://127.0.0.1:{unused_tcp_port}",
        "HAYHOOKS_A2A_TASK_STORE": "redis",
        "HAYHOOKS_A2A_REDIS_URL": redis_url,
        "HAYHOOKS_A2A_REDIS_KEY_PREFIX": f"{prefix}:a2a",
        "HAYHOOKS_DURABLE_STORE": "redis",
        "HAYHOOKS_DURABLE_REDIS_URL": redis_url,
        "HAYHOOKS_DURABLE_REDIS_KEY_PREFIX": f"{prefix}:durable",
        "HAYHOOKS_DURABLE_POLL_INTERVAL": "0.05",
        "HAYHOOKS_DURABLE_LEASE_DURATION_MS": "250",
        "HAYHOOKS_DURABLE_LEASE_COMMIT_SAFETY_MS": "25",
    }
    base_url = f"http://127.0.0.1:{unused_tcp_port}"
    factory = "tests.test_durable_process_recovery:create_a2a_recovery_app"
    servers: list[subprocess.Popen[str]] = []

    def submit(text: str) -> str:
        message = {"messageId": f"message-{text}", "role": "ROLE_USER", "parts": [{"text": text}]}
        result = _a2a_rpc(
            base_url,
            "SendMessage",
            {
                "message": message,
                "configuration": {"returnImmediately": True},
            },
            f"send-{text}",
        )
        return _a2a_task(result)["id"]

    try:
        server = start_server(
            unused_tcp_port,
            environment | {_CRASH_AFTER_A2A_SUBMIT_ENV: "1"},
            factory,
        )
        servers.append(server)
        wait_for_server(server, base_url)
        with pytest.raises(requests.RequestException):
            submit("resume")
        server.wait(timeout=3)
        redis = Redis.from_url(redis_url)
        try:
            task_store = RedisTaskStore(redis, "durable_agent", key_prefix=f"{prefix}:a2a")
            task_ids = redis.zrange(task_store._key("active"), 0, -1)
        finally:
            redis.close()
        assert len(task_ids) == 1
        resumable_id = task_ids[0].decode() if isinstance(task_ids[0], bytes) else task_ids[0]

        server = start_server(unused_tcp_port, environment, factory)
        servers.append(server)
        wait_for_server(server, base_url)

        assert _wait_for_task_state(base_url, resumable_id, "TASK_STATE_INPUT_REQUIRED")
        cancelable_id = submit("cancel")
        _wait_for_task_state(base_url, cancelable_id, "TASK_STATE_INPUT_REQUIRED")
        listed = _a2a_rpc(base_url, "ListTasks", {}, "list")
        assert {task["id"] for task in listed["tasks"]} == {resumable_id, cancelable_id}

        resumed = _a2a_rpc(
            base_url,
            "SendMessage",
            {
                "message": {
                    "messageId": "message-approved",
                    "taskId": resumable_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "approved"}],
                }
            },
            "resume",
        )
        if _a2a_task(resumed)["status"]["state"] != "TASK_STATE_COMPLETED":
            stop_server(server)
            pytest.fail(f"{json.dumps(resumed)}\n{server_error(server)}")

        canceled = _a2a_rpc(base_url, "CancelTask", {"id": cancelable_id}, "cancel")
        if _a2a_task(canceled)["status"]["state"] != "TASK_STATE_CANCELED":
            _wait_for_task_state(base_url, cancelable_id, "TASK_STATE_CANCELED")
    finally:
        for server in servers:
            stop_server(server)
        cleanup_redis(redis_url, prefix)
