import asyncio
import importlib.metadata
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from hayhooks.a2a import A2APipelineWrapper, TaskStoreProvider
from hayhooks.durable.models import ExecutionAdmissionError, ExecutionStatus, ExecutionStoreError
from hayhooks.durable.runtime import execution_id_for
from hayhooks.server.a2a.app import create_a2a_app
from hayhooks.server.a2a.durable_executor import DurableAgentExecutor, DurableTaskStore
from hayhooks.server.a2a.imports import TaskStore, new_task_from_user_message, new_text_part
from hayhooks.server.a2a.runtime import A2ARuntime
from hayhooks.server.pipelines.registry import registry
from hayhooks.settings import settings

pytestmark = pytest.mark.skipif(
    not importlib.metadata.version("haystack-ai").startswith("3."), reason="durable execution requires Haystack 3"
)


class _Deployment:
    def __init__(self, status=ExecutionStatus.COMPLETED) -> None:
        self.record = SimpleNamespace(
            status=status,
            progress=[],
            result={"last_message": {"content": "recovered"}},
            error=None,
            sequence=0,
            validated_input={},
        )
        self.execution_id = None
        self.submitted_payload = None
        self.resume_update = None
        self.cancel_requested = False

    async def start(self):
        return None

    async def submit(self, payload, *, execution_id=None, owner_id=None):
        self.submitted_payload = payload
        self.record.validated_input = payload
        self.execution_id = execution_id_for(owner_id, execution_id) if owner_id else execution_id
        self.record.execution_id = self.execution_id
        return True, self.record

    async def get(self, execution_id, **_kwargs):
        if self.execution_id is not None and execution_id != self.execution_id:
            raise KeyError(execution_id)
        return self.record

    async def resume(self, execution_id, update, **_kwargs):
        self.execution_id = execution_id
        self.resume_update = update
        self.record.status = ExecutionStatus.COMPLETED
        self.record.result = {"last_message": {"content": "resumed"}}
        self.record.sequence += 1
        return True

    async def request_cancel(self, _execution_id, **_kwargs):
        self.cancel_requested = True
        self.record.status = ExecutionStatus.CANCELED
        self.record.sequence += 1
        return True


class _BlockingDeployment(_Deployment):
    def __init__(self) -> None:
        super().__init__()
        self.submit_started = threading.Event()
        self.allow_submit = threading.Event()

    async def submit(self, *args, **kwargs):
        self.submit_started.set()
        await asyncio.to_thread(self.allow_submit.wait)
        return await super().submit(*args, **kwargs)


class _DurableHTTPWrapper(A2APipelineWrapper):
    durable_revision = "durable-http-wrapper"

    def setup(self):
        self.pipeline = object()


class _HTTPStore(TaskStore):
    def __init__(self) -> None:
        self.tasks = {}

    async def save(self, task, _context):
        self.tasks[task.id] = task

    async def get(self, task_id, _context):
        return self.tasks.get(task_id)

    async def list(self, _params, _context):
        from a2a.types import ListTasksResponse

        return ListTasksResponse(tasks=list(self.tasks.values()), page_size=len(self.tasks), total_size=len(self.tasks))

    async def delete(self, task_id, _context):
        self.tasks.pop(task_id, None)


class _HTTPStoreProvider(TaskStoreProvider):
    def __init__(self, store) -> None:
        self.store = store

    def create_task_store(self, _agent_name):
        return self.store


def _send_payload(text, *, task_id=None, return_immediately=False):
    message = {"messageId": f"message-{text}", "role": "ROLE_USER", "parts": [{"text": text}]}
    if task_id is not None:
        message["taskId"] = task_id
    params = {"message": message}
    if return_immediately:
        params["configuration"] = {"returnImmediately": True}
    return {"jsonrpc": "2.0", "id": "send", "method": "SendMessage", "params": params}


def _get_payload(task_id):
    return {"jsonrpc": "2.0", "id": "get", "method": "GetTask", "params": {"id": task_id}}


def _cancel_payload(task_id):
    return {"jsonrpc": "2.0", "id": "cancel", "method": "CancelTask", "params": {"id": task_id}}


def _response_task(response):
    result = response.json()["result"]
    return result.get("task", result)


def _recoverable_task():
    from a2a.types import Message, Role

    return new_task_from_user_message(
        Message(
            message_id="message",
            task_id="task",
            context_id="context",
            role=Role.ROLE_USER,
            parts=[new_text_part("recover me")],
        )
    )


def _recovery_store(task, *, saved=True):
    return SimpleNamespace(
        recoverable_task_batch=AsyncMock(return_value=([(task, "owner", 1)], None)),
        save_projection=AsyncMock(return_value=saved),
    )


def _http_app(store, deployment, monkeypatch):
    wrapper = _DurableHTTPWrapper()
    wrapper.setup()
    registry.add("durable-agent", wrapper, metadata={"description": "durable agent"})
    monkeypatch.setattr("hayhooks.durable.runtime.durable_runtime.deployment", lambda *_args: deployment)
    return create_a2a_app(
        base_url="http://a2a-test:1418",
        runtime=A2ARuntime(task_store_provider=_HTTPStoreProvider(store)),
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def http_store() -> _HTTPStore:
    return _HTTPStore()


def test_a2a_http_reads_completion_from_durable_execution(monkeypatch, http_store) -> None:
    app = _http_app(http_store, _Deployment(), monkeypatch)

    with TestClient(app, headers={"A2A-Version": "1.0"}) as client:
        completed = _response_task(client.post("/durable-agent/", json=_send_payload("initial")))

    assert completed["status"]["state"] == "TASK_STATE_COMPLETED"
    assert completed["artifacts"][-1]["name"] == "durable-result"


def test_expired_task_projection_uses_the_retained_execution(monkeypatch, http_store) -> None:
    deployment = _Deployment()
    app = _http_app(http_store, deployment, monkeypatch)

    with TestClient(app, headers={"A2A-Version": "1.0"}) as client:
        completed = _response_task(client.post("/durable-agent/", json=_send_payload("initial")))
        http_store.tasks.clear()
        deployment.submit = AsyncMock(side_effect=AssertionError("retained execution must not be resubmitted"))
        replayed = _response_task(client.post("/durable-agent/", json=_get_payload(completed["id"])))

    assert replayed["status"]["state"] == "TASK_STATE_COMPLETED"
    assert replayed["contextId"] == completed["contextId"]
    assert replayed["artifacts"][-1]["name"] == "durable-result"
    deployment.submit.assert_not_awaited()
    assert not http_store.tasks


def test_a2a_http_waiting_task_resumes_with_only_the_follow_up(monkeypatch, http_store) -> None:
    deployment = _Deployment(status=ExecutionStatus.WAITING)
    app = _http_app(http_store, deployment, monkeypatch)

    with TestClient(app, headers={"A2A-Version": "1.0"}) as client:
        waiting = _response_task(client.post("/durable-agent/", json=_send_payload("initial")))
        assert waiting["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        completed = _response_task(
            client.post("/durable-agent/", json=_send_payload("follow up", task_id=waiting["id"]))
        )

    assert completed["status"]["state"] == "TASK_STATE_COMPLETED"
    assert deployment.resume_update == {
        "messages": [{"role": "user", "meta": {}, "name": None, "content": [{"text": "follow up"}]}]
    }


def test_a2a_http_cancel_reaches_durable_execution(monkeypatch, http_store) -> None:
    deployment = _Deployment(status=ExecutionStatus.RUNNING)
    app = _http_app(http_store, deployment, monkeypatch)

    with TestClient(app, headers={"A2A-Version": "1.0"}) as client:
        active = _response_task(client.post("/durable-agent/", json=_send_payload("initial", return_immediately=True)))
        canceled = _response_task(client.post("/durable-agent/", json=_cancel_payload(active["id"])))

    assert deployment.cancel_requested
    assert canceled["status"]["state"] == "TASK_STATE_CANCELED"


def test_return_immediately_waits_for_durable_submission(monkeypatch, http_store) -> None:
    deployment = _BlockingDeployment()
    app = _http_app(http_store, deployment, monkeypatch)

    with TestClient(app, headers={"A2A-Version": "1.0"}) as client, ThreadPoolExecutor() as pool:
        response = pool.submit(client.post, "/durable-agent/", json=_send_payload("initial", return_immediately=True))
        try:
            assert deployment.submit_started.wait(timeout=1)
            assert http_store.tasks
            with pytest.raises(FutureTimeoutError):
                response.result(timeout=0.1)
        finally:
            deployment.allow_submit.set()
        assert response.result(timeout=2).status_code == 200


def test_returned_task_is_eventually_persisted_as_terminal(monkeypatch, http_store) -> None:
    from a2a.types import TaskState

    deployment = _Deployment(status=ExecutionStatus.COMPLETED)
    app = _http_app(http_store, deployment, monkeypatch)

    with TestClient(app, headers={"A2A-Version": "1.0"}) as client:
        active = _response_task(client.post("/durable-agent/", json=_send_payload("initial", return_immediately=True)))
        deadline = time.monotonic() + 1
        while (
            http_store.tasks[active["id"]].status.state != TaskState.TASK_STATE_COMPLETED
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        completed = _response_task(client.post("/durable-agent/", json=_get_payload(active["id"])))

    assert completed["status"]["state"] == "TASK_STATE_COMPLETED"
    assert http_store.tasks[active["id"]].status.state == TaskState.TASK_STATE_COMPLETED


async def test_expired_execution_preserves_retained_terminal_task(http_store) -> None:
    from a2a.types import Task, TaskState

    task = Task(id="retained-terminal", context_id="context")
    task.status.state = TaskState.TASK_STATE_COMPLETED
    deployment = _Deployment()
    deployment.get = AsyncMock(side_effect=KeyError("expired"))

    projected = await DurableTaskStore(http_store, deployment)._project(task, "owner", SimpleNamespace())

    assert projected.status.state == TaskState.TASK_STATE_COMPLETED


async def test_missing_execution_preserves_a_task_awaiting_submission(http_store) -> None:
    from a2a.types import TaskState

    task = _recoverable_task()
    deployment = _Deployment()
    deployment.get = AsyncMock(side_effect=KeyError("submission has not committed yet"))

    projected = await DurableTaskStore(http_store, deployment)._project(task, "owner", SimpleNamespace())

    assert projected.status.state == TaskState.TASK_STATE_SUBMITTED


@pytest.mark.parametrize(("configured", "expected"), [(0.05, 0.1), (5.0, 5.0)])
async def test_durable_a2a_polling_honors_its_configured_floor(monkeypatch, http_store, configured, expected) -> None:
    executor = DurableAgentExecutor("agent", http_store, _Deployment(status=ExecutionStatus.RUNNING))
    delays = []

    async def stop_after_one_poll(delay):
        delays.append(delay)
        executor._closed = True

    monkeypatch.setattr(settings, "durable_poll_interval", configured)
    monkeypatch.setattr(asyncio, "sleep", stop_after_one_poll)

    await executor._wait_for_update("execution", "owner", object())

    assert delays == [expected]


async def test_durable_a2a_polling_retries_store_outages(monkeypatch, http_store) -> None:
    deployment = _Deployment()
    deployment.get = AsyncMock(
        side_effect=[ExecutionStoreError("offline"), ExecutionStoreError("still offline"), deployment.record]
    )
    updater = SimpleNamespace(task_id="task", add_artifact=AsyncMock(), complete=AsyncMock())
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    await DurableAgentExecutor("agent", http_store, deployment)._wait_for_update("execution", "owner", updater)

    assert deployment.get.await_count == 3
    assert sleep.await_count == 2
    updater.complete.assert_awaited_once()


async def test_durable_a2a_cancellation_waits_past_input_required(monkeypatch, http_store) -> None:
    waiting = _Deployment(status=ExecutionStatus.WAITING).record
    canceled = _Deployment(status=ExecutionStatus.CANCELED).record
    canceled.sequence = 1
    deployment = _Deployment()
    deployment.get = AsyncMock(side_effect=[waiting, canceled])
    updater = SimpleNamespace(
        task_id="task",
        new_agent_message=lambda parts: parts,
        requires_input=AsyncMock(),
        cancel=AsyncMock(),
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    await DurableAgentExecutor("agent", http_store, deployment)._wait_for_update(
        "execution", "owner", updater, terminal_only=True
    )

    updater.requires_input.assert_awaited_once()
    updater.cancel.assert_awaited_once()
    sleep.assert_awaited_once()


@pytest.mark.parametrize("failure", [ExecutionStoreError("offline"), ExecutionAdmissionError("test")])
async def test_durable_a2a_submission_retries_transient_failures(monkeypatch, http_store, failure) -> None:
    deployment = _Deployment()
    deployment.submit = AsyncMock(side_effect=[failure, (True, deployment.record)])
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    task = SimpleNamespace(id="task", context_id="context")
    record = await DurableAgentExecutor("agent", http_store, deployment)._submit(task, "owner", [])

    assert record is deployment.record
    assert deployment.submit.await_count == 2
    sleep.assert_awaited_once()


def test_a2a_http_rejected_submission_is_persisted_as_failed(monkeypatch, http_store) -> None:
    from a2a.types import TaskState

    deployment = _Deployment()
    deployment.submit = AsyncMock(side_effect=ValueError("request is too large"))
    app = _http_app(http_store, deployment, monkeypatch)

    with TestClient(app, headers={"A2A-Version": "1.0"}) as client:
        failed = _response_task(client.post("/durable-agent/", json=_send_payload("initial")))

    assert failed["status"]["state"] == "TASK_STATE_FAILED"
    assert http_store.tasks[failed["id"]].status.state == TaskState.TASK_STATE_FAILED


async def test_recovery_submits_a_persisted_task_without_an_execution(http_store) -> None:
    from a2a.types import TaskState

    task = _recoverable_task()
    recovery_store = _recovery_store(task)
    deployment = _Deployment()

    async def missing_until_submitted(execution_id, **kwargs):
        if deployment.execution_id is None:
            raise KeyError(execution_id)
        return await _Deployment.get(deployment, execution_id, **kwargs)

    deployment.get = missing_until_submitted
    await DurableAgentExecutor("agent", http_store, deployment)._recover_tasks(recovery_store)

    assert deployment.execution_id is not None
    assert deployment.submitted_payload["messages"][0]["content"] == [{"text": "recover me"}]
    assert task.status.state == TaskState.TASK_STATE_COMPLETED


async def test_recovery_rejects_one_invalid_persisted_task_without_aborting(http_store) -> None:
    from a2a.types import TaskState

    task = _recoverable_task()
    recovery_store = _recovery_store(task)
    deployment = _Deployment()
    deployment.get = AsyncMock(side_effect=KeyError("missing"))
    deployment.submit = AsyncMock(side_effect=ValueError("request is too large"))

    await DurableAgentExecutor("agent", http_store, deployment)._recover_tasks(recovery_store)

    assert task.status.state == TaskState.TASK_STATE_FAILED
    recovery_store.save_projection.assert_awaited_once()


async def test_recovery_skips_a_projection_conflict(http_store) -> None:
    from a2a.types import Task, TaskState

    task = Task(id="task", context_id="context")
    task.status.state = TaskState.TASK_STATE_WORKING
    recovery_store = _recovery_store(task, saved=False)
    executor = DurableAgentExecutor("agent", http_store, _Deployment())

    await executor._recover_tasks(recovery_store)

    recovery_store.save_projection.assert_awaited_once()
