import asyncio
import importlib.metadata
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hayhooks.a2a import A2APipelineWrapper, TaskStoreProvider
from hayhooks.execution import ExecutionStatus
from hayhooks.server.a2a.executor import (
    DurableAgentExecutor,
    _Projection,
    _ProjectionConflictError,
    _RecoveryEventQueue,
)
from hayhooks.server.a2a.imports import TaskStore, TaskUpdater
from hayhooks.server.a2a.runtime import A2ARuntime
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.a2a_utils import create_a2a_app
from hayhooks.settings import settings

pytestmark = pytest.mark.skipif(
    not importlib.metadata.version("haystack-ai").startswith("3."), reason="durable execution requires Haystack 3"
)


class _RecordingQueue:
    def __init__(self) -> None:
        self.events = []

    async def enqueue_event(self, event) -> None:
        self.events.append(event)


class _RecoverableStore:
    def __init__(self, tasks) -> None:
        self.tasks = tasks
        self.saved = []

    async def recoverable_tasks(self):
        return self.tasks

    async def get_for_execution(self, task_id):
        return next((task for task in self.tasks if task.id == task_id), None)

    async def save_for_execution(self, task):
        self.saved.append(task)


class _Deployment:
    def __init__(self, status=ExecutionStatus.COMPLETED) -> None:
        self.record = SimpleNamespace(
            status=status, progress=[], result={"answer": "recovered"}, error=None, sequence=0
        )
        self.resume_update = None
        self.cancel_requested = False
        self.changed_calls = 0

    async def get(self, _execution_id):
        return self.record

    async def get_changed(self, known_sequences):
        self.changed_calls += 1
        return {
            execution_id: self.record
            for execution_id, sequence in known_sequences.items()
            if sequence != self.record.sequence
        }

    async def resume(self, _execution_id, update):
        self.resume_update = update
        self.record.status = ExecutionStatus.COMPLETED
        self.record.sequence += 1
        return True

    async def request_cancel(self, _execution_id):
        self.cancel_requested = True
        self.record.status = ExecutionStatus.CANCELED
        self.record.sequence += 1
        return True


class _HTTPDeployment(_Deployment):
    def __init__(self, status=ExecutionStatus.COMPLETED) -> None:
        super().__init__(status=status)
        self.execution_id = None
        self.record.result = {"last_message": {"content": "recovered"}}

    async def start(self):
        return None

    async def submit_agent_messages(self, _messages, *, execution_id=None):
        self.execution_id = execution_id
        return True, self.record

    async def get(self, execution_id):
        if self.execution_id is not None and execution_id != self.execution_id:
            raise KeyError(execution_id)
        return self.record

    async def resume(self, execution_id, update):
        self.execution_id = execution_id
        self.resume_update = update
        self.record.status = ExecutionStatus.COMPLETED
        self.record.result = {"last_message": {"content": "resumed"}}
        self.record.sequence += 1
        return True


class _DurableHTTPWrapper(A2APipelineWrapper):
    durable = True

    def setup(self):
        self.pipeline = object()


class _HTTPRecoverableStore(TaskStore):
    """TaskStore-compatible persistence used to exercise the full A2A app path."""

    def __init__(self, tasks=()) -> None:
        self.tasks = {task.id: task for task in tasks}

    async def save(self, task, _context):
        self.tasks[task.id] = task

    async def get(self, task_id, _context):
        return self.tasks.get(task_id)

    async def list(self, _params, _context):
        from a2a.types import ListTasksResponse

        return ListTasksResponse(tasks=list(self.tasks.values()), page_size=len(self.tasks), total_size=len(self.tasks))

    async def delete(self, task_id, _context):
        self.tasks.pop(task_id, None)

    async def recoverable_tasks(self):
        return list(self.tasks.values())

    async def get_for_execution(self, task_id):
        return self.tasks.get(task_id)

    async def save_for_execution(self, task):
        self.tasks[task.id] = task


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


def _register_http_wrapper():
    wrapper = _DurableHTTPWrapper()
    wrapper.setup()
    registry.add("durable-agent", wrapper, metadata={"description": "durable agent"})
    return wrapper


def _http_app(store, deployment, monkeypatch):
    _register_http_wrapper()

    def deployment_method(*_args, **_kwargs):
        return deployment

    monkeypatch.setattr("hayhooks.server.a2a.executor.durable_runtime.deployment", deployment_method)
    monkeypatch.setattr("hayhooks.durable_runtime.durable_runtime.deployment", deployment_method)
    runtime = A2ARuntime(task_store_provider=_HTTPStoreProvider(store))
    return create_a2a_app(base_url="http://a2a-test:1418", runtime=runtime)


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    yield
    registry.clear()


def _task(text="initial"):
    from a2a.helpers import new_task_from_user_message, new_text_message
    from a2a.types import Role, TaskState

    task = new_task_from_user_message(new_text_message(text, role=Role.ROLE_USER))
    task.status.state = TaskState.TASK_STATE_WORKING
    return task


def test_a2a_http_lifespan_recovers_persisted_projection(monkeypatch) -> None:
    from a2a.types import TaskState

    task = _task()
    store = _HTTPRecoverableStore([task])
    deployment = _HTTPDeployment(status=ExecutionStatus.COMPLETED)
    deployment.execution_id = task.id
    app = _http_app(store, deployment, monkeypatch)

    with TestClient(app, headers={"A2A-Version": "1.0"}) as client:
        for _ in range(100):
            recovered = _response_task(client.post("/durable-agent/", json=_get_payload(task.id)))
            if recovered["status"]["state"] == "TASK_STATE_COMPLETED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("A2A task projection was not recovered during app startup")

    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert recovered["artifacts"][-1]["name"] == "durable-result"


def test_a2a_http_waiting_task_resumes_with_follow_up(monkeypatch) -> None:
    store = _HTTPRecoverableStore()
    deployment = _HTTPDeployment(status=ExecutionStatus.WAITING)
    app = _http_app(store, deployment, monkeypatch)

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


def test_a2a_http_cancellation_reaches_durable_execution(monkeypatch) -> None:
    store = _HTTPRecoverableStore()
    deployment = _HTTPDeployment(status=ExecutionStatus.RUNNING)
    app = _http_app(store, deployment, monkeypatch)

    with TestClient(app, headers={"A2A-Version": "1.0"}) as client:
        active = _response_task(client.post("/durable-agent/", json=_send_payload("initial", return_immediately=True)))
        cancel_response = client.post("/durable-agent/", json=_cancel_payload(active["id"]))
        assert cancel_response.status_code == 200, cancel_response.text
        assert "error" not in cancel_response.json(), cancel_response.json()
        assert deployment.cancel_requested
        for _ in range(100):
            canceled = _response_task(client.post("/durable-agent/", json=_get_payload(active["id"])))
            if canceled["status"]["state"] == "TASK_STATE_CANCELED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("durable A2A task did not project cancellation")

    assert deployment.cancel_requested


@pytest.mark.asyncio
async def test_executor_start_recovers_persisted_task_projection_and_cleans_watcher(monkeypatch) -> None:
    from a2a.types import TaskState

    task = _task()
    store = _RecoverableStore([task])
    deployment = _Deployment()
    monkeypatch.setattr("hayhooks.server.a2a.executor.durable_runtime.deployment", lambda *_args: deployment)
    executor = DurableAgentExecutor(SimpleNamespace(), "agent", store)

    await executor.start()
    for _ in range(100):
        if task.status.state == TaskState.TASK_STATE_COMPLETED and not executor._watchers:
            break
        await asyncio.sleep(0.001)
    else:
        pytest.fail("persisted A2A task projection was not recovered")

    assert store.saved
    assert task.artifacts[-1].name == "durable-result"
    assert task.artifacts[-1].parts[0].text == '{"answer": "recovered"}'


@pytest.mark.asyncio
async def test_reconciler_batches_fairly_without_starving_later_tasks(monkeypatch) -> None:
    monkeypatch.setattr(settings, "a2a_projection_batch_size", 2)
    monkeypatch.setattr(settings, "a2a_projection_interval", 0.001)
    executor = DurableAgentExecutor(SimpleNamespace(), "agent", _RecoverableStore([]))
    execution_ids = [f"execution-{index}" for index in range(5)]
    executor._projections = {execution_id: _Projection(updater=object()) for execution_id in execution_ids}
    reconciled = []

    async def reconcile(batch):
        for execution_id, _projection in batch:
            reconciled.append(execution_id)
            executor._projections.pop(execution_id)
            if len(reconciled) == len(execution_ids):
                executor._closed = True

    monkeypatch.setattr(executor, "_reconcile_batch", reconcile)

    await asyncio.wait_for(executor._reconcile_loop(), timeout=1)

    assert reconciled == execution_ids


@pytest.mark.asyncio
async def test_reconciler_batch_skips_full_records_until_sequence_changes(monkeypatch) -> None:
    deployment = _Deployment(status=ExecutionStatus.RUNNING)
    monkeypatch.setattr("hayhooks.server.a2a.executor.durable_runtime.deployment", lambda *_args: deployment)
    executor = DurableAgentExecutor(SimpleNamespace(), "agent", _RecoverableStore([]))
    projection = _Projection(updater=object())
    batch = [("execution", projection)]

    await executor._reconcile_batch(batch)
    await executor._reconcile_batch(batch)

    assert deployment.changed_calls == 2
    assert projection.record_sequence == deployment.record.sequence
    assert projection.failures == 0


@pytest.mark.asyncio
async def test_waiting_follow_up_resumes_with_only_the_new_user_message(monkeypatch) -> None:
    from a2a.helpers import new_text_message
    from a2a.types import Role, TaskState

    task = _task()
    task.status.state = TaskState.TASK_STATE_INPUT_REQUIRED
    task.history.append(new_text_message("previous answer", role=Role.ROLE_AGENT))
    follow_up = new_text_message("follow up", role=Role.ROLE_USER)
    follow_up.task_id = task.id
    follow_up.context_id = task.context_id
    context = SimpleNamespace(current_task=task, message=follow_up)
    deployment = _Deployment(status=ExecutionStatus.WAITING)
    monkeypatch.setattr("hayhooks.server.a2a.executor.durable_runtime.deployment", lambda *_args: deployment)
    executor = DurableAgentExecutor(SimpleNamespace(), "agent", _RecoverableStore([]))

    await executor.execute(context, _RecordingQueue())
    await executor.close()

    assert deployment.resume_update == {
        "messages": [{"role": "user", "meta": {}, "name": None, "content": [{"text": "follow up"}]}]
    }


@pytest.mark.asyncio
async def test_running_task_rejects_follow_up_message(monkeypatch) -> None:
    from a2a.helpers import new_text_message
    from a2a.types import Role

    task = _task()
    follow_up = new_text_message("follow up", role=Role.ROLE_USER)
    follow_up.task_id = task.id
    context = SimpleNamespace(current_task=task, message=follow_up)
    deployment = _Deployment(status=ExecutionStatus.RUNNING)
    monkeypatch.setattr("hayhooks.server.a2a.executor.durable_runtime.deployment", lambda *_args: deployment)
    executor = DurableAgentExecutor(SimpleNamespace(), "agent", _RecoverableStore([]))

    from hayhooks.server.a2a.imports import InvalidParamsError

    with pytest.raises(InvalidParamsError, match="cannot accept another message"):
        await executor.execute(context, _RecordingQueue())
    await executor.close()


@pytest.mark.asyncio
async def test_resume_race_rejects_losing_follow_up(monkeypatch) -> None:
    from a2a.helpers import new_text_message
    from a2a.types import Role

    class LostResumeDeployment(_Deployment):
        async def resume(self, _execution_id, _update):
            return False

    task = _task()
    follow_up = new_text_message("follow up", role=Role.ROLE_USER)
    follow_up.task_id = task.id
    context = SimpleNamespace(current_task=task, message=follow_up)
    deployment = LostResumeDeployment(status=ExecutionStatus.WAITING)
    monkeypatch.setattr("hayhooks.server.a2a.executor.durable_runtime.deployment", lambda *_args: deployment)
    executor = DurableAgentExecutor(SimpleNamespace(), "agent", _RecoverableStore([]))

    from hayhooks.server.a2a.imports import InvalidParamsError

    with pytest.raises(InvalidParamsError, match="no longer accepting follow-up"):
        await executor.execute(context, _RecordingQueue())
    await executor.close()


@pytest.mark.asyncio
async def test_durable_executor_forwards_cancellation_to_execution(monkeypatch) -> None:
    task = _task()
    context = SimpleNamespace(current_task=task, message=None)
    deployment = _Deployment(status=ExecutionStatus.RUNNING)
    monkeypatch.setattr("hayhooks.server.a2a.executor.durable_runtime.deployment", lambda *_args: deployment)
    executor = DurableAgentExecutor(SimpleNamespace(), "agent", _RecoverableStore([]))

    await executor.cancel(context, _RecordingQueue())

    assert deployment.cancel_requested


@pytest.mark.asyncio
async def test_cancel_projection_keeps_redis_lease_fencing_until_terminal(monkeypatch) -> None:
    class PendingCancelDeployment(_Deployment):
        async def request_cancel(self, _execution_id):
            self.cancel_requested = True
            return True

    class LeasedStore(_RecoverableStore):
        def __init__(self, tasks) -> None:
            super().__init__(tasks)
            self.projection_saves = []
            self.released = []

        async def acquire_projection(self, _task_id, *, lease_ms):
            assert lease_ms > 0
            return "lease-token"

        async def renew_projection(self, _task_id, token, *, lease_ms):
            return token == "lease-token" and lease_ms > 0

        async def release_projection(self, task_id, token):
            self.released.append((task_id, token))

        async def projection_version(self, _task_id):
            return 4

        async def save_projection(self, task, token, expected_version):
            self.projection_saves.append((task.id, token, expected_version))
            return expected_version + 1

    task = _task()
    context = SimpleNamespace(current_task=task, message=None)
    deployment = PendingCancelDeployment(status=ExecutionStatus.RUNNING)
    store = LeasedStore([task])
    monkeypatch.setattr("hayhooks.server.a2a.executor.durable_runtime.deployment", lambda *_args: deployment)
    executor = DurableAgentExecutor(SimpleNamespace(), "agent", store)

    await executor.cancel(context, _RecordingQueue())
    assert task.id in executor._projections
    assert executor._projections[task.id].lease_token == "lease-token"

    deployment.record.status = ExecutionStatus.CANCELED
    deployment.record.sequence += 1
    await executor._reconcile_one(task.id, executor._projections[task.id])
    await executor.close()

    assert store.projection_saves
    assert all(token == "lease-token" for _, token, _ in store.projection_saves)
    assert not store.saved


@pytest.mark.asyncio
async def test_stale_projection_restores_task_before_event_retry() -> None:
    from a2a.types import TaskState

    class Store:
        stale = True

        async def save_projection(self, _task, _token, expected_version):
            return -2 if self.stale else expected_version + 1

    task = _task()
    store = Store()
    queue = _RecoveryEventQueue(store, task, lease_token="lease", expected_version=3)
    updater = TaskUpdater(queue, task.id, task.context_id)

    with pytest.raises(_ProjectionConflictError):
        await updater.complete()
    assert task.status.state == TaskState.TASK_STATE_WORKING

    store.stale = False
    await TaskUpdater(queue, task.id, task.context_id).complete()
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
