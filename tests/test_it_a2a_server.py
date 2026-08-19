import asyncio
import importlib.util
import json

import httpx
import pytest
from anyio import Path

from hayhooks.server.a2a.app import create_a2a_app
from hayhooks.server.pipelines import registry
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.deploy_utils import deploy_pipeline_files
from hayhooks.server.utils.module_loader import _set_method_implementation_flags

A2A_AVAILABLE = importlib.util.find_spec("a2a") is not None

# NOTE: Skip all tests in this file if a2a-sdk is not available
pytestmark = [
    pytest.mark.skipif(not A2A_AVAILABLE, reason="'a2a-sdk' package not installed"),
    pytest.mark.a2a,
]

BASE_URL = "http://a2a-test:1418"
MOCK_RESPONSE = "This is a mock response from the pipeline"


@pytest.fixture(autouse=True)
def cleanup_test_pipelines():
    yield
    registry.clear()


async def deploy_from_test_files(pipeline_name: str, dir_name: str) -> None:
    files_dir = Path(f"tests/test_files/files/{dir_name}")
    files = {path.name: await path.read_text() async for path in files_dir.iterdir()}
    deploy_pipeline_files(pipeline_name=pipeline_name, files=files, save_files=False)


@pytest.fixture
async def a2a_client():
    # A sync-streaming chat pipeline, an async-streaming one, and an API-only one (not exposed)
    await deploy_from_test_files("chat_agent", "chat_with_website_streaming")
    await deploy_from_test_files("async_chat_agent", "async_chat_with_website_streaming")
    await deploy_from_test_files("api_only", "no_chat")

    app = create_a2a_app(base_url=BASE_URL)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=transport,
            base_url=BASE_URL,
            headers={"A2A-Version": "1.0"},
        ) as client,
    ):
        yield client


def send_message_payload(text: str, method: str = "SendMessage") -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": {"message": {"messageId": "test-message", "role": "ROLE_USER", "parts": [{"text": text}]}},
    }


def get_task_payload(task_id: str, method: str = "GetTask") -> dict:
    return {"jsonrpc": "2.0", "id": "get-task", "method": method, "params": {"id": task_id}}


def cancel_task_payload(task_id: str) -> dict:
    return {"jsonrpc": "2.0", "id": "cancel-task", "method": "CancelTask", "params": {"id": task_id}}


def extract_task(response_payload: dict) -> dict:
    result = response_payload["result"]
    return result.get("task", result)


def artifact_text(task: dict) -> str:
    return "".join(part["text"] for artifact in task.get("artifacts", []) for part in artifact["parts"])


def send_message_v0_3_payload(text: str, method: str = "message/send") -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": {
            "message": {
                "messageId": "test-message-v03",
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }


class ControlledLongRunningWrapper(BasePipelineWrapper):
    emit_progress_chunk = True

    def setup(self):
        self.pipeline = object()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict):
        async def generator():
            self.entered.set()
            if self.emit_progress_chunk:
                yield "progress "
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            yield "done"

        return generator()


def register_test_wrapper(name: str, wrapper_cls: type[BasePipelineWrapper]) -> BasePipelineWrapper:
    wrapper = wrapper_cls()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add(
        name,
        wrapper,
        metadata={"description": f"{name} description", "skip_a2a": wrapper.skip_a2a, "a2a_card": wrapper.a2a_card},
    )
    return wrapper


@pytest.mark.asyncio
async def test_status_lists_exposed_agents(a2a_client):
    response = await a2a_client.get("/status")
    assert response.status_code == 200
    # api_only has no chat completion method, so it must not be listed
    body = response.json()
    assert body["status"] == "ok"
    assert body["agents"] == ["chat_agent", "async_chat_agent"]
    assert body["components"]["task_store"]["healthy"]
    assert body["components"]["maintenance"]["healthy"]
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_agent_card_is_served(a2a_client):
    response = await a2a_client.get("/chat_agent/.well-known/agent-card.json")
    assert response.status_code == 200

    card = response.json()
    assert card["name"] == "chat_agent"
    assert card["capabilities"]["streaming"] is True
    assert card["supportedInterfaces"] == [{"url": f"{BASE_URL}/chat_agent/", "protocolBinding": "JSONRPC"}]
    assert card["skills"][0]["id"] == "chat_agent"


@pytest.mark.asyncio
async def test_non_chat_pipeline_is_not_exposed(a2a_client):
    response = await a2a_client.get("/api_only/.well-known/agent-card.json")
    assert response.status_code == 404

    response = await a2a_client.post("/api_only/", json=send_message_payload("hi"))
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_send_message_returns_completed_task(a2a_client):
    response = await a2a_client.post("/chat_agent/", json=send_message_payload("What is Hayhooks?"))
    assert response.status_code == 200

    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"

    artifacts = task["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "response"
    text = "".join(part["text"] for part in artifacts[0]["parts"])
    assert text.strip() == MOCK_RESPONSE


@pytest.mark.asyncio
async def test_send_message_async_pipeline(a2a_client):
    response = await a2a_client.post("/async_chat_agent/", json=send_message_payload("What is Redis?"))
    assert response.status_code == 200

    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    text = "".join(part["text"] for part in task["artifacts"][0]["parts"])
    assert "Redis" in text


@pytest.mark.asyncio
async def test_send_streaming_message_streams_sse_events(a2a_client):
    payload = send_message_payload("What is Hayhooks?", method="SendStreamingMessage")

    events = []
    async with a2a_client.stream("POST", "/chat_agent/", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events.extend(
            [json.loads(line[len("data:") :]) async for line in response.aiter_lines() if line.startswith("data:")]
        )

    kinds = [next(iter(event["result"].keys())) for event in events]
    assert kinds[0] == "task"
    assert kinds[1] == "statusUpdate"
    assert "artifactUpdate" in kinds
    assert kinds[-1] == "statusUpdate"
    assert events[-1]["result"]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"

    streamed_text = "".join(
        part["text"]
        for event in events
        if "artifactUpdate" in event["result"]
        for part in event["result"]["artifactUpdate"]["artifact"]["parts"]
    )
    assert streamed_text.strip() == MOCK_RESPONSE


@pytest.mark.asyncio
async def test_a2a_v0_3_message_send_is_supported(a2a_client):
    response = await a2a_client.post(
        "/chat_agent/", json=send_message_v0_3_payload("What is Hayhooks?"), headers={"A2A-Version": "0.3"}
    )
    assert response.status_code == 200

    task = response.json()["result"]
    assert task["status"]["state"] == "completed"
    text = "".join(part["text"] for artifact in task["artifacts"] for part in artifact["parts"])
    assert text.strip() == MOCK_RESPONSE


@pytest.mark.asyncio
async def test_a2a_v1_method_with_v0_3_header_is_rejected(a2a_client):
    response = await a2a_client.post("/chat_agent/", json=send_message_payload("hi"), headers={"A2A-Version": "0.3"})
    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32009


@pytest.fixture
async def long_running_client():
    wrapper = register_test_wrapper("long_agent", ControlledLongRunningWrapper)
    app = create_a2a_app(base_url=BASE_URL)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL, headers={"A2A-Version": "1.0"}) as client:
        yield client, wrapper


async def poll_task(client: httpx.AsyncClient, task_id: str, ready, description: str, *, legacy: bool = False) -> dict:
    """Poll one task until *ready* accepts its projection; ``legacy`` uses the v0.3 alias."""
    payload = get_task_payload(task_id, method="tasks/get" if legacy else "GetTask")
    headers = {"A2A-Version": "0.3"} if legacy else None
    last_task = None
    for _ in range(50):
        response = await client.post("/long_agent/", json=payload, headers=headers)
        assert response.status_code == 200
        last_task = extract_task(response.json())
        if ready(last_task):
            return last_task
        await asyncio.sleep(0.01)
    msg = f"Task {task_id} did not {description}. Last task: {last_task}"
    raise AssertionError(msg)


async def start_detached_task(client: httpx.AsyncClient) -> dict:
    """Send one message that returns before the agent has finished."""
    payload = send_message_payload("start")
    payload["params"]["configuration"] = {"returnImmediately": True}
    response = await client.post("/long_agent/", json=payload)
    assert response.status_code == 200
    return extract_task(response.json())


async def test_detached_send_returns_non_terminal_task(long_running_client):
    client, wrapper = long_running_client

    task = await start_detached_task(client)

    assert task["id"]
    assert task["contextId"]
    assert task["status"]["state"] in {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING"}
    assert wrapper.entered.is_set()
    assert not wrapper.release.is_set()

    progress_task = await poll_task(
        client, task["id"], lambda task: artifact_text(task) == "progress ", "expose its progress artifact"
    )
    assert progress_task["status"]["state"] == "TASK_STATE_WORKING"

    wrapper.release.set()
    # The v0.3 alias must project the same task with lowercase states.
    last_task = await poll_task(
        client,
        task["id"],
        lambda task: task["status"]["state"] == "completed",
        "complete",
        legacy=True,
    )
    assert artifact_text(last_task) == "progress done"


async def test_default_send_remains_blocking(long_running_client):
    client, wrapper = long_running_client

    request_task = asyncio.create_task(client.post("/long_agent/", json=send_message_payload("start")))
    await asyncio.wait_for(wrapper.entered.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not request_task.done()

    wrapper.release.set()
    response = await asyncio.wait_for(request_task, timeout=1)
    task = extract_task(response.json())
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert artifact_text(task) == "progress done"


async def test_a2a_v0_3_blocking_false_returns_active_task(long_running_client):
    client, wrapper = long_running_client
    payload = send_message_v0_3_payload("start")
    payload["params"]["configuration"] = {"blocking": False}

    response = await client.post("/long_agent/", json=payload, headers={"A2A-Version": "0.3"})
    assert response.status_code == 200
    task = extract_task(response.json())
    assert task["status"]["state"] in {"submitted", "working"}

    wrapper.release.set()
    completed_task = await poll_task(
        client, task["id"], lambda task: task["status"]["state"] == "TASK_STATE_COMPLETED", "complete"
    )
    assert artifact_text(completed_task) == "progress done"


async def test_subscribe_to_active_task(long_running_client):
    client, wrapper = long_running_client
    task = await start_detached_task(client)

    subscribe_payload = get_task_payload(task["id"], method="SubscribeToTask")

    async def read_subscription_events() -> list[dict]:
        events = []
        async with client.stream("POST", "/long_agent/", json=subscribe_payload) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[len("data:") :]))
                    if (
                        events[-1]["result"].get("statusUpdate", {}).get("status", {}).get("state")
                        == "TASK_STATE_COMPLETED"
                    ):
                        break
        return events

    subscription_task = asyncio.create_task(read_subscription_events())
    await asyncio.sleep(0.01)
    wrapper.release.set()
    events = await asyncio.wait_for(subscription_task, timeout=1)

    assert next(iter(events[0]["result"].keys())) == "task"
    assert "artifactUpdate" in [next(iter(event["result"].keys())) for event in events]
    assert events[-1]["result"]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"


async def test_cooperative_async_cancellation(long_running_client):
    client, wrapper = long_running_client
    task = await start_detached_task(client)

    await asyncio.wait_for(wrapper.entered.wait(), timeout=1)
    cancel_response = await client.post("/long_agent/", json=cancel_task_payload(task["id"]))

    assert cancel_response.status_code == 200
    canceled_task = extract_task(cancel_response.json())
    assert canceled_task["status"]["state"] == "TASK_STATE_CANCELED"
    assert artifact_text(canceled_task) == "progress "
    assert wrapper.cancelled.is_set()
