import importlib.util
import json

import httpx
import pytest
from anyio import Path

from hayhooks.server.pipelines import registry
from hayhooks.server.utils.a2a_utils import create_a2a_app
from hayhooks.server.utils.deploy_utils import deploy_pipeline_files

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
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL, headers={"A2A-Version": "1.0"}) as client:
        yield client


def send_message_payload(text: str, method: str = "SendMessage") -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": {"message": {"messageId": "test-message", "role": "ROLE_USER", "parts": [{"text": text}]}},
    }


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


@pytest.mark.asyncio
async def test_status_lists_exposed_agents(a2a_client):
    response = await a2a_client.get("/status")
    assert response.status_code == 200
    # api_only has no chat completion method, so it must not be listed
    assert response.json() == {"status": "ok", "agents": ["chat_agent", "async_chat_agent"]}


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
async def test_unknown_agent_returns_404(a2a_client):
    response = await a2a_client.get("/non_existent/.well-known/agent-card.json")
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
