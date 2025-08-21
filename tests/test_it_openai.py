import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from hayhooks.server.pipelines import registry
from hayhooks.server.routers.deploy import DeployResponse
from hayhooks.server.routers.openai import ChatCompletion, ChatRequest, ModelObject, ModelsResponse
from hayhooks.settings import settings


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()
    if Path(settings.pipelines_dir).exists():
        shutil.rmtree(settings.pipelines_dir)
    yield


def collect_chunks(response):
    return [event for event in response.iter_lines() if event]


TEST_FILES_DIR = Path(__file__).parent / "test_files/files/chat_with_website"
SAMPLE_PIPELINE_FILES = {
    "pipeline_wrapper.py": (TEST_FILES_DIR / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (TEST_FILES_DIR / "chat_with_website.yml").read_text(),
}

TEST_FILES_DIR_STREAMING = Path(__file__).parent / "test_files/files/chat_with_website_streaming"
SAMPLE_PIPELINE_FILES_STREAMING = {
    "pipeline_wrapper.py": (TEST_FILES_DIR_STREAMING / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (TEST_FILES_DIR_STREAMING / "chat_with_website.yml").read_text(),
}

TEST_FILES_DIR_ASYNC_STREAMING = Path(__file__).parent / "test_files/files/async_chat_with_website_streaming"
SAMPLE_PIPELINE_FILES_ASYNC_STREAMING = {
    "pipeline_wrapper.py": (TEST_FILES_DIR_ASYNC_STREAMING / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (TEST_FILES_DIR_ASYNC_STREAMING / "chat_with_website.yml").read_text(),
}


def test_get_models_empty(client):
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {"data": [], "object": "list"}


def test_get_models(client) -> None:
    pipeline_data = {"name": "test_pipeline", "files": SAMPLE_PIPELINE_FILES}

    response = client.post("/deploy_files", json=pipeline_data)
    assert response.status_code == 200
    assert (
        response.json()
        == DeployResponse(name="test_pipeline", success=True, endpoint=f"/{pipeline_data['name']}/run").model_dump()
    )

    response = client.get("/models")
    response_data = response.json()

    expected_response = ModelsResponse(
        object="list",
        data=[
            ModelObject(
                id="test_pipeline",
                name="test_pipeline",
                object="model",
                created=response_data["data"][0]["created"],  # type: ignore
                owned_by="hayhooks",
            )
        ],
    )

    assert response.status_code == 200
    assert response_data == expected_response.model_dump()


def test_chat_completion_success(client, deploy_files):
    pipeline_data = {"name": "test_pipeline", "files": SAMPLE_PIPELINE_FILES}

    response = deploy_files(client, pipeline_data["name"], pipeline_data["files"])
    assert response.status_code == 200
    assert (
        response.json()
        == DeployResponse(name="test_pipeline", success=True, endpoint=f"/{pipeline_data['name']}/run").model_dump()
    )

    # This is a sample request coming from openai-webui
    request = ChatRequest(
        stream=False,
        model="test_pipeline",
        messages=[{"role": "user", "content": "what is Redis?"}],
        features={"web_search": False},
        session_id="_Qtpw_fE4g9dMKVKAAAP",
        chat_id="7d436049-d316-462a-b1c6-c61740f979c9",
        id="b8050e7d-d6ec-4dbc-b69e-6b38d36d847e",
        background_tasks={"title_generation": True, "tags_generation": True},
    )

    response = client.post("/chat/completions", json=request.model_dump())
    assert response.status_code == 200

    response_data = response.json()
    chat_completion = ChatCompletion(**response_data)
    assert chat_completion.object == "chat.completion"
    assert chat_completion.model == "test_pipeline"
    assert len(chat_completion.choices) == 1
    assert chat_completion.choices[0].message.content
    assert chat_completion.choices[0].index == 0
    assert chat_completion.choices[0].logprobs is None


def test_chat_completion_invalid_model(client):
    request = ChatRequest(model="nonexistent_model", messages=[{"role": "user", "content": "Hello"}])

    response = client.post("/chat/completions", json=request.model_dump())
    assert response.status_code == 404


def test_chat_completion_not_implemented(client, deploy_files) -> None:
    pipeline_file = Path(__file__).parent / "test_files/files/no_chat/pipeline_wrapper.py"
    pipeline_data = {"name": "test_pipeline_no_chat", "files": {"pipeline_wrapper.py": pipeline_file.read_text()}}

    response = deploy_files(client, pipeline_data["name"], pipeline_data["files"])
    assert response.status_code == 200
    assert (
        response.json()
        == DeployResponse(
            name="test_pipeline_no_chat", success=True, endpoint=f"/{pipeline_data['name']}/run"
        ).model_dump()
    )

    request = ChatRequest(model="test_pipeline_no_chat", messages=[{"role": "user", "content": "Hello"}])

    response = client.post("/chat/completions", json=request.model_dump())
    assert response.status_code == 501

    err_body: dict[str, Any] = response.json()
    assert err_body["detail"] == "Chat endpoint not implemented for this model"


def _test_streaming_chat_completion(client, deploy_files, pipeline_name: str, pipeline_files: dict[str, str]):
    """
    Helper function to test the streaming chat completion.
    Used in tests for both sync and async streaming.
    """
    response = deploy_files(client, pipeline_name, pipeline_files)
    assert response.status_code == 200
    assert (
        response.json()
        == DeployResponse(name=pipeline_name, success=True, endpoint=f"/{pipeline_name}/run").model_dump()
    )

    request = ChatRequest(
        model=pipeline_name,
        messages=[{"role": "user", "content": "what is Redis?"}],
    )

    response = client.post("/chat/completions", json=request.model_dump())

    # response is a stream of SSE events
    assert response.status_code == 200

    headers: dict[str, Any] = response.headers
    assert headers["Content-Type"] == "text/event-stream; charset=utf-8"

    # collect the chunks
    chunks = collect_chunks(response)

    # check if the chunks are valid
    assert len(chunks) > 0
    assert chunks[0].startswith("data:")
    assert chunks[-1].startswith("data:")
    return chunks


def test_chat_completion_streaming(client, deploy_files) -> None:
    pipeline_name = "test_pipeline_streaming"
    pipeline_files = SAMPLE_PIPELINE_FILES_STREAMING
    chunks = _test_streaming_chat_completion(client, deploy_files, pipeline_name, pipeline_files)

    # check if the chunks are valid ChatCompletion objects
    sample_chunk = chunks[1]
    chat_completion = ChatCompletion(**json.loads(sample_chunk.split("data:")[1]))  # type: ignore
    assert chat_completion.object == "chat.completion.chunk"
    assert chat_completion.model == pipeline_name
    assert chat_completion.choices[0].delta.content
    assert chat_completion.choices[0].delta.role == "assistant"
    assert chat_completion.choices[0].index == 0
    assert chat_completion.choices[0].logprobs is None

    # check if last chunk contains a delta with empty content
    last_chunk = chunks[-1]
    last_chat_completion = ChatCompletion(**json.loads(last_chunk.split("data:")[1]))  # type: ignore
    assert last_chat_completion.choices[0].delta.content == ""
    assert last_chat_completion.choices[0].delta.role == "assistant"
    assert last_chat_completion.choices[0].index == 0
    assert last_chat_completion.choices[0].logprobs is None


def test_chat_completion_concurrent_requests(client, deploy_files):
    pipeline_data = {"name": "test_pipeline_streaming", "files": SAMPLE_PIPELINE_FILES_STREAMING}

    response = deploy_files(client, pipeline_data["name"], pipeline_data["files"])
    assert response.status_code == 200
    assert (
        response.json()
        == DeployResponse(
            name="test_pipeline_streaming", success=True, endpoint=f"/{pipeline_data['name']}/run"
        ).model_dump()
    )

    request_1 = ChatRequest(model="test_pipeline_streaming", messages=[{"role": "user", "content": "what is Redis?"}])
    request_2 = ChatRequest(model="test_pipeline_streaming", messages=[{"role": "user", "content": "what is MongoDB?"}])

    # run the requests concurrently
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(client.post, "/chat/completions", json=request_1.model_dump()),
            executor.submit(client.post, "/chat/completions", json=request_2.model_dump()),
        ]
        results = [future.result() for future in futures]

    assert results[0].status_code == 200
    assert results[1].status_code == 200

    chunks_1 = collect_chunks(results[0])
    chunks_2 = collect_chunks(results[1])

    # check if the responses are valid
    assert "Redis" in chunks_1[0]  # "Redis" is the first chunk (see pipeline_wrapper.py)
    assert "This" in chunks_2[0]  # "This" is the first chunk (see pipeline_wrapper.py)


def test_async_chat_completion_streaming(client, deploy_files) -> None:
    pipeline_name = "test_pipeline_async_streaming"
    pipeline_files = SAMPLE_PIPELINE_FILES_ASYNC_STREAMING
    chunks = _test_streaming_chat_completion(client, deploy_files, pipeline_name, pipeline_files)

    # check if the chunks are valid ChatCompletion objects
    sample_chunk = chunks[1]
    chat_completion = ChatCompletion(**json.loads(sample_chunk.split("data:")[1]))  # type: ignore
    assert chat_completion.object == "chat.completion.chunk"
    assert chat_completion.model == pipeline_name
    assert chat_completion.choices[0].delta.content

    # check if last chunk contains a delta with empty content
    last_chunk = chunks[-1]
    last_chat_completion = ChatCompletion(**json.loads(last_chunk.split("data:")[1]))  # type: ignore
    assert last_chat_completion.choices[0].delta.content == ""
    assert last_chat_completion.choices[0].delta.role == "assistant"
    assert last_chat_completion.choices[0].index == 0
    assert last_chat_completion.choices[0].logprobs is None
