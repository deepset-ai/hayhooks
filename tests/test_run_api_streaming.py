from pathlib import Path

TEST_FILES_ROOT = Path(__file__).parent / "test_files/files"

RUN_API_STREAMING_FILES = {
    "pipeline_wrapper.py": (TEST_FILES_ROOT / "run_api_streaming" / "pipeline_wrapper.py").read_text(),
}

ASYNC_RUN_API_STREAMING_FILES = {
    "pipeline_wrapper.py": (TEST_FILES_ROOT / "async_run_api_streaming" / "pipeline_wrapper.py").read_text(),
}

RUN_API_EVENT_FILES = {
    "pipeline_wrapper.py": (TEST_FILES_ROOT / "run_api_openwebui_event" / "pipeline_wrapper.py").read_text(),
}


def _collect_stream_text(response) -> str:
    return "".join(chunk for chunk in response.iter_text())


def test_run_api_streaming_response(client, deploy_files):
    pipeline_name = "test_run_api_streaming"
    response = deploy_files(client, pipeline_name=pipeline_name, pipeline_files=RUN_API_STREAMING_FILES)
    assert response.status_code == 200

    with client.stream(
        "POST",
        f"/{pipeline_name}/run",
        json={"query": "streaming response"},
    ) as stream_response:
        assert stream_response.status_code == 200
        assert stream_response.headers["content-type"] == "text/plain; charset=utf-8"
        streamed = _collect_stream_text(stream_response)

    assert streamed == "streaming response "


def test_run_api_async_streaming_response(client, deploy_files):
    pipeline_name = "test_run_api_async_streaming"
    response = deploy_files(client, pipeline_name=pipeline_name, pipeline_files=ASYNC_RUN_API_STREAMING_FILES)
    assert response.status_code == 200

    with client.stream(
        "POST",
        f"/{pipeline_name}/run",
        json={"query": "async streaming"},
    ) as stream_response:
        assert stream_response.status_code == 200
        assert stream_response.headers["content-type"] == "text/plain; charset=utf-8"
        streamed = _collect_stream_text(stream_response)

    assert streamed == "async streaming "


def test_run_api_streaming_rejects_openwebui_events(client, deploy_files):
    pipeline_name = "test_run_api_streaming_event"
    response = deploy_files(client, pipeline_name=pipeline_name, pipeline_files=RUN_API_EVENT_FILES)
    assert response.status_code == 200

    with client.stream(
        "POST",
        f"/{pipeline_name}/run",
        json={"query": "should fail"},
    ) as stream_response:
        assert stream_response.status_code == 200
        assert stream_response.headers["content-type"] == "text/plain; charset=utf-8"
        streamed = _collect_stream_text(stream_response)

    assert streamed == ""
