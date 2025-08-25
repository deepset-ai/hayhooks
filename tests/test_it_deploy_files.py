import shutil
from pathlib import Path
from typing import Any

import pytest

from hayhooks.server.pipelines.registry import registry
from hayhooks.server.routers.deploy import DeployResponse
from hayhooks.server.routers.status import PipelineStatusResponse


def clear_registry_and_files(pipelines_dir: str):
    registry.clear()
    if Path(pipelines_dir).exists():
        shutil.rmtree(pipelines_dir)


@pytest.fixture(autouse=True)
def cleanup(test_settings):
    clear_registry_and_files(test_settings.pipelines_dir)
    yield


TEST_FILES_DIR = Path(__file__).parent / "test_files/files"
MISSING_METHODS_DIR = TEST_FILES_DIR / "missing_methods"
SETUP_ERROR_DIR = TEST_FILES_DIR / "setup_error"
RUN_API_ERROR_DIR = TEST_FILES_DIR / "run_api_error"
UPLOAD_FILES_DIR = TEST_FILES_DIR / "upload_files"
ASYNC_TEST_FILES_DIR = TEST_FILES_DIR / "async_chat_with_website"

ASYNC_SAMPLE_PIPELINE_FILES = {
    "pipeline_wrapper.py": (ASYNC_TEST_FILES_DIR / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (ASYNC_TEST_FILES_DIR / "chat_with_website.yml").read_text(),
}
SAMPLE_PIPELINE_FILES = {
    "pipeline_wrapper.py": (TEST_FILES_DIR / "chat_with_website" / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (TEST_FILES_DIR / "chat_with_website" / "chat_with_website.yml").read_text(),
}

SAMPLE_PIPELINE_FILES_NO_CHAT_COMPLETION = {
    "pipeline_wrapper.py": (TEST_FILES_DIR / "no_chat" / "pipeline_wrapper.py").read_text(),
}


@pytest.mark.parametrize(
    "pipeline_files",
    [("test_pipeline_1", SAMPLE_PIPELINE_FILES), ("test_pipeline_2", SAMPLE_PIPELINE_FILES_NO_CHAT_COMPLETION)],
)
def test_deploy_files_ok(status_pipeline, pipeline_files: tuple[str, dict], client, deploy_files):
    pipeline_data = {"name": pipeline_files[0], "files": pipeline_files[1]}

    response = deploy_files(client, pipeline_name=pipeline_data["name"], pipeline_files=pipeline_data["files"])
    assert response.status_code == 200
    assert (
        response.json()
        == DeployResponse(name=pipeline_files[0], success=True, endpoint=f"/{pipeline_files[0]}/run").model_dump()
    )

    status_response = status_pipeline(client, pipeline_files[0])
    assert status_response.status_code == 200

    status_body = PipelineStatusResponse(**status_response.json())
    assert status_body.pipeline == pipeline_files[0]

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200

    status_response = status_pipeline(client, pipeline_files[0])
    assert status_response.status_code == 200

    status_body = PipelineStatusResponse(**status_response.json())
    assert status_body.pipeline == pipeline_files[0]

    # Test if /{pipeline_name}/run endpoint accepts the expected parameters
    if pipeline_data["name"] == "test_pipeline_1":
        response = client.post(
            f"/{pipeline_data['name']}/run",
            json={"urls": ["https://www.redis.io"], "question": "What is Redis?"},
        )
        assert response.status_code == 200
        assert response.json() == {"result": "This is a mock response from the pipeline"}

    if pipeline_data["name"] == "test_pipeline_2":
        response = client.post(
            f"/{pipeline_data['name']}/run",
            json={"test_param": "test_value"},
        )
        assert response.status_code == 200
        assert response.json() == {"result": "Dummy result with test_value"}


def test_deploy_files_missing_wrapper(client, deploy_files) -> None:
    pipeline_data: dict[str, Any] = {"name": "test_pipeline", "files": SAMPLE_PIPELINE_FILES.copy()}
    pipeline_data["files"].pop("pipeline_wrapper.py")

    response = deploy_files(client, pipeline_name=pipeline_data["name"], pipeline_files=pipeline_data["files"])
    assert response.status_code == 422

    err_body: dict[str, Any] = response.json()
    assert "Missing required file" in err_body["detail"][0]["msg"]


def test_deploy_files_invalid_wrapper(client, deploy_files) -> None:
    invalid_files: dict[str, Any] = {
        "pipeline_wrapper.py": "invalid python code",
        "chat_with_website.yml": SAMPLE_PIPELINE_FILES["chat_with_website.yml"],
    }

    response = deploy_files(client, pipeline_name="test_pipeline", pipeline_files=invalid_files)
    assert response.status_code == 422

    err_body: dict[str, Any] = response.json()
    assert "Failed to load pipeline module" in err_body["detail"]


def test_deploy_files_duplicate_pipeline(client, deploy_files) -> None:
    response = deploy_files(client, pipeline_name="test_pipeline", pipeline_files=SAMPLE_PIPELINE_FILES)
    assert response.status_code == 200

    response = deploy_files(client, pipeline_name="test_pipeline", pipeline_files=SAMPLE_PIPELINE_FILES)
    assert response.status_code == 409

    err_body: dict[str, Any] = response.json()
    assert "Pipeline 'test_pipeline' already exists" in err_body["detail"]


def test_pipeline_endpoint_error_handling(client, deploy_files) -> None:
    pipeline_files: dict[str, Any] = {
        "pipeline_wrapper.py": (RUN_API_ERROR_DIR / "pipeline_wrapper.py").read_text(),
    }  # This pipeline wrapper will raise an error in run_api

    response = deploy_files(client, pipeline_name="errored_pipeline", pipeline_files=pipeline_files)
    assert response.status_code == 200

    run_response = client.post(
        "/errored_pipeline/run",
        json={"test_param": "test_value"},
    )
    assert run_response.status_code == 500

    err_body: dict[str, Any] = run_response.json()
    assert "Pipeline execution failed" in err_body["detail"]
    assert "This is a test error" in err_body["detail"]


def test_deploy_files_missing_required_methods(client, deploy_files) -> None:
    invalid_files: dict[str, Any] = {
        "pipeline_wrapper.py": (MISSING_METHODS_DIR / "pipeline_wrapper.py").read_text(),
        "chat_with_website.yml": SAMPLE_PIPELINE_FILES["chat_with_website.yml"],
    }

    response = deploy_files(client, pipeline_name="test_pipeline", pipeline_files=invalid_files)
    assert response.status_code == 422

    err_body: dict[str, Any] = response.json()
    assert (
        "At least one of run_api, run_api_async, run_chat_completion, or run_chat_completion_async must be implemented"
        in err_body["detail"]
    )


def test_deploy_files_setup_error(client, deploy_files) -> None:
    invalid_files: dict[str, Any] = {
        "pipeline_wrapper.py": (SETUP_ERROR_DIR / "pipeline_wrapper.py").read_text(),
        "chat_with_website.yml": SAMPLE_PIPELINE_FILES["chat_with_website.yml"],
    }

    response = deploy_files(client, pipeline_name="test_pipeline", pipeline_files=invalid_files)
    assert response.status_code == 422

    err_body: dict[str, Any] = response.json()
    assert "Failed to call setup() on pipeline wrapper instance: Setup failed!" in err_body["detail"]


def test_deploy_files_using_overwrite(status_pipeline, client, deploy_files, chat_completion) -> None:
    # Deploy the pipeline with the chat completion method
    response = deploy_files(
        client,
        pipeline_name="test_pipeline",
        pipeline_files=SAMPLE_PIPELINE_FILES,
        overwrite=True,
    )
    assert response.status_code == 200

    # Check status
    status = status_pipeline(client, "test_pipeline")
    assert status.status_code == 200

    status_body = PipelineStatusResponse(**status.json())
    assert status_body.pipeline == "test_pipeline"

    # Call chat endpoint (which is implemented)
    response = chat_completion(
        client,
        pipeline_name="test_pipeline",
        messages=[{"role": "user", "content": "Dummy question"}],
    )
    assert response.status_code == 200

    # Deploy the pipeline without the chat completion method
    response = deploy_files(
        client,
        pipeline_name="test_pipeline",
        pipeline_files=SAMPLE_PIPELINE_FILES_NO_CHAT_COMPLETION,
        overwrite=True,
    )
    assert response.status_code == 200

    status = status_pipeline(client, "test_pipeline")
    assert status.status_code == 200

    status_body = PipelineStatusResponse(**status.json())
    assert status_body.pipeline == "test_pipeline"

    # Call chat completion (which is NOT implemented after the overwrite)
    response = chat_completion(
        client,
        pipeline_name="test_pipeline",
        messages=[{"role": "user", "content": "Dummy question"}],
    )
    assert response.status_code == 501


def test_deploy_files_overwrite_without_saving(status_pipeline, client, deploy_files, test_settings) -> None:
    # First deploy with chat completion and save files
    response = deploy_files(
        client,
        pipeline_name="test_pipeline",
        pipeline_files=SAMPLE_PIPELINE_FILES,
        overwrite=True,
        save_files=False,
    )
    assert response.status_code == 200

    # Check that the pipeline files are not saved to disk
    pipeline_dir = Path(test_settings.pipelines_dir) / "test_pipeline"
    assert not pipeline_dir.exists()

    # Check that the pipeline is available
    status = status_pipeline(client, "test_pipeline")
    assert status.status_code == 200

    status_body = PipelineStatusResponse(**status.json())
    assert status_body.pipeline == "test_pipeline"

    # Deploy again without saving files
    response = deploy_files(
        client,
        pipeline_name="test_pipeline",
        pipeline_files=SAMPLE_PIPELINE_FILES_NO_CHAT_COMPLETION,
        overwrite=True,
        save_files=False,
    )
    assert response.status_code == 200

    # Check that the pipeline files are not saved to disk
    assert not pipeline_dir.exists()

    # Check that the pipeline is available
    status = status_pipeline(client, "test_pipeline")
    assert status.status_code == 200

    status_body = PipelineStatusResponse(**status.json())
    assert status_body.pipeline == "test_pipeline"


def test_deploy_files_overwrite_should_update_code(client, deploy_files) -> None:
    # First deploy with chat completion and save files
    response = deploy_files(
        client,
        pipeline_name="test_pipeline",
        pipeline_files=SAMPLE_PIPELINE_FILES,
        overwrite=True,
        save_files=False,
    )
    assert response.status_code == 200

    # Check route response
    response = client.post(
        "/test_pipeline/run",
        json={"urls": ["https://www.redis.io"], "question": "What is Redis?"},  # Dummy values
    )
    assert response.status_code == 200
    assert response.json() == {"result": "This is a mock response from the pipeline"}

    # Now let's update the code
    UPDATED_PIPELINE_FILES = SAMPLE_PIPELINE_FILES.copy()
    UPDATED_PIPELINE_FILES["pipeline_wrapper.py"] = UPDATED_PIPELINE_FILES["pipeline_wrapper.py"].replace(
        "This is a mock response from the pipeline", "This is an updated mock response from the pipeline"
    )

    response = deploy_files(
        client,
        pipeline_name="test_pipeline",
        pipeline_files=UPDATED_PIPELINE_FILES,
        overwrite=True,
        save_files=False,
    )
    assert response.status_code == 200

    # Check route response after update
    response = client.post(
        "/test_pipeline/run",
        json={"urls": ["https://www.redis.io"], "question": "What is Redis?"},  # Dummy values
    )
    assert response.status_code == 200
    assert response.json() == {"result": "This is an updated mock response from the pipeline"}


def test_deploy_files_with_file_upload(client, deploy_files, tmp_path) -> None:
    # Deploy the pipeline
    response = deploy_files(
        client,
        pipeline_name="file_upload_pipeline",
        pipeline_files={
            "pipeline_wrapper.py": (UPLOAD_FILES_DIR / "pipeline_wrapper.py").read_text(),
        },
    )
    assert response.status_code == 200

    # Call the endpoint with a param but WITHOUT a file
    response = client.post(
        "/file_upload_pipeline/run",
        data={"test_param": "test_value"},
    )
    assert response.status_code == 200
    assert response.json() == {"result": "No files received, param: test_value"}

    # Create a test file
    test_file_path = tmp_path / "test_file.txt"
    test_file_path.write_text("This is a test file")

    # Call the endpoint with a param AND a file
    # NOTE: This is a multipart/form-data request
    with open(test_file_path, "rb") as f:
        response = client.post(
            "/file_upload_pipeline/run",
            data={"test_param": "test_value"},
            files={"files": ("test_file.txt", f, "text/plain")},
        )

    assert response.status_code == 200

    resp_body = response.json()
    assert "Received files: test_file.txt" in resp_body.get("result", "")
    assert "with param test_value" in resp_body.get("result", "")


def test_deploy_async_pipeline_wrapper(status_pipeline, client, deploy_files) -> None:
    response = deploy_files(
        client,
        pipeline_name="async_pipeline",
        pipeline_files=ASYNC_SAMPLE_PIPELINE_FILES,
    )
    assert response.status_code == 200

    status_response = status_pipeline(client, "async_pipeline")
    assert status_response.status_code == 200

    status_body = PipelineStatusResponse(**status_response.json())
    assert status_body.pipeline == "async_pipeline"

    response = client.post(
        "/async_pipeline/run",
        json={"urls": ["https://www.redis.io"], "question": "What is Redis?"},
    )
    assert response.status_code == 200
    assert response.json() == {"result": "This is a mock response from the pipeline"}
