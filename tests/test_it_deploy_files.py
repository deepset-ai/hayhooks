import pytest
from fastapi.testclient import TestClient
from hayhooks.server import app
from pathlib import Path
from hayhooks.server.pipelines.registry import registry
from hayhooks.settings import settings
import shutil

client = TestClient(app)


def cleanup():
    registry.clear()
    if Path(settings.pipelines_dir).exists():
        shutil.rmtree(settings.pipelines_dir)


@pytest.fixture(autouse=True)
def clear_registry():
    cleanup()
    yield


@pytest.fixture(scope="session", autouse=True)
def final_cleanup():
    yield
    cleanup()


# Load test pipeline files from test directory
TEST_FILES_DIR = Path(__file__).parent / "test_files/python/chat_with_website"
PIPELINE_FILES = {
    "pipeline_wrapper.py": (TEST_FILES_DIR / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (TEST_FILES_DIR / "chat_with_website.yml").read_text(),
}


def test_deploy_files_ok(status_pipeline):
    pipeline_data = {"name": "test_pipeline", "files": PIPELINE_FILES}

    response = client.post("/deploy_files", json=pipeline_data)
    assert response.status_code == 200
    assert response.json() == {"name": "test_pipeline"}

    status_response = status_pipeline(client, pipeline_data["name"])
    assert pipeline_data["name"] in status_response.json()["pipeline"]

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200

    status_response = status_pipeline(client, pipeline_data["name"])
    assert pipeline_data["name"] in status_response.json()["pipeline"]


def test_deploy_files_missing_wrapper():
    pipeline_data = {"name": "test_pipeline", "files": PIPELINE_FILES.copy()}
    pipeline_data["files"].pop("pipeline_wrapper.py")

    response = client.post("/deploy_files", json=pipeline_data)
    assert response.status_code == 422
    assert "Required file" in response.json()["detail"]


def test_deploy_files_invalid_wrapper():
    invalid_files = {
        "pipeline_wrapper.py": "invalid python code",
        "chat_with_website.yml": PIPELINE_FILES["chat_with_website.yml"],
    }

    response = client.post("/deploy_files", json={"name": "test_pipeline", "files": invalid_files})
    assert response.status_code == 422
    assert "Failed to load pipeline module" in response.json()["detail"]


def test_deploy_files_duplicate_pipeline():
    response = client.post("/deploy_files", json={"name": "test_pipeline", "files": PIPELINE_FILES})
    assert response.status_code == 200

    response = client.post("/deploy_files", json={"name": "test_pipeline", "files": PIPELINE_FILES})
    print(response.json())
    assert response.status_code == 409
    assert "Pipeline 'test_pipeline' already exists" in response.json()["detail"]
