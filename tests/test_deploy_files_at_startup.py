import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from hayhooks.server.app import create_app
from hayhooks.settings import settings


@pytest.fixture
def test_pipelines_dir():
    return Path("tests/test_files/python")


@pytest.fixture
def app_with_pipelines(test_pipelines_dir, monkeypatch):
    monkeypatch.setattr(settings, "pipelines_dir", str(test_pipelines_dir))
    app = create_app()
    return app


@pytest.fixture
def test_client(app_with_pipelines):
    return TestClient(app_with_pipelines)


def test_app_loads_pipeline_from_directory(test_client, test_pipelines_dir):
    response = test_client.get("/status")
    assert response.status_code == 200

    pipelines = response.json()["pipelines"]
    assert len(pipelines) == 1  # only one pipeline should be loaded
    assert "chat_with_website" in pipelines
