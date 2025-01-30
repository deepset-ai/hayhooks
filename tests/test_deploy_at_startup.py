import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from hayhooks.server.app import create_app
from hayhooks.settings import settings
from hayhooks.server.pipelines.registry import registry


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()
    yield


@pytest.fixture
def test_files_pipelines_dir():
    return Path("tests/test_files/files")


@pytest.fixture
def test_yaml_pipelines_dir():
    return Path("tests/test_files/yaml/working_pipelines")


@pytest.fixture
def test_mixed_pipelines_dir():
    return Path("tests/test_files/mixed")


@pytest.fixture
def app_with_files_pipelines(test_files_pipelines_dir, monkeypatch):
    monkeypatch.setattr(settings, "pipelines_dir", str(test_files_pipelines_dir))
    app = create_app()
    return app


@pytest.fixture
def app_with_yaml_pipelines(test_yaml_pipelines_dir, monkeypatch):
    monkeypatch.setattr(settings, "pipelines_dir", str(test_yaml_pipelines_dir))
    app = create_app()
    return app


@pytest.fixture
def app_with_mixed_pipelines(test_mixed_pipelines_dir, monkeypatch):
    monkeypatch.setattr(settings, "pipelines_dir", str(test_mixed_pipelines_dir))
    app = create_app()
    return app


@pytest.fixture
def test_client_files(app_with_files_pipelines):
    return TestClient(app_with_files_pipelines)


@pytest.fixture
def test_client_yaml(app_with_yaml_pipelines):
    return TestClient(app_with_yaml_pipelines)


@pytest.fixture
def test_client_mixed(app_with_mixed_pipelines):
    return TestClient(app_with_mixed_pipelines)


def test_app_loads_pipeline_from_files_directory(test_client_files, test_files_pipelines_dir):
    response = test_client_files.get("/status")
    assert response.status_code == 200

    pipelines = response.json()["pipelines"]
    assert len(pipelines) >= 1  # at least one pipeline should be loaded
    assert "chat_with_website" in pipelines


def test_app_loads_pipeline_from_yaml_directory(test_client_yaml, test_yaml_pipelines_dir):
    response = test_client_yaml.get("/status")
    assert response.status_code == 200

    pipelines = response.json()["pipelines"]
    assert len(pipelines) == len(list(test_yaml_pipelines_dir.rglob("*")))


def test_app_loads_pipeline_from_mixed_directory(test_client_mixed, test_mixed_pipelines_dir):
    response = test_client_mixed.get("/status")
    assert response.status_code == 200

    pipelines = response.json()["pipelines"]
    assert len(pipelines) == 2  # 1 file, 1 yaml
