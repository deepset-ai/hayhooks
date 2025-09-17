from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from packaging.version import InvalidVersion, Version

from hayhooks.server.app import create_app
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
    return Path("tests/test_files/yaml")


@pytest.fixture
def test_mixed_pipelines_dir():
    return Path("tests/test_files/mixed")


@pytest.fixture
def app_with_files_pipelines(test_settings, test_files_pipelines_dir, monkeypatch):
    monkeypatch.setattr(test_settings, "pipelines_dir", str(test_files_pipelines_dir))
    app = create_app()
    return app


@pytest.fixture
def app_with_yaml_pipelines(test_settings, test_yaml_pipelines_dir, monkeypatch):
    monkeypatch.setattr(test_settings, "pipelines_dir", str(test_yaml_pipelines_dir))
    app = create_app()
    return app


@pytest.fixture
def app_with_mixed_pipelines(test_settings, test_mixed_pipelines_dir, monkeypatch):
    monkeypatch.setattr(test_settings, "pipelines_dir", str(test_mixed_pipelines_dir))
    app = create_app()
    return app


# When testing the Hayhooks server, it's important to ensure that the FastAPI "lifespan" events
# (startup and shutdown) are properly executed. The pipeline deployment logic is now located in
# the startup event via the app's lifespan function.
#
# Using the TestClient as a context manager (i.e., with the "with" statement) ensures that the
# startup event is triggered, which in turn deploys all the pipelines from the specified directory.
# Without the context manager, the lifespan events would not run, and the pipelines would not be deployed.
@pytest.fixture
def test_client_files(app_with_files_pipelines):
    with TestClient(app_with_files_pipelines) as client:
        yield client


@pytest.fixture
def test_client_yaml(app_with_yaml_pipelines):
    with TestClient(app_with_yaml_pipelines) as client:
        yield client


@pytest.fixture
def test_client_mixed(app_with_mixed_pipelines):
    with TestClient(app_with_mixed_pipelines) as client:
        yield client


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
    assert len(pipelines) == 2


def test_app_loads_pipeline_from_mixed_directory(test_client_mixed, test_mixed_pipelines_dir):
    response = test_client_mixed.get("/status")
    assert response.status_code == 200

    pipelines = response.json()["pipelines"]
    assert len(pipelines) == 2  # 1 file, 1 yaml


def test_app_version_is_pep440_compliant():
    app = create_app()
    assert hasattr(app, "version"), "App should have a 'version' attribute"
    assert app.version not in (None, "", "unknown"), "App version should be defined and not 'unknown'"
    try:
        Version(app.version)
    except InvalidVersion as e:
        msg = f"App version '{app.version}' is not a valid PEP 440 version"
        raise AssertionError(msg) from e
