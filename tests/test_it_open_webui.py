import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from hayhooks.server import app
from hayhooks.server.pipelines import registry

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()


def test_get_models_empty():
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_get_models(deploy_pipeline):
    pipeline_file = Path(__file__).parent / "test_files" / "working_pipelines/test_pipeline_01.yml"
    deploy_pipeline(client, pipeline_file.stem, pipeline_file.read_text())

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json() == {"models": ["test_pipeline_01"]}
