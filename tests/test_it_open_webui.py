import pytest
from hayhooks.server.schema import ModelObject, ModelsResponse
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
    assert response.json() == {"data": [], "object": "list"}


def test_get_models(deploy_pipeline):
    pipeline_file = Path(__file__).parent / "test_files" / "working_pipelines/test_pipeline_01.yml"
    deploy_pipeline(client, pipeline_file.stem, pipeline_file.read_text())

    response = client.get("/models")
    response_data = response.json()

    expected_response = ModelsResponse(
        object="list",
        data=[
            ModelObject(
                id="test_pipeline_01",
                name="test_pipeline_01",
                object="model",
                created=response_data["data"][0]["created"],
                owned_by="hayhooks",
            )
        ],
    )

    assert response.status_code == 200
    assert response_data == expected_response.model_dump()
