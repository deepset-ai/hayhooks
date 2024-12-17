import pytest
from fastapi.testclient import TestClient
from hayhooks.server import app
from pathlib import Path

client = TestClient(app)

# Load pipeline definitions from test_files
test_files = Path(__file__).parent / "test_files" / "working_pipelines"
pipeline_data = [{"name": file.stem, "source_code": file.read_text()} for file in test_files.glob("*.yml")]


@pytest.mark.parametrize("pipeline_data", pipeline_data)
def test_deploy_pipeline_def(pipeline_data: dict):
    deploy_response = client.post(
        "/deploy", json={"name": pipeline_data["name"], "source_code": pipeline_data["source_code"]}
    )
    assert deploy_response.status_code == 200

    status_response = client.get("/status")
    assert pipeline_data["name"] in status_response.json()["pipelines"]

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
