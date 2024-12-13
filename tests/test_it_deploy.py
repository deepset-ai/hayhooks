import pytest
from fastapi.testclient import TestClient
from hayhooks.server import app
from pathlib import Path

client = TestClient(app)

# Load pipeline definitions from test_files
test_files = Path(__file__).parent / "test_files"
pipeline_names = [file.stem for file in test_files.glob("*.yml")]


@pytest.mark.parametrize("pipeline_name", pipeline_names)
def test_deploy_pipeline_def(pipeline_name: str):
    pipeline_def = (Path(__file__).parent / "test_files" / f"{pipeline_name}.yml").read_text()

    deploy_response = client.post("/deploy", json={"name": pipeline_name, "source_code": pipeline_def})
    assert deploy_response.status_code == 200

    status_response = client.get("/status")
    assert pipeline_name in status_response.json()["pipelines"]

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
