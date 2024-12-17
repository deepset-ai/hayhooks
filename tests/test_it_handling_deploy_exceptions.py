from fastapi.testclient import TestClient
from hayhooks.server import app
from pathlib import Path

client = TestClient(app)


def test_gracefully_handle_deploy_exception():
    pipeline_name = "broken_rag_pipeline"
    pipeline_def = (Path(__file__).parent / "test_files" / "broken_rag_pipeline.yml").read_text()

    deploy_response = client.post("/deploy", json={"name": pipeline_name, "source_code": pipeline_def})
    assert deploy_response.status_code == 500
    assert "Couldn't deserialize component 'llm'" in deploy_response.json()["detail"]
