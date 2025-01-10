from fastapi.testclient import TestClient
from hayhooks.server import app
from tests.test_it_deploy import deploy_pipeline
from pathlib import Path

client = TestClient(app)


def test_draw_pipeline():
    pipeline_file = Path(__file__).parent / "test_files" / "working_pipelines/test_pipeline_01.yml"
    pipeline_data = {"name": pipeline_file.stem, "source_code": pipeline_file.read_text()}

    deploy_response = deploy_pipeline(pipeline_data)
    assert deploy_response.status_code == 200

    draw_response = client.get(f"/draw/{pipeline_data['name']}")
    assert draw_response.status_code == 200

    assert draw_response.headers["Content-Type"] == "image/png"
    assert len(draw_response.content) > 0


def test_draw_non_existent_pipeline():
    draw_response = client.get("/draw/non_existent_pipeline")
    assert draw_response.status_code == 404
