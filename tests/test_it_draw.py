from fastapi.testclient import TestClient
from hayhooks.server import app
from pathlib import Path

client = TestClient(app)


def test_draw_pipeline(deploy_pipeline, draw_pipeline):
    pipeline_file = Path(__file__).parent / "test_files/yaml" / "working_pipelines/test_pipeline_01.yml"
    pipeline_data = {"name": pipeline_file.stem, "source_code": pipeline_file.read_text()}

    deploy_pipeline(client, pipeline_data["name"], pipeline_data["source_code"])

    draw_response = draw_pipeline(client, pipeline_data["name"])
    assert draw_response.status_code == 200

    assert draw_response.headers["Content-Type"] == "image/png"
    assert len(draw_response.content) > 0


def test_draw_non_existent_pipeline(draw_pipeline):
    draw_response = draw_pipeline(client, "non_existent_pipeline")
    assert draw_response.status_code == 404
