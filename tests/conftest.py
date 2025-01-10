import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def deploy_pipeline():
    def _deploy_pipeline(client: TestClient, pipeline_name: str, pipeline_source_code: str):
        deploy_response = client.post("/deploy", json={"name": pipeline_name, "source_code": pipeline_source_code})
        return deploy_response
    return _deploy_pipeline


@pytest.fixture
def undeploy_pipeline():
    def _undeploy_pipeline(client: TestClient, pipeline_name: str):
        undeploy_response = client.post(f"/undeploy/{pipeline_name}")
        return undeploy_response
    return _undeploy_pipeline


@pytest.fixture
def draw_pipeline():
    def _draw_pipeline(client: TestClient, pipeline_name: str):
        draw_response = client.get(f"/draw/{pipeline_name}")
        return draw_response
    return _draw_pipeline


@pytest.fixture
def status_pipeline():
    def _status_pipeline(client: TestClient, pipeline_name: str):
        status_response = client.get(f"/status/{pipeline_name}")
        return status_response
    return _status_pipeline
