from pathlib import Path

from hayhooks.server.routers.deploy import DeployResponse

SAMPLE_CALC_PIPELINE_PATH = Path(__file__).parent / "test_files" / "yaml" / "sample_calc_pipeline.yml"


def test_deploy_yaml_route_and_run_ok(client):
    yaml_source = SAMPLE_CALC_PIPELINE_PATH.read_text().strip()

    # Deploy via the new route
    response = client.post("/deploy-yaml", json={"name": "calc", "source_code": yaml_source, "overwrite": True})
    assert response.status_code == 200
    assert response.json() == DeployResponse(name="calc", success=True, endpoint="/calc/run").model_dump()

    # /status should include the pipeline
    status_response = client.get("/status/calc")
    assert status_response.status_code == 200
    assert status_response.json()["pipeline"] == "calc"

    # OpenAPI docs should render
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200

    # Run the flat endpoint using declared inputs
    run_response = client.post("/calc/run", json={"value": 3})
    assert run_response.status_code == 200

    # (3 + 2) * 2 = 10
    assert run_response.json() == {"result": 10}


def test_deploy_yaml_saves_file(client, test_settings):
    yaml_source = SAMPLE_CALC_PIPELINE_PATH.read_text().strip()
    response = client.post(
        "/deploy-yaml",
        json={
            "name": "save_me",
            "source_code": yaml_source,
            "overwrite": True,
            "save_file": True,
        },
    )
    assert response.status_code == 200

    file_path = Path(test_settings.pipelines_dir) / "save_me.yml"
    assert file_path.exists()
    assert file_path.read_text() == yaml_source
