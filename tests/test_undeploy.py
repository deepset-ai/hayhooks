from pathlib import Path
from fastapi.testclient import TestClient
from hayhooks.server.pipelines.registry import registry


TEST_FILES_DIR = Path(__file__).parent / "test_files/files"

SAMPLE_PIPELINE_FILES = {
    "pipeline_wrapper.py": (TEST_FILES_DIR / "chat_with_website" / "pipeline_wrapper.py").read_text(),
    "chat_with_website.yml": (TEST_FILES_DIR / "chat_with_website" / "chat_with_website.yml").read_text(),
}


def test_undeploy_standard_pipeline(client: TestClient, deploy_pipeline, undeploy_pipeline):
    deploy_response = deploy_pipeline(
        client,
        pipeline_name="test_undeploy_pipeline",
        pipeline_source_code=SAMPLE_PIPELINE_FILES["chat_with_website.yml"],
    )
    assert deploy_response.status_code == 200
    assert deploy_response.json()["name"] == "test_undeploy_pipeline"

    # Verify pipeline exists in registry
    assert "test_undeploy_pipeline" in registry.get_names()

    # Undeploy the pipeline
    undeploy_response = undeploy_pipeline(client, pipeline_name="test_undeploy_pipeline")
    assert undeploy_response.status_code == 200
    assert undeploy_response.json()["success"] is True
    assert undeploy_response.json()["name"] == "test_undeploy_pipeline"

    # Verify pipeline no longer exists in registry
    assert "test_undeploy_pipeline" not in registry.get_names()

    # Verify pipeline endpoint no longer exists
    response = client.post(f"/test_undeploy_pipeline", json={})
    print(response.json())
    assert response.status_code == 404


def test_undeploy_wrapper_pipeline(client: TestClient, deploy_files, undeploy_pipeline):
    deploy_response = deploy_files(client, pipeline_name="test_undeploy_wrapper", pipeline_files=SAMPLE_PIPELINE_FILES)
    assert deploy_response.status_code == 200
    assert deploy_response.json()["name"] == "test_undeploy_wrapper"

    # Verify pipeline exists in registry
    assert "test_undeploy_wrapper" in registry.get_names()

    # Verify run endpoint exists
    run_response = client.post(
        f"/test_undeploy_wrapper/run", json={"question": "test", "urls": ["https://www.google.com"]}
    )
    assert run_response.status_code == 200

    # Undeploy the pipeline
    undeploy_response = undeploy_pipeline(client, pipeline_name="test_undeploy_wrapper")
    assert undeploy_response.status_code == 200
    assert undeploy_response.json()["success"] is True
    assert undeploy_response.json()["name"] == "test_undeploy_wrapper"

    # Verify pipeline no longer exists in registry
    assert "test_undeploy_wrapper" not in registry.get_names()

    # Verify run endpoint no longer exists
    run_response = client.post(
        f"/test_undeploy_wrapper/run", json={"question": "test", "urls": ["https://www.google.com"]}
    )
    assert run_response.status_code == 404


def test_undeploy_nonexistent_pipeline(client: TestClient, undeploy_pipeline):
    undeploy_response = undeploy_pipeline(client, pipeline_name="nonexistent_pipeline")
    assert undeploy_response.status_code == 404
    assert "not found" in undeploy_response.json().get("detail", "").lower()

    # Attempt to undeploy the nonexistent pipeline
    undeploy_response = undeploy_pipeline(client, pipeline_name="nonexistent_pipeline")
    assert undeploy_response.status_code == 404
    assert "not found" in undeploy_response.json().get("detail", "").lower()


def test_undeploy_removes_files(client: TestClient, deploy_files, undeploy_pipeline, test_settings):
    deploy_response = deploy_files(
        client, pipeline_name="test_undeploy_files", pipeline_files=SAMPLE_PIPELINE_FILES, save_files=True
    )
    assert deploy_response.status_code == 200

    # Verify files exist on disk
    pipeline_dir = Path(test_settings.pipelines_dir) / "test_undeploy_files"
    assert pipeline_dir.exists()
    assert (pipeline_dir / "pipeline_wrapper.py").exists()
    assert (pipeline_dir / "chat_with_website.yml").exists()

    # Undeploy the pipeline
    undeploy_response = undeploy_pipeline(client, pipeline_name="test_undeploy_files")
    assert undeploy_response.status_code == 200

    # Verify files no longer exist on disk
    assert not pipeline_dir.exists()
