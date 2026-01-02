from pathlib import Path

from hayhooks.server.pipelines.registry import registry

TEST_FILES_DIR = Path(__file__).parent / "test_files/files"
RUN_API_ERROR_DIR = TEST_FILES_DIR / "run_api_error"


def test_pipeline_execution_error_logs_traceback(client, deploy_files, caplog):
    registry.clear()

    pipeline_files = {
        "pipeline_wrapper.py": (RUN_API_ERROR_DIR / "pipeline_wrapper.py").read_text(),
    }

    response = deploy_files(client, pipeline_name="error_pipeline", pipeline_files=pipeline_files)
    assert response.status_code == 200

    response = client.post("/error_pipeline/run", json={"test_param": "test_value"})
    assert response.status_code == 500
    assert "Pipeline execution error: " in caplog.text
    assert "Traceback" in caplog.text
    assert "ValueError: This is a test error" in caplog.text
