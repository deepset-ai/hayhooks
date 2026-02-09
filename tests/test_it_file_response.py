import shutil
from pathlib import Path

import pytest

from hayhooks.server.pipelines.registry import registry


def clear_registry_and_files(pipelines_dir: str):
    registry.clear()
    if Path(pipelines_dir).exists():
        shutil.rmtree(pipelines_dir)


@pytest.fixture(autouse=True)
def cleanup(test_settings):
    clear_registry_and_files(test_settings.pipelines_dir)
    yield


TEST_FILES_DIR = Path(__file__).parent / "test_files/files"
FILE_RESPONSE_DIR = TEST_FILES_DIR / "file_response"

FILE_RESPONSE_PIPELINE_FILES = {
    "pipeline_wrapper.py": (FILE_RESPONSE_DIR / "pipeline_wrapper.py").read_text(),
}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.mark.integration
def test_run_api_returns_image(client, deploy_files):
    """Test that run_api can return a file response when the return type is annotated as FileResponse."""
    response = deploy_files(
        client,
        pipeline_name="image_pipeline",
        pipeline_files=FILE_RESPONSE_PIPELINE_FILES,
    )
    assert response.status_code == 200

    # Call the run endpoint
    response = client.post(
        "/image_pipeline/run",
        json={"width": 16, "height": 16},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert 'filename="random_image.png"' in response.headers["content-disposition"]

    # Verify the response body is a valid PNG
    assert response.content[:8] == PNG_SIGNATURE
    assert len(response.content) > 0


@pytest.mark.integration
def test_run_api_returns_image_with_defaults(client, deploy_files):
    """Test that run_api works with default width/height parameters."""
    response = deploy_files(
        client,
        pipeline_name="image_pipeline_defaults",
        pipeline_files=FILE_RESPONSE_PIPELINE_FILES,
    )
    assert response.status_code == 200

    # Call without specifying width/height to use defaults (64x64)
    response = client.post(
        "/image_pipeline_defaults/run",
        json={},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == PNG_SIGNATURE
