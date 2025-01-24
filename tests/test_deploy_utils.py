import pytest
import shutil
from pathlib import Path
from typing import Callable
from hayhooks.server.utils.deploy_utils import (
    load_pipeline_module,
    save_pipeline_files,
    create_request_model_from_callable,
    create_response_model_from_callable,
)
from hayhooks.server.exceptions import (
    PipelineFilesError,
    PipelineModuleLoadError,
)

TEST_PIPELINES_DIR = Path("tests/test_files/test_pipelines")


@pytest.fixture(autouse=True)
def cleanup_test_pipelines():
    yield

    if TEST_PIPELINES_DIR.exists():
        shutil.rmtree(TEST_PIPELINES_DIR)


def test_load_pipeline_module():
    pipeline_name = "chat_with_website"
    pipeline_folder_path = Path("tests/test_files/files/chat_with_website")

    module = load_pipeline_module(pipeline_name, pipeline_folder_path)

    assert module is not None
    assert hasattr(module, "PipelineWrapper")
    assert isinstance(getattr(module.PipelineWrapper, "run_api"), Callable)
    assert isinstance(getattr(module.PipelineWrapper, "run_chat"), Callable)
    assert isinstance(getattr(module.PipelineWrapper, "setup"), Callable)


def test_load_pipeline_wrong_folder():
    pipeline_name = "chat_with_website"
    pipeline_folder_path = Path("tests/test_files/files/wrong_folder")

    with pytest.raises(
        PipelineModuleLoadError,
        match="Required file 'tests/test_files/files/wrong_folder/pipeline_wrapper.py' not found",
    ):
        load_pipeline_module(pipeline_name, pipeline_folder_path)


def test_load_pipeline_no_wrapper():
    pipeline_name = "chat_with_website"
    pipeline_folder_path = Path("tests/test_files/files/no_wrapper")

    with pytest.raises(
        PipelineModuleLoadError,
        match="Required file 'tests/test_files/files/no_wrapper/pipeline_wrapper.py' not found",
    ):
        load_pipeline_module(pipeline_name, pipeline_folder_path)


def test_save_pipeline_files_basic():
    files = {
        "pipeline_wrapper.py": "print('hello')",
        "extra_file.txt": "extra content",
    }

    saved_paths = save_pipeline_files("test_pipeline", files, pipelines_dir=TEST_PIPELINES_DIR)

    assert len(saved_paths) == 2
    for filename, path in saved_paths.items():
        assert Path(path).exists()
        assert Path(path).read_text() == files[filename]


def test_save_pipeline_files_empty():
    pipeline_name = "test_pipeline"
    files = {}

    saved_paths = save_pipeline_files(pipeline_name, files, pipelines_dir=TEST_PIPELINES_DIR)

    assert len(saved_paths) == 0
    assert (TEST_PIPELINES_DIR / pipeline_name).exists()
    assert (TEST_PIPELINES_DIR / pipeline_name).is_dir()
    assert len([file for file in (TEST_PIPELINES_DIR / pipeline_name).iterdir()]) == 0


def test_save_pipeline_files_raises_error(tmp_path):
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o444)

    files = {"test.py": "print('hello')"}

    with pytest.raises(PipelineFilesError) as exc_info:
        save_pipeline_files(pipeline_name="test_pipeline", files=files, pipelines_dir=str(readonly_dir))

    assert "Failed to save pipeline files" in str(exc_info.value)


def test_create_request_model_from_callable():
    def sample_func(name: str, age: int = 25, optional: str = ""):
        pass

    model = create_request_model_from_callable(sample_func, "Test")

    assert model.__name__ == "TestRequest"
    assert model.model_fields["name"].annotation == str
    assert model.model_fields["name"].is_required
    assert model.model_fields["age"].annotation == int
    assert model.model_fields["age"].default == 25
    assert model.model_fields["optional"].annotation == str
    assert model.model_fields["optional"].default == ""


def test_create_response_model_from_callable():
    def sample_func() -> dict:
        return {"result": "test"}

    model = create_response_model_from_callable(sample_func, "Test")

    assert model.__name__ == "TestResponse"
    assert model.model_fields["result"].annotation == dict
    assert model.model_fields["result"].is_required
