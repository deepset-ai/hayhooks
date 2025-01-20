import pytest
import shutil
from pathlib import Path
from typing import Callable
from hayhooks.server.utils.deploy_utils import load_pipeline_module, save_pipeline_files
from hayhooks.server.exceptions import PipelineFilesError
from hayhooks.settings import settings

TEST_PIPELINES_DIR = Path("tests/test_files/test_pipelines")


@pytest.fixture(autouse=True)
def cleanup_test_pipelines():
    yield

    if TEST_PIPELINES_DIR.exists():
        shutil.rmtree(TEST_PIPELINES_DIR)


def test_load_pipeline_module():
    pipeline_name = "chat_with_website"
    pipeline_folder_path = Path("tests/test_files/python/chat_with_website")

    module = load_pipeline_module(pipeline_name, pipeline_folder_path)

    assert module is not None
    assert hasattr(module, "PipelineWrapper")
    assert isinstance(getattr(module.PipelineWrapper, "run_api"), Callable)
    assert isinstance(getattr(module.PipelineWrapper, "run_chat"), Callable)
    assert isinstance(getattr(module.PipelineWrapper, "setup"), Callable)


def test_load_pipeline_wrong_folder():
    pipeline_name = "chat_with_website"
    pipeline_folder_path = Path("tests/test_files/python/wrong_folder")

    with pytest.raises(
        ValueError,
        match="Required file 'tests/test_files/python/wrong_folder/pipeline_wrapper.py' not found",
    ):
        load_pipeline_module(pipeline_name, pipeline_folder_path)


def test_load_pipeline_no_wrapper():
    pipeline_name = "chat_with_website"
    pipeline_folder_path = Path("tests/test_files/python/no_wrapper")

    with pytest.raises(
        ValueError,
        match="Required file 'tests/test_files/python/no_wrapper/pipeline_wrapper.py' not found",
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
