from fastapi.routing import APIRoute
import pytest
import shutil
from haystack import Pipeline
from pathlib import Path
from typing import Callable
from hayhooks.server.utils.deploy_utils import (
    load_pipeline_module,
    save_pipeline_files,
    create_request_model_from_callable,
    create_response_model_from_callable,
    create_pipeline_wrapper_instance,
    deploy_pipeline_files,
)
from hayhooks.server.exceptions import (
    PipelineFilesError,
    PipelineModuleLoadError,
    PipelineWrapperError,
)
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


@pytest.fixture(autouse=True)
def cleanup_test_pipelines(test_settings):
    yield

    if Path(test_settings.pipelines_dir).exists():
        shutil.rmtree(test_settings.pipelines_dir)


def test_load_pipeline_module():
    pipeline_name = "chat_with_website"
    pipeline_dir_path = Path("tests/test_files/files/chat_with_website")

    module = load_pipeline_module(pipeline_name, pipeline_dir_path)

    assert module is not None
    assert hasattr(module, "PipelineWrapper")
    assert isinstance(getattr(module.PipelineWrapper, "run_api"), Callable)
    assert isinstance(getattr(module.PipelineWrapper, "run_chat_completion"), Callable)
    assert isinstance(getattr(module.PipelineWrapper, "setup"), Callable)


def test_load_pipeline_wrong_dir():
    pipeline_name = "chat_with_website"
    pipeline_dir_path = Path("tests/test_files/files/wrong_dir")

    with pytest.raises(
        PipelineModuleLoadError,
        match="Required file 'tests/test_files/files/wrong_dir/pipeline_wrapper.py' not found",
    ):
        load_pipeline_module(pipeline_name, pipeline_dir_path)


def test_load_pipeline_no_wrapper():
    pipeline_name = "chat_with_website"
    pipeline_dir_path = Path("tests/test_files/files/no_wrapper")

    with pytest.raises(
        PipelineModuleLoadError,
        match="Required file 'tests/test_files/files/no_wrapper/pipeline_wrapper.py' not found",
    ):
        load_pipeline_module(pipeline_name, pipeline_dir_path)


def test_save_pipeline_files_basic(test_settings):
    files = {
        "pipeline_wrapper.py": "print('hello')",
        "extra_file.txt": "extra content",
    }

    saved_paths = save_pipeline_files("test_pipeline", files, pipelines_dir=test_settings.pipelines_dir)

    assert len(saved_paths) == 2
    for filename, path in saved_paths.items():
        assert Path(path).exists()
        assert Path(path).read_text() == files[filename]


def test_save_pipeline_files_empty(test_settings):
    pipeline_name = "test_pipeline"
    files = {}

    saved_paths = save_pipeline_files(pipeline_name, files, pipelines_dir=test_settings.pipelines_dir)

    assert len(saved_paths) == 0
    assert (Path(test_settings.pipelines_dir) / pipeline_name).exists()
    assert (Path(test_settings.pipelines_dir) / pipeline_name).is_dir()
    assert len([file for file in (Path(test_settings.pipelines_dir) / pipeline_name).iterdir()]) == 0


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


def test_create_pipeline_wrapper_instance_success():
    class ValidPipelineWrapper(BasePipelineWrapper):
        def setup(self):
            self.pipeline = Pipeline()

        def run_api(self):
            pass

        def run_chat_completion(self, model, messages, body):
            pass

    module = type('Module', (), {'PipelineWrapper': ValidPipelineWrapper})

    wrapper = create_pipeline_wrapper_instance(module)
    assert isinstance(wrapper, BasePipelineWrapper)
    assert hasattr(wrapper, 'run_api')
    assert hasattr(wrapper, 'run_chat_completion')
    assert isinstance(wrapper.pipeline, Pipeline)


def test_create_pipeline_wrapper_instance_init_error():
    class BrokenPipelineWrapper:
        def __init__(self):
            raise ValueError("Init error")

    module = type('Module', (), {'PipelineWrapper': BrokenPipelineWrapper})

    with pytest.raises(PipelineWrapperError, match="Failed to create pipeline wrapper instance: Init error"):
        create_pipeline_wrapper_instance(module)


def test_create_pipeline_wrapper_instance_setup_error():
    class BrokenSetupWrapper(BasePipelineWrapper):
        def setup(self):
            raise ValueError("Setup error")

        def run_api(self):
            pass

    module = type('Module', (), {'PipelineWrapper': BrokenSetupWrapper})

    with pytest.raises(
        PipelineWrapperError, match="Failed to call setup\\(\\) on pipeline wrapper instance: Setup error"
    ):
        create_pipeline_wrapper_instance(module)


def test_create_pipeline_wrapper_instance_missing_methods():
    class IncompleteWrapper(BasePipelineWrapper):
        def setup(self):
            self.pipeline = Pipeline()

    module = type('Module', (), {'PipelineWrapper': IncompleteWrapper})

    with pytest.raises(
        PipelineWrapperError, match="At least one of run_api or run_chat_completion must be implemented"
    ):
        create_pipeline_wrapper_instance(module)


def test_deploy_pipeline_files_without_saving(test_settings, mocker):
    mock_app = mocker.Mock()

    # We're saving the pipeline wrapper file in the test_files directory
    test_file_path = Path("tests/test_files/files/no_chat/pipeline_wrapper.py")
    files = {"pipeline_wrapper.py": test_file_path.read_text()}

    # Mock the app routes to mimic the existing route
    mock_app.routes = [APIRoute(path="/test_pipeline/run", endpoint=lambda: None, methods=["POST"])]

    # Run deploy_pipeline_files without saving the files
    result = deploy_pipeline_files(app=mock_app, pipeline_name="test_pipeline", files=files, save_files=False)
    assert result == {"name": "test_pipeline"}

    # Check that pipeline files are not saved to disk
    pipeline_dir = Path(test_settings.pipelines_dir) / "test_pipeline"
    assert not pipeline_dir.exists()

    # Verify the pipeline was deployed successfully
    assert result == {"name": "test_pipeline"}

    # Verify FastAPI routes were set up
    assert mock_app.add_api_route.called
    assert mock_app.setup.called
