import inspect
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import docstring_parser
import pytest
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from haystack import Pipeline

from hayhooks.server.exceptions import PipelineFilesError, PipelineModuleLoadError, PipelineWrapperError
from hayhooks.server.pipelines import registry
from hayhooks.server.pipelines.sse import SSEStream
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.deploy_utils import (
    _streaming_response_from_result,
    create_request_model_from_callable,
    create_response_model_from_callable,
    deploy_pipeline_files,
    save_pipeline_files,
    undeploy_pipeline,
)
from hayhooks.server.utils.module_loader import (
    create_pipeline_wrapper_instance,
    load_pipeline_module,
    unload_pipeline_modules,
)


@pytest.fixture(autouse=True)
def cleanup_test_pipelines(test_settings):
    yield

    registry.clear()
    if Path(test_settings.pipelines_dir).exists():
        shutil.rmtree(test_settings.pipelines_dir)


def test_streaming_response_from_gen_honors_media_type():
    def generator():
        yield "chunk"

    response = _streaming_response_from_result(SSEStream(generator()))

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"


@pytest.mark.asyncio
async def test_streaming_response_from_async_gen_honors_media_type():
    async def async_generator():
        yield "chunk"

    sse_stream = SSEStream(async_generator())
    response = _streaming_response_from_result(sse_stream)

    try:
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
    finally:
        await sse_stream.stream.aclose()


def test_load_pipeline_module():
    pipeline_name = "chat_with_website"
    pipeline_dir_path = Path("tests/test_files/files/chat_with_website")

    module = load_pipeline_module(pipeline_name, pipeline_dir_path)

    assert module is not None
    assert hasattr(module, "PipelineWrapper")
    assert isinstance(module.PipelineWrapper.run_api, Callable)
    assert isinstance(module.PipelineWrapper.run_chat_completion, Callable)
    assert isinstance(module.PipelineWrapper.setup, Callable)


def test_load_pipeline_module_async():
    pipeline_name = "async_chat_with_website"
    pipeline_dir_path = Path("tests/test_files/files/async_chat_with_website")

    module = load_pipeline_module(pipeline_name, pipeline_dir_path)

    assert module is not None
    assert hasattr(module, "PipelineWrapper")
    assert isinstance(module.PipelineWrapper.run_api_async, Callable)
    assert isinstance(module.PipelineWrapper.run_chat_completion_async, Callable)
    assert isinstance(module.PipelineWrapper.setup, Callable)


def test_load_pipeline_module_with_relative_imports():
    pipeline_name = "with_helper_module"
    pipeline_dir_path = Path("tests/test_files/files/with_helper_module")
    wrapper_module_name = f"{pipeline_name}.pipeline_wrapper"

    # Ensure modules are not already loaded
    unload_pipeline_modules(pipeline_name)

    module = load_pipeline_module(pipeline_name, pipeline_dir_path)

    # Verify module was loaded correctly
    assert module is not None
    assert hasattr(module, "PipelineWrapper")
    assert isinstance(module.PipelineWrapper.run_api, Callable)

    # Verify package and wrapper module are registered in sys.modules
    assert pipeline_name in sys.modules  # Package module
    assert wrapper_module_name in sys.modules  # Wrapper module
    assert sys.modules[wrapper_module_name] is module

    # Verify __package__ is set correctly for relative imports
    assert module.__package__ == pipeline_name

    # Verify the parent directory is in sys.path
    parent_dir_str = str(pipeline_dir_path.resolve().parent)
    assert parent_dir_str in sys.path

    # Create an instance and verify the relative imports work
    wrapper = module.PipelineWrapper()
    wrapper.setup()
    result = wrapper.run_api(name="World", a=3, b=4)

    assert result["greeting"] == "Hello, World!"
    assert result["multiply_result"] == 12


def test_load_pipeline_module_registers_in_sys_modules():
    pipeline_name = "test_sys_modules_registration"
    pipeline_dir_path = Path("tests/test_files/files/chat_with_website")
    wrapper_module_name = f"{pipeline_name}.pipeline_wrapper"

    # Ensure modules are not already loaded
    unload_pipeline_modules(pipeline_name)

    module = load_pipeline_module(pipeline_name, pipeline_dir_path)

    # Verify both package and wrapper module are registered in sys.modules
    assert pipeline_name in sys.modules  # Package module
    assert wrapper_module_name in sys.modules  # Wrapper module
    assert sys.modules[wrapper_module_name] is module

    # Verify the module can be found by name (simulates what tracing libraries do)
    imported_module = sys.modules[wrapper_module_name]
    assert imported_module is module
    assert hasattr(imported_module, "PipelineWrapper")


def test_load_pipeline_module_with_init_py():
    pipeline_name = "with_init_py"
    pipeline_dir_path = Path("tests/test_files/files/with_init_py")

    # Ensure modules are not already loaded
    unload_pipeline_modules(pipeline_name)

    module = load_pipeline_module(pipeline_name, pipeline_dir_path)

    # Verify module was loaded correctly
    assert module is not None
    assert hasattr(module, "PipelineWrapper")

    # Verify __init__.py was executed (PACKAGE_VERSION should be available)
    package_module = sys.modules[pipeline_name]
    assert hasattr(package_module, "PACKAGE_VERSION")
    assert package_module.PACKAGE_VERSION == "1.0.0"

    # Verify the wrapper works with imports from both __init__.py and helpers.py
    wrapper = module.PipelineWrapper()
    wrapper.setup()
    result = wrapper.run_api(value=5)

    assert result["result"] == 10  # double(5) = 10
    assert result["version"] == "1.0.0"


def test_unload_pipeline_modules():
    pipeline_name = "test_unload"
    pipeline_dir_path = Path("tests/test_files/files/chat_with_website")

    # First, load the module
    unload_pipeline_modules(pipeline_name)
    load_pipeline_module(pipeline_name, pipeline_dir_path)

    # Verify modules are in sys.modules
    assert pipeline_name in sys.modules
    assert f"{pipeline_name}.pipeline_wrapper" in sys.modules

    # Unload the modules
    unload_pipeline_modules(pipeline_name)

    # Verify modules are removed
    assert pipeline_name not in sys.modules
    assert f"{pipeline_name}.pipeline_wrapper" not in sys.modules


def test_unload_pipeline_modules_nonexistent():
    # Should not raise any errors
    unload_pipeline_modules("nonexistent_pipeline_xyz")


def test_load_pipeline_module_cleans_up_on_import_error():
    pipeline_name = "broken_import"
    pipeline_dir_path = Path("tests/test_files/files/broken_import")

    # Ensure modules are not already loaded
    unload_pipeline_modules(pipeline_name)

    # Loading should fail due to broken import
    with pytest.raises(PipelineModuleLoadError, match="nonexistent_module"):
        load_pipeline_module(pipeline_name, pipeline_dir_path)

    # Verify modules were cleaned up from sys.modules
    assert pipeline_name not in sys.modules
    assert f"{pipeline_name}.pipeline_wrapper" not in sys.modules


def test_load_pipeline_module_reload():
    pipeline_name = "test_reload"
    pipeline_dir_path = Path("tests/test_files/files/chat_with_website")

    # Load the module first time
    unload_pipeline_modules(pipeline_name)
    module1 = load_pipeline_module(pipeline_name, pipeline_dir_path)

    # Load the module second time (should replace the first)
    module2 = load_pipeline_module(pipeline_name, pipeline_dir_path)

    # The second load should have replaced the first in sys.modules
    assert sys.modules[f"{pipeline_name}.pipeline_wrapper"] is module2
    assert module1 is not module2  # Different module objects


def test_multiple_pipelines_with_helpers_dont_interfere():
    # Load first pipeline
    pipeline1_name = "helper_test_1"
    pipeline1_dir = Path("tests/test_files/files/with_helper_module")
    unload_pipeline_modules(pipeline1_name)
    module1 = load_pipeline_module(pipeline1_name, pipeline1_dir)

    # Load second pipeline (reusing the same test files but with different name)
    pipeline2_name = "helper_test_2"
    pipeline2_dir = Path("tests/test_files/files/with_init_py")
    unload_pipeline_modules(pipeline2_name)
    module2 = load_pipeline_module(pipeline2_name, pipeline2_dir)

    # Both should be loaded and independent
    assert pipeline1_name in sys.modules
    assert pipeline2_name in sys.modules
    assert f"{pipeline1_name}.pipeline_wrapper" in sys.modules
    assert f"{pipeline2_name}.pipeline_wrapper" in sys.modules

    # Create instances and verify they work independently
    wrapper1 = module1.PipelineWrapper()
    wrapper1.setup()
    result1 = wrapper1.run_api(name="Test1", a=2, b=3)

    wrapper2 = module2.PipelineWrapper()
    wrapper2.setup()
    result2 = wrapper2.run_api(value=7)

    # Verify results are correct for each pipeline
    assert result1["greeting"] == "Hello, Test1!"
    assert result1["multiply_result"] == 6
    assert result2["result"] == 14
    assert result2["version"] == "1.0.0"

    # Clean up
    unload_pipeline_modules(pipeline1_name)
    unload_pipeline_modules(pipeline2_name)


def test_load_pipeline_wrong_dir():
    pipeline_name = "chat_with_website"
    pipeline_dir_path = Path("tests/test_files/files/wrong_dir")

    with pytest.raises(
        PipelineModuleLoadError,
        match=r"pipeline_wrapper\.py.*not found",
    ):
        load_pipeline_module(pipeline_name, pipeline_dir_path)


def test_load_pipeline_no_wrapper():
    pipeline_name = "chat_with_website"
    pipeline_dir_path = Path("tests/test_files/files/no_wrapper")

    with pytest.raises(
        PipelineModuleLoadError,
        match=r"pipeline_wrapper\.py.*not found",
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
    assert len(list((Path(test_settings.pipelines_dir) / pipeline_name).iterdir())) == 0


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
        """
        Sample function with docstring.

        Args:
            name: The name of the person.
            age: The age of the person.
            optional: An optional string.
        """
        pass

    docstring = docstring_parser.parse(inspect.getdoc(sample_func) or "")
    model = create_request_model_from_callable(sample_func, "Test", docstring)
    schema = model.model_json_schema()

    assert model.__name__ == "TestRequest"
    assert schema["properties"]["name"]["type"] == "string"
    assert "default" not in schema["properties"]["name"]
    assert schema["properties"]["name"]["description"] == "The name of the person."
    assert "name" in schema["required"]

    assert schema["properties"]["age"]["type"] == "integer"
    assert schema["properties"]["age"]["default"] == 25
    assert schema["properties"]["age"]["description"] == "The age of the person."
    assert "age" not in schema.get("required", [])

    assert schema["properties"]["optional"]["type"] == "string"
    assert schema["properties"]["optional"]["default"] == ""
    assert schema["properties"]["optional"]["description"] == "An optional string."
    assert "optional" not in schema.get("required", [])


def test_create_request_model_no_docstring():
    def sample_func_no_doc(name: str, age: int = 30):
        pass

    docstring = docstring_parser.parse(inspect.getdoc(sample_func_no_doc) or "")
    model = create_request_model_from_callable(sample_func_no_doc, "NoDoc", docstring)
    schema = model.model_json_schema()

    assert model.__name__ == "NoDocRequest"
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["name"]["description"] == "Parameter 'name'"
    assert "name" in schema["required"]

    assert schema["properties"]["age"]["type"] == "integer"
    assert schema["properties"]["age"]["default"] == 30
    assert schema["properties"]["age"]["description"] == "Parameter 'age'"
    assert "age" not in schema.get("required", [])


def test_create_request_model_partial_docstring():
    def sample_func_partial_doc(documented_param: str, undocumented_param: int = 42):
        """
        Sample function with partial docstring.

        Args:
            documented_param: This parameter is documented.
        """
        pass

    docstring = docstring_parser.parse(inspect.getdoc(sample_func_partial_doc) or "")
    model = create_request_model_from_callable(sample_func_partial_doc, "PartialDoc", docstring)
    schema = model.model_json_schema()

    assert model.__name__ == "PartialDocRequest"

    assert schema["properties"]["documented_param"]["type"] == "string"
    assert "default" not in schema["properties"]["documented_param"]
    assert schema["properties"]["documented_param"]["description"] == "This parameter is documented."
    assert "documented_param" in schema["required"]

    assert schema["properties"]["undocumented_param"]["type"] == "integer"
    assert schema["properties"]["undocumented_param"]["default"] == 42
    assert schema["properties"]["undocumented_param"]["description"] == "Parameter 'undocumented_param'"
    assert "undocumented_param" not in schema.get("required", [])


def test_create_response_model_from_callable():
    def sample_func() -> dict:
        """
        Sample function with return description.

        Returns:
            A dictionary result.
        """
        return {"result": "test"}

    docstring = docstring_parser.parse(inspect.getdoc(sample_func) or "")
    model = create_response_model_from_callable(sample_func, "Test", docstring)
    schema = model.model_json_schema()

    assert model.__name__ == "TestResponse"
    assert schema["properties"]["result"]["type"] == "object"
    assert "default" not in schema["properties"]["result"]
    assert schema["properties"]["result"]["description"] == "A dictionary result."
    assert "result" in schema["required"]


def test_create_response_model_no_docstring():
    def sample_func_no_doc() -> int:
        return 1

    docstring = docstring_parser.parse(inspect.getdoc(sample_func_no_doc) or "")
    model = create_response_model_from_callable(sample_func_no_doc, "NoDoc", docstring)
    schema = model.model_json_schema()

    assert model.__name__ == "NoDocResponse"
    assert schema["properties"]["result"]["type"] == "integer"
    assert schema["properties"]["result"].get("description") is None
    assert "result" in schema["required"]


def test_create_pipeline_wrapper_instance_success():
    class ValidPipelineWrapper(BasePipelineWrapper):
        def setup(self):
            self.pipeline = Pipeline()

        def run_api(self):
            pass

        def run_chat_completion(self, model, messages, body):
            pass

    module = type("Module", (), {"PipelineWrapper": ValidPipelineWrapper})

    wrapper = create_pipeline_wrapper_instance(module)
    assert isinstance(wrapper, BasePipelineWrapper)
    assert hasattr(wrapper, "run_api")
    assert hasattr(wrapper, "run_chat_completion")
    assert isinstance(wrapper.pipeline, Pipeline)


def test_create_pipeline_wrapper_instance_init_error():
    class BrokenPipelineWrapper:
        def __init__(self):
            msg = "Init error"
            raise ValueError(msg)

    module = type("Module", (), {"PipelineWrapper": BrokenPipelineWrapper})

    with pytest.raises(PipelineWrapperError, match="Failed to create pipeline wrapper instance: Init error"):
        create_pipeline_wrapper_instance(module)


def test_create_pipeline_wrapper_instance_setup_error():
    class BrokenSetupWrapper(BasePipelineWrapper):
        def setup(self):
            msg = "Setup error"
            raise ValueError(msg)

        def run_api(self):
            pass

    module = type("Module", (), {"PipelineWrapper": BrokenSetupWrapper})

    with pytest.raises(
        PipelineWrapperError, match=re.escape("Failed to call setup() on pipeline wrapper instance: Setup error")
    ):
        create_pipeline_wrapper_instance(module)


def test_create_pipeline_wrapper_instance_missing_methods():
    class IncompleteWrapper(BasePipelineWrapper):
        def setup(self):
            self.pipeline = Pipeline()

    module = type("Module", (), {"PipelineWrapper": IncompleteWrapper})

    with pytest.raises(
        PipelineWrapperError,
        match=re.escape("At least one of run_api, run_api_async, run_chat_completion, or run_chat_completion_async"),
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


def test_deploy_pipeline_files_without_adding_api_route(test_settings, mocker):
    mock_app = mocker.Mock()

    # We're saving the pipeline wrapper file in the test_files directory
    test_file_path = Path("tests/test_files/files/no_chat/pipeline_wrapper.py")
    files = {"pipeline_wrapper.py": test_file_path.read_text()}

    # Run deploy_pipeline_files without adding the API route
    # NOTE: we can also simply omit the `app` argument
    result = deploy_pipeline_files(pipeline_name="test_pipeline_no_route", files=files, app=None, save_files=False)
    assert result == {"name": "test_pipeline_no_route"}

    # Verify the pipeline was deployed successfully
    assert result == {"name": "test_pipeline_no_route"}
    assert registry.get("test_pipeline_no_route") is not None

    # Verify FastAPI routes were NOT set up
    assert not mock_app.add_api_route.called
    assert not mock_app.setup.called


def test_deploy_pipeline_files_skip_mcp(mocker):
    mock_app = mocker.Mock()
    mock_app.routes = []

    # This pipeline wrapper has skip_mcp class attribute set to True
    test_file_path = Path("tests/test_files/files/chat_with_website_mcp_skip/pipeline_wrapper.py")
    files = {"pipeline_wrapper.py": test_file_path.read_text()}

    result = deploy_pipeline_files(
        app=mock_app, pipeline_name="chat_with_website_mcp_skip", files=files, save_files=False
    )
    assert result == {"name": "chat_with_website_mcp_skip"}

    assert registry.get_metadata("chat_with_website_mcp_skip").get("skip_mcp") is True


def test_undeploy_pipeline_without_app(test_settings):
    pipeline_name = "test_undeploy_no_app"
    test_file_path = Path("tests/test_files/files/no_chat/pipeline_wrapper.py")
    files = {"pipeline_wrapper.py": test_file_path.read_text()}

    # 1. Deploy a dummy pipeline without passing an Hayhooks app instance
    deploy_pipeline_files(pipeline_name=pipeline_name, files=files)

    # Assert initial state: pipeline in registry and files exist
    assert registry.get(pipeline_name) is not None

    pipeline_dir = Path(test_settings.pipelines_dir) / pipeline_name
    assert pipeline_dir.exists()
    assert (pipeline_dir / "pipeline_wrapper.py").exists()

    # 2. Call undeploy_pipeline without passing an Hayhooks app instance
    undeploy_pipeline(pipeline_name=pipeline_name)

    # 3. Assert pipeline is removed from registry
    assert registry.get(pipeline_name) is None

    # 4. Assert pipeline files are deleted
    assert not pipeline_dir.exists()


def test_deploy_pipeline_files_with_async_run_api():
    pipeline_name = "async_question_answer"
    pipeline_wrapper_path = Path("tests/test_files/files/async_question_answer/pipeline_wrapper.py")
    pipeline_yml_path = Path("tests/test_files/files/async_question_answer/question_answer.yml")
    files = {
        "pipeline_wrapper.py": pipeline_wrapper_path.read_text(),
        "question_answer.yml": pipeline_yml_path.read_text(),
    }

    deploy_pipeline_files(pipeline_name=pipeline_name, files=files, save_files=False)

    pipeline_wrapper = registry.get(pipeline_name)
    assert pipeline_wrapper is not None

    metadata = registry.get_metadata(pipeline_name)
    assert metadata is not None
    assert "request_model" in metadata
    assert metadata["request_model"] is not None

    assert pipeline_wrapper._is_run_api_async_implemented is True
    assert pipeline_wrapper._is_run_api_implemented is False

    request_model = metadata["request_model"]
    schema = request_model.model_json_schema()
    assert "question" in schema["properties"]
    assert schema["properties"]["question"]["type"] == "string"


def test_deploy_pipeline_files_without_return_type(test_settings, mocker):
    mock_app = mocker.Mock()

    test_file_path = Path("tests/test_files/files/no_return_type/pipeline_wrapper.py")
    files = {"pipeline_wrapper.py": test_file_path.read_text()}

    with pytest.raises(
        PipelineWrapperError, match=re.escape("Pipeline wrapper is missing a return type for 'run_api' method")
    ):
        deploy_pipeline_files(app=mock_app, pipeline_name="test_pipeline_no_return_type", files=files, save_files=False)


def test_deploy_pipeline_files_with_utf8_characters(test_settings):
    """Test that pipeline files with UTF-8 characters are loaded and deployed correctly."""
    test_file_path = Path("tests/test_files/files/utf8_pipeline/pipeline_wrapper.py")
    files = {"pipeline_wrapper.py": test_file_path.read_text(encoding="utf-8")}

    # Verify UTF-8 characters are present in the file contents
    assert "你好世界" in files["pipeline_wrapper.py"]
    assert "🌍" in files["pipeline_wrapper.py"]
    assert "こんにちは" in files["pipeline_wrapper.py"]
    assert "мир" in files["pipeline_wrapper.py"]

    # Deploy the pipeline without an app (no routes added)
    result = deploy_pipeline_files(pipeline_name="utf8_test_pipeline", files=files, save_files=False)
    assert result == {"name": "utf8_test_pipeline"}

    # Verify the pipeline was deployed successfully
    assert registry.get("utf8_test_pipeline") is not None

    # Verify the pipeline can be called and returns UTF-8 characters
    pipeline_wrapper = registry.get("utf8_test_pipeline")
    response = pipeline_wrapper.run_api(test_param="test")
    assert "你好世界" in response["message"]
    assert "🌍" in response["message"]
    assert "мир" in response["message"]


def test_save_pipeline_files_with_utf8_characters(test_settings):
    """Test that pipeline files with UTF-8 characters can be saved and loaded correctly."""
    files = {
        "pipeline_wrapper.py": """# -*- coding: utf-8 -*-
# Test UTF-8: 你好世界 🌍 こんにちは мир
from haystack import Pipeline
from hayhooks import BasePipelineWrapper

class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = Pipeline()

    def run_api(self, param: str) -> dict:
        return {"result": f"UTF-8 test: {param} 你好 🌍"}
""",
        "config.txt": "Configuration with UTF-8: 测试 🚀 привет",
    }

    saved_paths = save_pipeline_files("utf8_pipeline_save_test", files, pipelines_dir=test_settings.pipelines_dir)

    assert len(saved_paths) == 2
    for filename, path in saved_paths.items():
        assert Path(path).exists()
        # Read back with UTF-8 encoding and verify contents
        content = Path(path).read_text(encoding="utf-8")
        assert content == files[filename]

        # Verify UTF-8 characters are preserved
        if filename == "pipeline_wrapper.py":
            assert "你好世界" in content
            assert "🌍" in content
        elif filename == "config.txt":
            assert "测试" in content
            assert "🚀" in content
