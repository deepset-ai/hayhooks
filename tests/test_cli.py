import pytest
from typer.testing import CliRunner

import hayhooks.cli.base as base_module
from hayhooks.cli.base import hayhooks_cli
from hayhooks.settings import settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def set_default_settings():
    """
    Ensure that settings always have default values for tests.
    The callback in our CLI reads these values.
    """
    settings.host = "localhost"
    settings.port = 1416
    settings.disable_ssl = False
    settings.pipelines_dir = "dummy_pipelines"
    settings.root_path = "/"
    settings.dashboard_enabled = False
    settings.dashboard_path = "/dashboard"
    settings.dashboard_dist_dir = "dashboard/dist"
    settings.durable_execution_concurrency = 1
    settings.a2a_task_store = "memory"
    settings.a2a_task_store_provider = ""
    settings.a2a_redis_url = "redis://localhost:6379/0"
    settings.a2a_redis_key_prefix = "hayhooks:a2a"


def test_run_command_with_reload(monkeypatch):
    """
    With --reload, the CLI must use uvicorn.run with a string import path
    and factory=True (uvicorn requirement for reload/multi-worker).
    """
    import uvicorn

    calls = []

    def fake_uvicorn_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    result = runner.invoke(
        hayhooks_cli,
        ["run", "--host", "localhost", "--port", "1416", "--reload"],
    )
    assert result.exit_code == 0, result.output
    assert calls, "uvicorn.run was not called"

    args, kwargs = calls[0]
    assert args[0] == "hayhooks.server.app:create_app"
    assert kwargs.get("host") == "localhost"
    assert kwargs.get("port") == 1416
    assert kwargs.get("reload") is True
    assert kwargs.get("factory") is True


def test_run_command_with_workers(monkeypatch):
    """
    With --workers > 1, the CLI must use the string import path (same as reload).
    """
    import uvicorn

    calls = []

    def fake_uvicorn_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    result = runner.invoke(
        hayhooks_cli,
        ["run", "--host", "localhost", "--port", "1416", "--workers", "4"],
    )
    assert result.exit_code == 0, result.output
    assert calls, "uvicorn.run was not called"

    args, kwargs = calls[0]
    assert args[0] == "hayhooks.server.app:create_app"
    assert kwargs.get("workers") == 4
    assert kwargs.get("factory") is True


def test_run_command_single_worker(monkeypatch):
    """
    Without --reload or --workers > 1, the CLI uses run_app() which passes
    the app object directly to uvicorn.run.
    """
    import uvicorn
    from fastapi import FastAPI

    calls = []

    def fake_uvicorn_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    result = runner.invoke(
        hayhooks_cli,
        ["run", "--host", "localhost", "--port", "1416"],
    )
    assert result.exit_code == 0, result.output
    assert calls, "uvicorn.run was not called"

    args, kwargs = calls[0]
    assert isinstance(args[0], FastAPI), "run_app should pass a FastAPI object, not a string"
    assert kwargs.get("host") == "localhost"
    assert kwargs.get("port") == 1416


def test_run_command_with_tracing_dashboard_flag(monkeypatch, tmp_path):
    import uvicorn

    calls = []
    build_inputs = []
    built_dist_dir = tmp_path / "built-dashboard-dist"

    def fake_uvicorn_run(*args, **kwargs):
        calls.append((args, kwargs))

    def fake_prepare_tracing_dashboard_assets(dashboard_dist_dir: str) -> str:
        build_inputs.append(dashboard_dist_dir)
        return str(built_dist_dir)

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setattr(base_module, "_prepare_tracing_dashboard_assets", fake_prepare_tracing_dashboard_assets)
    monkeypatch.setattr(base_module, "_dashboard_assets_available", lambda _dir: False)

    result = runner.invoke(
        hayhooks_cli,
        ["run", "--host", "localhost", "--port", "1416", "--reload", "--with-tracing-dashboard"],
    )
    assert result.exit_code == 0, result.output
    assert calls, "uvicorn.run was not called"
    assert build_inputs == ["dashboard/dist"]
    assert settings.dashboard_enabled is True
    assert settings.dashboard_dist_dir == str(built_dist_dir)


def test_a2a_run_debug_enables_tracebacks(monkeypatch):
    import uvicorn

    from hayhooks.server.utils import a2a_utils, deploy_utils
    from hayhooks.settings import settings

    calls = []

    def fake_uvicorn_run(*args, **kwargs):
        calls.append((args, kwargs))

    def fake_create_a2a_app(*, debug: bool = False):
        return object()

    def fake_deploy_pipelines() -> None:
        return None

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setattr(deploy_utils, "deploy_pipelines", fake_deploy_pipelines)
    monkeypatch.setattr(a2a_utils, "create_a2a_app", fake_create_a2a_app)
    # Neutralize the optional-dependency guard so this CLI-plumbing test runs
    # even when the optional 'a2a' package isn't installed.
    monkeypatch.setattr(a2a_utils.a2a_import, "check", lambda: None)

    settings.show_tracebacks = False
    result = runner.invoke(hayhooks_cli, ["a2a", "run", "--debug", "--pipelines-dir", "dummy_pipelines"])

    assert result.exit_code == 0, result.output
    assert settings.show_tracebacks is True
    assert calls, "uvicorn.run was not called"


def test_a2a_run_sets_task_store_provider(monkeypatch):
    import uvicorn

    from hayhooks.server.utils import a2a_utils, deploy_utils
    from hayhooks.settings import settings

    def fake_uvicorn_run(*_args, **_kwargs):
        return None

    def fake_create_a2a_app(**_kwargs):
        return object()

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setattr(deploy_utils, "deploy_pipelines", lambda: None)
    monkeypatch.setattr(a2a_utils, "create_a2a_app", fake_create_a2a_app)
    monkeypatch.setattr(a2a_utils.a2a_import, "check", lambda: None)
    monkeypatch.setattr(settings, "a2a_task_store_provider", "")

    result = runner.invoke(
        hayhooks_cli,
        [
            "a2a",
            "run",
            "--task-store-provider",
            "my_project.a2a:ProjectTaskStoreProvider",
            "--pipelines-dir",
            "dummy_pipelines",
        ],
    )

    assert result.exit_code == 0, result.output
    assert settings.a2a_task_store_provider == "my_project.a2a:ProjectTaskStoreProvider"


def test_a2a_run_sets_builtin_redis_task_store(monkeypatch):
    import uvicorn

    from hayhooks.server.utils import a2a_utils, deploy_utils

    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(deploy_utils, "deploy_pipelines", lambda: None)
    monkeypatch.setattr(a2a_utils, "create_a2a_app", lambda **_kwargs: object())
    monkeypatch.setattr(a2a_utils.a2a_import, "check", lambda: None)

    result = runner.invoke(
        hayhooks_cli,
        [
            "a2a",
            "run",
            "--task-store",
            "redis",
            "--a2a-redis-url",
            "redis://localhost:6379/4",
            "--a2a-redis-key-prefix",
            "demo:a2a",
            "--pipelines-dir",
            "dummy_pipelines",
        ],
    )

    assert result.exit_code == 0, result.output
    assert settings.a2a_task_store == "redis"
    assert settings.a2a_redis_url == "redis://localhost:6379/4"
    assert settings.a2a_redis_key_prefix == "demo:a2a"


def test_a2a_run_rejects_ambiguous_task_store_configuration(monkeypatch):
    from hayhooks.server.utils import a2a_utils, deploy_utils

    deployed = False

    def record_deploy():
        nonlocal deployed
        deployed = True

    monkeypatch.setattr(deploy_utils, "deploy_pipelines", record_deploy)
    monkeypatch.setattr(a2a_utils.a2a_import, "check", lambda: None)

    result = runner.invoke(
        hayhooks_cli,
        [
            "a2a",
            "run",
            "--task-store",
            "redis",
            "--task-store-provider",
            "my_project.a2a:ProjectTaskStoreProvider",
        ],
    )

    assert result.exit_code != 0
    assert "task-store-provider" in result.output
    assert "together" in result.output
    assert not deployed


def test_a2a_run_sets_durable_execution_concurrency(monkeypatch):
    import uvicorn

    from hayhooks.server.utils import a2a_utils, deploy_utils

    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(deploy_utils, "deploy_pipelines", lambda: None)
    monkeypatch.setattr(a2a_utils, "create_a2a_app", lambda **_kwargs: object())
    monkeypatch.setattr(a2a_utils.a2a_import, "check", lambda: None)

    result = runner.invoke(
        hayhooks_cli,
        ["a2a", "run", "--durable-execution-concurrency", "3", "--pipelines-dir", "dummy_pipelines"],
    )

    assert result.exit_code == 0, result.output
    assert settings.durable_execution_concurrency == 3


def test_a2a_run_sets_durable_execution_store_configuration(monkeypatch):
    import uvicorn

    from hayhooks.server.utils import a2a_utils, deploy_utils

    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(deploy_utils, "deploy_pipelines", lambda: None)
    monkeypatch.setattr(a2a_utils, "create_a2a_app", lambda **_kwargs: object())
    monkeypatch.setattr(a2a_utils.a2a_import, "check", lambda: None)

    result = runner.invoke(
        hayhooks_cli,
        [
            "a2a",
            "run",
            "--execution-store",
            "redis",
            "--execution-redis-url",
            "redis://localhost:6379/5",
            "--execution-redis-key-prefix",
            "demo:durable",
            "--pipelines-dir",
            "dummy_pipelines",
        ],
    )

    assert result.exit_code == 0, result.output
    assert settings.durable_store == "redis"
    assert settings.durable_redis_url == "redis://localhost:6379/5"
    assert settings.durable_redis_key_prefix == "demo:durable"


def test_status_command(monkeypatch):
    """
    Test the status command. We patch make_request (used in the status command)
    to return a dummy response.
    """
    from hayhooks.cli import base

    fake_response = {"pipelines": ["test_pipeline"]}

    def fake_make_request(*args, **kwargs):
        return fake_response

    monkeypatch.setattr(base, "make_request", fake_make_request)

    # Invoke the status command.
    result = runner.invoke(hayhooks_cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "Hayhooks server is up and running" in result.output
    assert "test_pipeline" in result.output


def test_deploy_files_command_name_options(monkeypatch, tmp_path):
    from hayhooks.cli import pipeline

    # Create a dummy pipeline directory with a file
    pipeline_dir = tmp_path / "test_pipeline"
    pipeline_dir.mkdir()
    (pipeline_dir / "main.py").write_text("print('test')")

    calls = []

    def fake_deploy_with_progress(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(pipeline, "_deploy_with_progress", fake_deploy_with_progress)

    # Test long form --name
    result = runner.invoke(
        hayhooks_cli,
        ["pipeline", "deploy-files", "--name", "test-long", str(pipeline_dir)],
    )
    assert result.exit_code == 0
    assert calls[0]["name"] == "test-long"

    # Test short form -n
    result = runner.invoke(
        hayhooks_cli,
        ["pipeline", "deploy-files", "-n", "test-short", str(pipeline_dir)],
    )
    assert result.exit_code == 0
    assert calls[1]["name"] == "test-short"


def test_cli_undeploy_command(monkeypatch):
    from typer.testing import CliRunner

    import hayhooks.cli.pipeline as pipeline_module
    from hayhooks.cli.base import hayhooks_cli

    runner = CliRunner()

    # Mock the make_request function to return a successful response
    def mock_make_request(*args, **kwargs):
        return {"success": True, "name": "test_pipeline"}

    monkeypatch.setattr(pipeline_module, "make_request", mock_make_request)

    # Test the undeploy command
    result = runner.invoke(hayhooks_cli, ["pipeline", "undeploy", "test_pipeline"])
    assert result.exit_code == 0
    assert "successfully undeployed" in result.stdout.lower()

    # Mock the make_request function to return an error response
    def mock_make_request_error(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline_module, "make_request", mock_make_request_error)

    # Test the undeploy command with an error
    result = runner.invoke(hayhooks_cli, ["pipeline", "undeploy", "nonexistent_pipeline"])
    assert result.exit_code != 0
    assert "error" in result.stdout.lower()


def test_pipeline_run_with_dir_uploads(monkeypatch, tmp_path):
    import hayhooks.cli.utils as utils_module
    from hayhooks.cli.base import hayhooks_cli

    # Create directory and files
    upload_dir = tmp_path / "files_to_index"
    (upload_dir / "nested").mkdir(parents=True)
    contents = {"a.txt": b"hello A", "b.md": b"hello B", "nested/c.log": b"hello C"}
    for rel_path, data in contents.items():
        p = upload_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    expected_by_name = {rel.split("/")[-1]: data for rel, data in contents.items()}
    expected_names = set(expected_by_name.keys())
    received_names: list[str] = []

    def fake_post(url, data=None, files=None, verify=True, **kwargs):
        assert url.endswith("/indexing/run")
        assert verify is True
        received_names.clear()
        for _, (filename, file_obj, _ctype) in files:
            assert file_obj.read() == expected_by_name[filename]
            received_names.append(filename)

        class _R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"result": {"files": received_names}}

        return _R()

    monkeypatch.setattr(utils_module.requests, "post", fake_post)

    result = runner.invoke(hayhooks_cli, ["pipeline", "run", "indexing", "--dir", str(upload_dir)])
    assert result.exit_code == 0
    assert set(received_names) == expected_names
    assert "executed successfully" in result.stdout.lower()
