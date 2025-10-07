import pytest
from typer.testing import CliRunner

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


def test_run_command(monkeypatch):
    """
    Test that the `run` command calls uvicorn.run with the provided options.
    """
    import uvicorn

    calls = []

    def fake_uvicorn_run(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    # Invoke the run command with custom host, port, and reload flag.
    result = runner.invoke(
        hayhooks_cli,
        [
            "run",
            "--host",
            "localhost",
            "--port",
            "1416",
            "--reload",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls, "uvicorn.run was not called"

    uvicorn_kwargs = calls[0]
    assert uvicorn_kwargs.get("host") == "localhost"
    assert uvicorn_kwargs.get("port") == 1416
    assert uvicorn_kwargs.get("reload") is True


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
