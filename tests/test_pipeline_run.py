from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from hayhooks.cli.base import hayhooks_cli


@pytest.fixture
def runner():
    return CliRunner()


def create_test_files(base_dir, file_dict):
    """
    Helper to create test files from a dictionary.

    Args:
        base_dir: The base directory to create files in
        file_dict: Dict mapping relative paths to file content

    Returns:
        Dictionary mapping filenames to Path objects
    """
    paths = {}
    for rel_path, content in file_dict.items():
        path = base_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is not None:  # None means directory
            path.write_text(content)
        else:
            path.mkdir(exist_ok=True)
        paths[rel_path] = path
    return paths


def test_run_with_string_params(runner, monkeypatch):
    mock_run = Mock()
    monkeypatch.setattr("hayhooks.cli.pipeline.run_pipeline_with_files", mock_run)

    result = runner.invoke(
        hayhooks_cli, ["pipeline", "run", "test_pipeline", "--param", "key1=value1", "--param", "key2=value2"]
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["pipeline_name"] == "test_pipeline"
    assert kwargs["params"] == {"key1": "value1", "key2": "value2"}
    assert kwargs["files"] == {}


def test_run_with_json_params(runner, monkeypatch):
    mock_run = Mock()
    monkeypatch.setattr("hayhooks.cli.pipeline.run_pipeline_with_files", mock_run)

    result = runner.invoke(
        hayhooks_cli,
        [
            "pipeline",
            "run",
            "test_pipeline",
            "--param",
            "list=[1,2,3]",
            "--param",
            'dict={"a":1,"b":2}',
            "--param",
            'list_of_strings=["a","b","c"]',
        ],
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["pipeline_name"] == "test_pipeline"
    assert kwargs["params"] == {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}, "list_of_strings": ["a", "b", "c"]}


def test_run_with_invalid_param_format(runner):
    result = runner.invoke(hayhooks_cli, ["pipeline", "run", "test_pipeline", "--param", "invalid_format"])

    assert result.exit_code != 0
    assert "Invalid parameter format" in result.stdout


def test_run_with_file(runner, tmp_path, monkeypatch):
    files = create_test_files(tmp_path, {"test.txt": "test content"})

    mock_run = Mock()
    monkeypatch.setattr("hayhooks.cli.pipeline.run_pipeline_with_files", mock_run)

    result = runner.invoke(hayhooks_cli, ["pipeline", "run", "test_pipeline", "--file", str(files["test.txt"])])

    assert result.exit_code == 0
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["pipeline_name"] == "test_pipeline"
    assert len(kwargs["files"]) == 1
    assert "test.txt" in kwargs["files"]


def test_run_with_mixed_params(runner, monkeypatch):
    mock_run = Mock()
    monkeypatch.setattr("hayhooks.cli.pipeline.run_pipeline_with_files", mock_run)

    result = runner.invoke(
        hayhooks_cli,
        [
            "pipeline",
            "run",
            "test_pipeline",
            "--param",
            "string=value",
            "--param",
            "number=42",
            "--param",
            "array=[1,2,3]",
        ],
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["pipeline_name"] == "test_pipeline"
    assert kwargs["params"] == {
        "string": "value",
        "number": 42,  # Should be converted to integer
        "array": [1, 2, 3],  # Should be parsed as JSON array
    }


def test_run_with_complex_json_param(runner, monkeypatch):
    complex_json = '{"name":"test","values":[1,2,3],"nested":{"key":"value"}}'

    mock_run = Mock()
    monkeypatch.setattr("hayhooks.cli.pipeline.run_pipeline_with_files", mock_run)

    result = runner.invoke(hayhooks_cli, ["pipeline", "run", "test_pipeline", "--param", f"complex={complex_json}"])

    assert result.exit_code == 0
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["pipeline_name"] == "test_pipeline"
    assert kwargs["params"]["complex"] == {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}


def test_run_with_nonexistent_file(runner, tmp_path):
    nonexistent_file = "nonexistent.txt"

    result = runner.invoke(hayhooks_cli, ["pipeline", "run", "test_pipeline", "--file", str(nonexistent_file)])

    assert result.exit_code != 0
    assert "does not exist" in result.stdout


def test_run_with_directory(runner, tmp_path, monkeypatch):
    files = create_test_files(
        tmp_path,
        {
            "test_dir": None,  # None indicates directory
            "test_dir/file1.txt": "file1 content",
            "test_dir/file2.txt": "file2 content",
            "test_dir/.hidden": "hidden content",
        },
    )

    mock_run = Mock()
    monkeypatch.setattr("hayhooks.cli.pipeline.run_pipeline_with_files", mock_run)

    result = runner.invoke(hayhooks_cli, ["pipeline", "run", "test_pipeline", "--dir", str(files["test_dir"])])

    assert result.exit_code == 0
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["pipeline_name"] == "test_pipeline"
    assert len(kwargs["files"]) == 2
    assert "file1.txt" in kwargs["files"]
    assert "file2.txt" in kwargs["files"]
    # Hidden file should be excluded
    assert not any(str(path).endswith(".hidden") for path in kwargs["files"].values())


def test_run_with_multiple_sources(runner, tmp_path, monkeypatch):
    files = create_test_files(
        tmp_path,
        {
            "test1.txt": "test1 content",
            "test2.txt": "test2 content",
            "test_dir": None,
            "test_dir/dir_file.txt": "dir file content",
        },
    )

    mock_run = Mock()
    monkeypatch.setattr("hayhooks.cli.pipeline.run_pipeline_with_files", mock_run)

    result = runner.invoke(
        hayhooks_cli,
        [
            "pipeline",
            "run",
            "test_pipeline",
            "--file",
            str(files["test1.txt"]),
            "--file",
            str(files["test2.txt"]),
            "--dir",
            str(files["test_dir"]),
            "--param",
            "key1=value1",
            "--param",
            "key2=[1,2,3]",
        ],
    )

    assert result.exit_code == 0

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["pipeline_name"] == "test_pipeline"
    # Check files (3 total: 2 direct files + 1 from directory)
    assert len(kwargs["files"]) == 3
    # Check params
    assert kwargs["params"] == {"key1": "value1", "key2": [1, 2, 3]}
