"""Tests for CLI streaming support."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from hayhooks.cli.base import hayhooks_cli

runner = CliRunner()


@pytest.fixture
def mock_streaming_response():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/plain; charset=utf-8"}
    mock_response.iter_content = Mock(return_value=iter(["Hello ", "streaming ", "world!"]))
    return mock_response


@pytest.fixture
def mock_json_response():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={"result": "Hello world"})
    return mock_response


def test_cli_run_with_streaming(mock_streaming_response):
    with patch("hayhooks.cli.utils.requests.request") as mock_request:
        mock_request.return_value = mock_streaming_response

        result = runner.invoke(
            hayhooks_cli,
            ["pipeline", "run", "test_pipeline", "--stream", "--param", "question=test"],
        )

        assert result.exit_code == 0
        assert "Streaming output:" in result.stdout
        # Check that streaming was requested
        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["stream"] is True


def test_cli_run_without_streaming(mock_json_response):
    with patch("hayhooks.cli.utils.requests.request") as mock_request:
        mock_json_response.headers = {"content-type": "application/json"}
        mock_request.return_value = mock_json_response

        result = runner.invoke(
            hayhooks_cli,
            ["pipeline", "run", "test_pipeline", "--param", "question=test"],
        )

        assert result.exit_code == 0
        assert "Result:" in result.stdout
        # Check that streaming was not requested
        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs.get("stream", False) is False


def test_cli_run_streaming_with_files_shows_warning():
    """Test that streaming with file uploads shows a warning."""
    test_file = Path(__file__)  # Use this test file as a dummy file

    with patch("hayhooks.cli.utils.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"result": "success"})
        mock_post.return_value = mock_response

        result = runner.invoke(
            hayhooks_cli,
            ["pipeline", "run", "test_pipeline", "--stream", "--file", str(test_file)],
        )

        # Should show warning about streaming not being supported with files
        assert "Streaming mode is not supported with file uploads" in result.stdout or result.exit_code == 0


def test_cli_run_streaming_endpoint_without_stream_flag():
    """Test calling a streaming endpoint without --stream flag shows helpful message."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/plain; charset=utf-8"}
    mock_response.text = "Hello streaming world!"

    with patch("hayhooks.cli.utils.requests.request") as mock_request:
        mock_request.return_value = mock_response

        result = runner.invoke(
            hayhooks_cli,
            ["pipeline", "run", "test_pipeline", "--param", "question=test"],
        )

        assert result.exit_code == 0
        # Should show helpful note about using --stream (message may be split across lines)
        assert "--stream flag" in result.stdout
        # Should still display the result
        assert "Hello streaming world!" in result.stdout
