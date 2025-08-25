from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from requests import ConnectionError, HTTPError  # noqa: A004

from hayhooks.cli.utils import get_server_url, make_request


@pytest.fixture
def mock_requests(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("hayhooks.cli.utils.requests.request", mock)
    return mock


def configure_mock_response(
    mock_requests, *, json_return_value: Optional[dict[str, Any]] = None, raise_return_value: Optional[Exception] = None
):
    if json_return_value is None:
        json_return_value = {}
    mock_response = MagicMock()
    mock_response.json.return_value = json_return_value

    if isinstance(raise_return_value, Exception):
        mock_response.raise_for_status.side_effect = raise_return_value
    else:
        mock_response.raise_for_status.return_value = raise_return_value
    mock_requests.return_value = mock_response

    return mock_response


def test_get_server_url():
    assert get_server_url(host="localhost", port=8000) == "http://localhost:8000"
    assert get_server_url(host="localhost", port=8000, https=False) == "http://localhost:8000"
    assert get_server_url(host="localhost", port=8000, https=True) == "https://localhost:8000"


def test_make_request_http_get_success(mock_requests):
    configure_mock_response(mock_requests, json_return_value={"status": "ok"})

    response = make_request(host="localhost", port=8000, endpoint="/test_endpoint")

    mock_requests.assert_called_once_with(
        method="GET",
        url="http://localhost:8000/test_endpoint",
        json=None,
        verify=True,
    )
    assert response == {"status": "ok"}


def test_make_request_https_success(mock_requests):
    configure_mock_response(mock_requests, json_return_value={"status": "ok_https"})

    response = make_request(host="localhost", port=8000, endpoint="/secure_endpoint", use_https=True)

    mock_requests.assert_called_once_with(
        method="GET",
        url="https://localhost:8000/secure_endpoint",
        json=None,
        verify=True,
    )
    assert response == {"status": "ok_https"}


def test_make_request_https_disable_ssl_verification(mock_requests):
    configure_mock_response(mock_requests, json_return_value={"status": "ok_https_noverify"})

    response = make_request(
        host="localhost", port=8000, endpoint="/secure_endpoint_noverify", use_https=True, disable_ssl=True
    )

    mock_requests.assert_called_once_with(
        method="GET",
        url="https://localhost:8000/secure_endpoint_noverify",
        json=None,
        verify=False,
    )
    assert response == {"status": "ok_https_noverify"}


def test_make_request_connection_error(mock_requests, capsys):
    mock_requests.side_effect = ConnectionError("Test connection error")

    with pytest.raises(Exception):
        make_request(host="localhost", port=8000, endpoint="/test_endpoint")

    mock_requests.assert_called_once_with(
        method="GET", url="http://localhost:8000/test_endpoint", json=None, verify=True
    )
    captured = capsys.readouterr()
    assert "Hayhooks server is not responding." in captured.out
    assert "To start one, run `hayhooks run`" in captured.out


def test_make_request_http_error_with_detail(mock_requests, capsys):
    http_error = HTTPError("404 Client Error")
    mock_response = configure_mock_response(
        mock_requests, json_return_value={"detail": "Item not found"}, raise_return_value=http_error
    )
    http_error.response = mock_response

    with pytest.raises(Exception):
        make_request(host="localhost", port=8000, endpoint="/notfound")

    mock_requests.assert_called_once_with(method="GET", url="http://localhost:8000/notfound", json=None, verify=True)
    captured = capsys.readouterr()
    assert "Server error" in captured.out
    assert "Item not found" in captured.out


def test_make_request_http_error_no_detail(mock_requests, capsys):
    http_error = HTTPError("500 Server Error")
    mock_response = configure_mock_response(mock_requests, json_return_value={}, raise_return_value=http_error)
    http_error.response = mock_response

    with pytest.raises(Exception):
        make_request(host="localhost", port=8000, endpoint="/servererror")

    mock_requests.assert_called_once_with(method="GET", url="http://localhost:8000/servererror", json=None, verify=True)
    captured = capsys.readouterr()
    assert "Server error" in captured.out
    assert "Unknown error" in captured.out  # Default message


def test_make_request_unexpected_error(mock_requests, capsys):
    mock_requests.side_effect = Exception("Something totally unexpected")

    with pytest.raises(Exception):
        make_request(host="localhost", port=8000, endpoint="/test_endpoint")

    mock_requests.assert_called_once_with(
        method="GET", url="http://localhost:8000/test_endpoint", json=None, verify=True
    )
    captured = capsys.readouterr()
    assert "Unexpected error" in captured.out
    assert "Something totally unexpected" in captured.out
