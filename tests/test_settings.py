import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hayhooks.server.app import create_app
from hayhooks.settings import AppSettings, check_cors_settings


@pytest.fixture
def temp_dir(tmp_path):
    yield tmp_path

    if tmp_path.exists():
        shutil.rmtree(tmp_path)


def test_custom_pipelines_dir(temp_dir):
    custom_dir = temp_dir / "custom_pipelines"
    settings = AppSettings(pipelines_dir=str(custom_dir))
    assert settings.pipelines_dir == str(custom_dir)


def test_default_pipelines_dir(monkeypatch):
    monkeypatch.delenv("HAYHOOKS_PIPELINES_DIR", raising=False)
    settings = AppSettings()
    assert settings.pipelines_dir == str(Path.cwd() / "pipelines")


def test_root_path():
    settings = AppSettings(root_path="test_root")
    assert settings.root_path == "test_root"


def test_host():
    settings = AppSettings(host="test_host")
    assert settings.host == "test_host"


def test_port():
    settings = AppSettings(port=1234)
    assert settings.port == 1234


def test_env_var_prefix(monkeypatch):
    monkeypatch.setenv("HAYHOOKS_PORT", "5678")
    settings = AppSettings()
    assert settings.port == 5678


def test_cors():
    default_settings = AppSettings()
    assert default_settings.cors_allow_origins == ["*"]
    assert default_settings.cors_allow_methods == ["*"]
    assert default_settings.cors_allow_headers == ["*"]
    assert default_settings.cors_allow_credentials is False
    assert default_settings.cors_allow_origin_regex is None
    assert default_settings.cors_expose_headers == []
    assert default_settings.cors_max_age == 600

    custom_settings = AppSettings(
        cors_allow_origins=["https://example.com", "https://test.com"],
        cors_allow_methods=["GET", "POST"],
        cors_allow_headers=["X-Custom-Header"],
        cors_allow_credentials=True,
        cors_allow_origin_regex="https://.*\\.example\\.com",
        cors_expose_headers=["X-Custom-Expose"],
        cors_max_age=3600,
    )
    assert custom_settings.cors_allow_origins == ["https://example.com", "https://test.com"]
    assert custom_settings.cors_allow_methods == ["GET", "POST"]
    assert custom_settings.cors_allow_headers == ["X-Custom-Header"]
    assert custom_settings.cors_allow_credentials is True
    assert custom_settings.cors_allow_origin_regex == "https://.*\\.example\\.com"
    assert custom_settings.cors_expose_headers == ["X-Custom-Expose"]
    assert custom_settings.cors_max_age == 3600


def test_cors_env_vars(monkeypatch):
    monkeypatch.setenv("HAYHOOKS_CORS_ALLOW_ORIGINS", '["https://example.com"]')
    monkeypatch.setenv("HAYHOOKS_CORS_ALLOW_METHODS", '["GET", "POST"]')
    monkeypatch.setenv("HAYHOOKS_CORS_ALLOW_HEADERS", '["X-Test-Header"]')
    monkeypatch.setenv("HAYHOOKS_CORS_ALLOW_CREDENTIALS", "true")
    monkeypatch.setenv("HAYHOOKS_CORS_ALLOW_ORIGIN_REGEX", "https://.*\\.test\\.com")
    monkeypatch.setenv("HAYHOOKS_CORS_EXPOSE_HEADERS", '["X-Expose-Test"]')
    monkeypatch.setenv("HAYHOOKS_CORS_MAX_AGE", "1800")

    settings = AppSettings()
    assert settings.cors_allow_origins == ["https://example.com"]
    assert settings.cors_allow_methods == ["GET", "POST"]
    assert settings.cors_allow_headers == ["X-Test-Header"]
    assert settings.cors_allow_credentials is True
    assert settings.cors_allow_origin_regex == "https://.*\\.test\\.com"
    assert settings.cors_expose_headers == ["X-Expose-Test"]
    assert settings.cors_max_age == 1800


def test_cors_warning():
    with patch("hayhooks.server.logger.log.warning") as mock_log_warning:
        check_cors_settings()
        mock_log_warning.assert_called_once_with(
            "Using default CORS settings - All origins, methods, and headers are allowed."
        )

    with patch("hayhooks.server.logger.log.warning") as mock_log_warning:
        AppSettings(
            cors_allow_origins=["https://example.com"],
            cors_allow_methods=["GET", "POST"],
            cors_allow_headers=["X-Custom-Header"],
        )
        mock_log_warning.assert_not_called()


def test_additional_python_path():
    custom_path = "/custom/python/path"
    settings = AppSettings(additional_python_path=custom_path)
    assert settings.additional_python_path == custom_path


def test_additional_python_path_env_var(monkeypatch):
    custom_path = "/env/var/path"
    monkeypatch.setenv("HAYHOOKS_ADDITIONAL_PYTHON_PATH", custom_path)
    settings = AppSettings()
    assert settings.additional_python_path == custom_path


def test_additional_python_path_in_sys_path(test_settings):
    original_sys_path = sys.path.copy()

    try:
        # Add a test path directly to the settings object
        test_path = "/test/python/path"
        test_settings.additional_python_path = test_path

        # Create the app which should add the path to sys.path
        # And verify the path was added
        create_app()
        assert test_path in sys.path

    finally:
        # Restore original sys.path
        sys.path = original_sys_path


def test_additional_python_path_in_sys_path_via_env(monkeypatch, test_settings):
    original_sys_path = sys.path.copy()

    try:
        # Set a test path via environment variable
        test_path = "/test/python/path"
        monkeypatch.setenv("HAYHOOKS_ADDITIONAL_PYTHON_PATH", test_path)

        # Reimport create_app after monkeypatch.setenv
        from hayhooks.server.app import create_app

        # Create the app which should add the path to sys.path
        # And verify the path was added
        create_app()
        assert test_path in sys.path

    finally:
        # Restore original sys.path
        sys.path = original_sys_path
