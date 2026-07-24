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


def test_durable_redis_settings_defaults(monkeypatch):
    names = (
        "HAYHOOKS_DURABLE_REDIS_CLAIM_IDLE_MS",
        "HAYHOOKS_DURABLE_REDIS_QUEUE_BLOCK_MS",
        "HAYHOOKS_DURABLE_REDIS_RECLAIM_INTERVAL",
        "HAYHOOKS_DURABLE_REDIS_CANCELLATION_TTL_SECONDS",
        "HAYHOOKS_DURABLE_REDIS_STREAM_MAX_LENGTH",
        "HAYHOOKS_DURABLE_REDIS_DELAYED_PROMOTION_INTERVAL",
        "HAYHOOKS_DURABLE_REDIS_DELAYED_PROMOTION_BATCH_SIZE",
        "HAYHOOKS_DURABLE_REDIS_SOCKET_TIMEOUT",
        "HAYHOOKS_DURABLE_REDIS_SOCKET_CONNECT_TIMEOUT",
        "HAYHOOKS_DURABLE_REDIS_HEALTH_CHECK_INTERVAL",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    settings = AppSettings()

    assert settings.durable_redis_claim_idle_ms == 30_000
    assert settings.durable_redis_queue_block_ms == 1_000
    assert settings.durable_redis_reclaim_interval == 1.0
    assert settings.durable_redis_cancellation_ttl_seconds == 86_400
    assert settings.durable_redis_stream_max_length == 0
    assert settings.durable_redis_delayed_promotion_interval == 0.25
    assert settings.durable_redis_delayed_promotion_batch_size == 100
    assert settings.durable_redis_socket_timeout == 5.0
    assert settings.durable_redis_socket_connect_timeout == 5.0
    assert settings.durable_redis_health_check_interval == 30


def test_durable_redis_settings_from_environment(monkeypatch):
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_CLAIM_IDLE_MS", "45000")
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_QUEUE_BLOCK_MS", "2500")
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_RECLAIM_INTERVAL", "2.5")
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_CANCELLATION_TTL_SECONDS", "120")
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_STREAM_MAX_LENGTH", "2500")
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_DELAYED_PROMOTION_INTERVAL", "0.75")
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_DELAYED_PROMOTION_BATCH_SIZE", "250")
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_SOCKET_TIMEOUT", "3.5")
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_SOCKET_CONNECT_TIMEOUT", "2.5")
    monkeypatch.setenv("HAYHOOKS_DURABLE_REDIS_HEALTH_CHECK_INTERVAL", "20")

    settings = AppSettings()

    assert settings.durable_redis_claim_idle_ms == 45_000
    assert settings.durable_redis_queue_block_ms == 2_500
    assert settings.durable_redis_reclaim_interval == 2.5
    assert settings.durable_redis_cancellation_ttl_seconds == 120
    assert settings.durable_redis_stream_max_length == 2_500
    assert settings.durable_redis_delayed_promotion_interval == 0.75
    assert settings.durable_redis_delayed_promotion_batch_size == 250
    assert settings.durable_redis_socket_timeout == 3.5
    assert settings.durable_redis_socket_connect_timeout == 2.5
    assert settings.durable_redis_health_check_interval == 20


def test_cors():
    default_settings = AppSettings()
    assert default_settings.cors_allow_origins == ["*"]
    assert default_settings.cors_allow_methods == ["*"]
    assert default_settings.cors_allow_headers == ["*"]
    assert default_settings.cors_allow_credentials is False
    assert default_settings.cors_allow_origin_regex is None
    assert default_settings.cors_expose_headers == ["X-Hayhooks-Trace-Cursor"]
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


def test_access_log_excluded_path_prefixes_default():
    settings = AppSettings()
    assert settings.access_log_excluded_path_prefixes == [
        "/dashboard/api/config",
        "/dashboard/api/entrypoints",
        "/dashboard/api/traces",
    ]


def test_access_log_excluded_path_prefixes_env_var(monkeypatch):
    monkeypatch.setenv("HAYHOOKS_ACCESS_LOG_EXCLUDED_PATH_PREFIXES", '["/status", "/metrics"]')
    settings = AppSettings()
    assert settings.access_log_excluded_path_prefixes == ["/status", "/metrics"]


def test_dashboard_trace_include_haystack_spans_default():
    settings = AppSettings()
    assert settings.dashboard_trace_include_haystack_spans is True


def test_dashboard_trace_include_haystack_spans_env_var(monkeypatch):
    monkeypatch.setenv("HAYHOOKS_DASHBOARD_TRACE_INCLUDE_HAYSTACK_SPANS", "true")
    settings = AppSettings()
    assert settings.dashboard_trace_include_haystack_spans is True


def test_dashboard_trace_buffer_capacity_default():
    settings = AppSettings()
    assert settings.dashboard_trace_buffer_capacity == 200


def test_dashboard_trace_buffer_capacity_env_var(monkeypatch):
    monkeypatch.setenv("HAYHOOKS_DASHBOARD_TRACE_BUFFER_CAPACITY", "2000")
    settings = AppSettings()
    assert settings.dashboard_trace_buffer_capacity == 2000


def test_dashboard_ui_slow_component_min_duration_ms_default():
    settings = AppSettings()
    assert settings.dashboard_ui_slow_component_min_duration_ms == 1000


def test_dashboard_ui_slow_component_min_duration_ms_env_var(monkeypatch):
    monkeypatch.setenv("HAYHOOKS_DASHBOARD_UI_SLOW_COMPONENT_MIN_DURATION_MS", "2500")
    settings = AppSettings()
    assert settings.dashboard_ui_slow_component_min_duration_ms == 2500
