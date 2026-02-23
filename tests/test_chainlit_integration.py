import logging
import sys
from unittest.mock import MagicMock

import pytest
import uvicorn

import hayhooks.server.app as server_app
import hayhooks.server.utils.chainlit_utils as chainlit_utils_mod
from hayhooks.server.utils.chainlit_utils import DEFAULT_CHAINLIT_APP, is_chainlit_available, mount_chainlit_app
from hayhooks.settings import AppSettings


@pytest.fixture
def mock_chainlit_modules(monkeypatch):
    """Inject fake chainlit modules into sys.modules so local imports succeed."""
    mock_mount = MagicMock()
    fake_chainlit = MagicMock()
    fake_chainlit_utils = MagicMock()
    fake_chainlit_utils.mount_chainlit = mock_mount

    monkeypatch.setitem(sys.modules, "chainlit", fake_chainlit)
    monkeypatch.setitem(sys.modules, "chainlit.utils", fake_chainlit_utils)
    return mock_mount


class TestIsChainlitAvailable:
    def test_returns_true_when_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "chainlit", MagicMock())
        assert is_chainlit_available() is True

    def test_returns_false_when_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "chainlit", None)
        assert is_chainlit_available() is False


class TestMountChainlitApp:
    def test_raises_import_error_when_chainlit_missing(self, monkeypatch):
        from fastapi import FastAPI

        app = FastAPI()
        monkeypatch.setitem(sys.modules, "chainlit", None)
        monkeypatch.setitem(sys.modules, "chainlit.utils", None)
        with pytest.raises(ImportError, match="hayhooks\\[ui\\]"):
            mount_chainlit_app(app)

    def test_raises_file_not_found_for_missing_target(self, mock_chainlit_modules):
        from fastapi import FastAPI

        app = FastAPI()
        with pytest.raises(FileNotFoundError, match="not\\_a\\_real\\_app\\_\\.py"):
            mount_chainlit_app(app, target="/nonexistent/not_a_real_app.py")

    def test_uses_default_app_when_no_target(self, mock_chainlit_modules):
        from fastapi import FastAPI

        app = FastAPI()
        mount_chainlit_app(app, path="/chat")

        mock_chainlit_modules.assert_called_once_with(
            app=app,
            target=str(DEFAULT_CHAINLIT_APP),
            path="/chat",
        )

    def test_uses_custom_target(self, tmp_path, mock_chainlit_modules):
        from fastapi import FastAPI

        app = FastAPI()
        custom_app = tmp_path / "my_app.py"
        custom_app.write_text("# custom chainlit app")

        mount_chainlit_app(app, target=str(custom_app), path="/ui")

        mock_chainlit_modules.assert_called_once_with(
            app=app,
            target=str(custom_app),
            path="/ui",
        )

    def test_default_chainlit_app_file_exists(self):
        assert DEFAULT_CHAINLIT_APP.exists(), f"Default Chainlit app not found at {DEFAULT_CHAINLIT_APP}"


class TestCreateAppWithUI:
    def test_ui_not_mounted_by_default(self, monkeypatch):
        from hayhooks.server.app import create_app

        test_settings = AppSettings(pipelines_dir="", ui_enabled=False)
        monkeypatch.setattr(server_app, "settings", test_settings)

        app = create_app()
        mounted_paths = [getattr(r, "path", "") for r in app.routes if type(r).__name__ == "Mount"]
        assert "/chat" not in mounted_paths

    def test_logs_warning_when_chainlit_not_installed(self, monkeypatch, caplog):
        from hayhooks.server.app import create_app

        test_settings = AppSettings(pipelines_dir="", ui_enabled=True, ui_path="/chat")
        monkeypatch.setattr(server_app, "settings", test_settings)
        monkeypatch.setattr(chainlit_utils_mod, "is_chainlit_available", lambda: False)

        create_app()
        assert "not installed" in caplog.text.lower()

    def test_mounts_chainlit_when_enabled_and_available(self, monkeypatch, mock_chainlit_modules):
        from hayhooks.server.app import create_app

        test_settings = AppSettings(pipelines_dir="", ui_enabled=True, ui_path="/chat")
        monkeypatch.setattr(server_app, "settings", test_settings)
        monkeypatch.setattr(chainlit_utils_mod, "is_chainlit_available", lambda: True)

        create_app()

        mock_chainlit_modules.assert_called_once()
        _, kwargs = mock_chainlit_modules.call_args
        assert kwargs["path"] == "/chat"


class TestCLIUIWarning:
    def test_warns_when_ui_path_without_with_ui(self, monkeypatch, caplog):
        from hayhooks.cli.base import run

        monkeypatch.setattr(uvicorn, "run", MagicMock())

        with caplog.at_level(logging.WARNING):
            run(with_ui=False, ui_path="/chat")

        assert "--ui-path was provided but --with-ui is not set" in caplog.text

    def test_no_warning_when_both_flags_set(self, monkeypatch, caplog):
        from hayhooks.cli.base import run

        monkeypatch.setattr(uvicorn, "run", MagicMock())

        with caplog.at_level(logging.WARNING):
            run(with_ui=True, ui_path="/chat")

        assert "--ui-path was provided but --with-ui is not set" not in caplog.text
