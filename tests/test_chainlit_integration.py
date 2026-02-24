import logging
import sys
from unittest.mock import MagicMock

import pytest
import uvicorn

import hayhooks.server.app as server_app
import hayhooks.server.utils.chainlit_utils as chainlit_utils_mod
from hayhooks.server.utils.chainlit_utils import (
    DEFAULT_CHAINLIT_APP,
    DEFAULT_CHAINLIT_APP_DIR,
    _merge_custom_elements,
    _seed_public_assets,
    is_chainlit_available,
    mount_chainlit_app,
)
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
        import builtins

        original_import = builtins.__import__

        def _import_mock(name, *args, **kwargs):
            if name == "chainlit":
                msg = "No module named 'chainlit'"
                raise ImportError(msg)
            return original_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "chainlit", raising=False)
        monkeypatch.setattr(builtins, "__import__", _import_mock)
        assert is_chainlit_available() is False


class TestMountChainlitApp:
    def test_raises_import_error_when_chainlit_missing(self, monkeypatch):
        from fastapi import FastAPI

        app = FastAPI()
        monkeypatch.setitem(sys.modules, "chainlit", None)
        monkeypatch.setitem(sys.modules, "chainlit.utils", None)
        with pytest.raises(ImportError, match="hayhooks\\[chainlit\\]"):
            mount_chainlit_app(app)

    def test_raises_file_not_found_for_missing_target(self, mock_chainlit_modules):
        from fastapi import FastAPI

        app = FastAPI()
        with pytest.raises(FileNotFoundError, match="not_a_real_app\\.py"):
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

    def test_mounts_public_directory_when_it_exists(self, mock_chainlit_modules):
        from fastapi import FastAPI

        app = FastAPI()
        mount_chainlit_app(app, path="/chat")

        # Verify that public directory was mounted
        mounted_paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/public" in mounted_paths, "Public directory should be mounted at /public"


class TestCreateAppWithUI:
    def test_ui_not_mounted_by_default(self, monkeypatch):
        from hayhooks.server.app import create_app

        test_settings = AppSettings(pipelines_dir="", chainlit_enabled=False)
        monkeypatch.setattr(server_app, "settings", test_settings)

        app = create_app()
        mounted_paths = [getattr(r, "path", "") for r in app.routes if type(r).__name__ == "Mount"]
        assert "/chat" not in mounted_paths

    def test_logs_warning_when_chainlit_not_installed(self, monkeypatch, caplog):
        from hayhooks.server.app import create_app

        test_settings = AppSettings(pipelines_dir="", chainlit_enabled=True, chainlit_path="/chat")
        monkeypatch.setattr(server_app, "settings", test_settings)
        monkeypatch.setattr(chainlit_utils_mod, "is_chainlit_available", lambda: False)

        create_app()
        assert "not installed" in caplog.text.lower()

    def test_mounts_chainlit_when_enabled_and_available(self, monkeypatch, mock_chainlit_modules):
        from hayhooks.server.app import create_app

        test_settings = AppSettings(pipelines_dir="", chainlit_enabled=True, chainlit_path="/chat")
        monkeypatch.setattr(server_app, "settings", test_settings)
        monkeypatch.setattr(chainlit_utils_mod, "is_chainlit_available", lambda: True)

        create_app()

        mock_chainlit_modules.assert_called_once()
        _, kwargs = mock_chainlit_modules.call_args
        assert kwargs["path"] == "/chat"


class TestCLIUIWarning:
    def test_warns_when_chainlit_path_without_with_chainlit(self, monkeypatch, caplog):
        from hayhooks.cli.base import run

        monkeypatch.setattr(uvicorn, "run", MagicMock())

        with caplog.at_level(logging.WARNING):
            run(with_chainlit=False, chainlit_path="/chat")

        assert "--chainlit-path was provided but --with-chainlit is not set" in caplog.text

    def test_no_warning_when_both_flags_set(self, monkeypatch, caplog):
        from hayhooks.cli.base import run

        monkeypatch.setattr(uvicorn, "run", MagicMock())

        with caplog.at_level(logging.WARNING):
            run(with_chainlit=True, chainlit_path="/chat")

        assert "--chainlit-path was provided but --with-chainlit is not set" not in caplog.text

    def test_warns_when_custom_elements_dir_without_with_chainlit(self, monkeypatch, caplog):
        from hayhooks.cli.base import run

        monkeypatch.setattr(uvicorn, "run", MagicMock())

        with caplog.at_level(logging.WARNING):
            run(with_chainlit=False, chainlit_custom_elements_dir="/some/dir")

        assert "--chainlit-custom-elements-dir was provided but --with-chainlit is not set" in caplog.text

    def test_no_custom_elements_warning_when_with_chainlit_set(self, monkeypatch, caplog):
        from hayhooks.cli.base import run

        monkeypatch.setattr(uvicorn, "run", MagicMock())

        with caplog.at_level(logging.WARNING):
            run(with_chainlit=True, chainlit_custom_elements_dir="/some/dir")

        assert "--chainlit-custom-elements-dir was provided but --with-chainlit is not set" not in caplog.text


class TestSeedPublicAssets:
    def test_seeds_builtin_assets_into_empty_target(self, tmp_path):
        _seed_public_assets(str(tmp_path))

        target_public = tmp_path / "public"
        assert target_public.is_dir()

        builtin_public = DEFAULT_CHAINLIT_APP_DIR / "public"
        builtin_files = {f.name for f in builtin_public.iterdir() if f.is_file()}
        seeded_files = {f.name for f in target_public.iterdir() if f.is_file()}
        assert builtin_files == seeded_files

    def test_does_not_overwrite_existing_files(self, tmp_path):
        target_public = tmp_path / "public"
        target_public.mkdir()
        custom_theme = target_public / "theme.json"
        custom_theme.write_text('{"custom": true}')

        _seed_public_assets(str(tmp_path))

        assert custom_theme.read_text() == '{"custom": true}'

    def test_seeds_missing_files_alongside_existing(self, tmp_path):
        target_public = tmp_path / "public"
        target_public.mkdir()
        (target_public / "theme.json").write_text('{"custom": true}')

        _seed_public_assets(str(tmp_path))

        builtin_public = DEFAULT_CHAINLIT_APP_DIR / "public"
        builtin_files = {f.name for f in builtin_public.iterdir() if f.is_file()}
        seeded_files = {f.name for f in target_public.iterdir() if f.is_file()}
        assert builtin_files == seeded_files
        assert (target_public / "theme.json").read_text() == '{"custom": true}'

    def test_noop_when_builtin_public_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hayhooks.server.utils.chainlit_utils.DEFAULT_CHAINLIT_APP_DIR",
            tmp_path / "nonexistent",
        )
        _seed_public_assets(str(tmp_path))
        assert not (tmp_path / "public").exists()

    def test_creates_public_dir_if_absent(self, tmp_path):
        target = tmp_path / "app_root"
        target.mkdir()
        assert not (target / "public").exists()

        _seed_public_assets(str(target))

        assert (target / "public").is_dir()


class TestMergeCustomElements:
    def test_noop_when_setting_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("hayhooks.server.utils.chainlit_utils.settings.chainlit_custom_elements_dir", "")
        _merge_custom_elements(str(tmp_path))
        assert not (tmp_path / "public" / "elements").exists()

    def test_warns_when_dir_does_not_exist(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(
            "hayhooks.server.utils.chainlit_utils.settings.chainlit_custom_elements_dir",
            str(tmp_path / "nope"),
        )
        with caplog.at_level(logging.WARNING):
            _merge_custom_elements(str(tmp_path))
        assert "is not a directory" in caplog.text

    def test_warns_when_no_jsx_files(self, tmp_path, monkeypatch, caplog):
        custom_dir = tmp_path / "elements_src"
        custom_dir.mkdir()
        (custom_dir / "readme.txt").write_text("not jsx")
        monkeypatch.setattr(
            "hayhooks.server.utils.chainlit_utils.settings.chainlit_custom_elements_dir",
            str(custom_dir),
        )
        with caplog.at_level(logging.WARNING):
            _merge_custom_elements(str(tmp_path))
        assert "No .jsx files found" in caplog.text

    def test_copies_jsx_files(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "elements_src"
        custom_dir.mkdir()
        (custom_dir / "MyWidget.jsx").write_text("export default function MyWidget() {}")
        (custom_dir / "Other.jsx").write_text("export default function Other() {}")
        (custom_dir / "ignored.txt").write_text("not a jsx file")

        monkeypatch.setattr(
            "hayhooks.server.utils.chainlit_utils.settings.chainlit_custom_elements_dir",
            str(custom_dir),
        )

        app_root = tmp_path / "app"
        app_root.mkdir()
        _merge_custom_elements(str(app_root))

        elements_dir = app_root / "public" / "elements"
        assert (elements_dir / "MyWidget.jsx").exists()
        assert (elements_dir / "Other.jsx").exists()
        assert not (elements_dir / "ignored.txt").exists()

    def test_warns_on_override(self, tmp_path, monkeypatch, caplog):
        custom_dir = tmp_path / "elements_src"
        custom_dir.mkdir()
        (custom_dir / "Existing.jsx").write_text("new version")

        app_root = tmp_path / "app"
        elements_dir = app_root / "public" / "elements"
        elements_dir.mkdir(parents=True)
        (elements_dir / "Existing.jsx").write_text("old version")

        monkeypatch.setattr(
            "hayhooks.server.utils.chainlit_utils.settings.chainlit_custom_elements_dir",
            str(custom_dir),
        )
        with caplog.at_level(logging.WARNING):
            _merge_custom_elements(str(app_root))

        assert "overrides built-in element" in caplog.text
        assert (elements_dir / "Existing.jsx").read_text() == "new version"
