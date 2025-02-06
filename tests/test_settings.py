import pytest
import shutil
from pathlib import Path
from hayhooks.settings import AppSettings


@pytest.fixture
def temp_dir(tmp_path):
    yield tmp_path

    if tmp_path.exists():
        shutil.rmtree(tmp_path)


def test_default_pipelines_dir():
    settings = AppSettings()
    assert settings.pipelines_dir == str(Path(__file__).parent.parent / "pipelines")


def test_custom_pipelines_dir(temp_dir):
    custom_dir = temp_dir / "custom_pipelines"
    settings = AppSettings(pipelines_dir=str(custom_dir))
    assert settings.pipelines_dir == str(custom_dir)


def test_root_path():
    settings = AppSettings(root_path="test_root")
    assert settings.root_path == "test_root"


def test_host():
    settings = AppSettings(host="test_host")
    assert settings.host == "test_host"


def test_port():
    settings = AppSettings(port=1234)
    assert settings.port == 1234
