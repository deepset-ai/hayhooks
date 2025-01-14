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

    assert Path(settings.pipelines_dir).exists()
    assert Path(settings.pipelines_dir).is_dir()

    shutil.rmtree(settings.pipelines_dir)


def test_custom_pipelines_dir(temp_dir):
    custom_dir = temp_dir / "custom_pipelines"

    settings = AppSettings(pipelines_dir=str(custom_dir))

    assert Path(settings.pipelines_dir).exists()
    assert Path(settings.pipelines_dir).is_dir()


def test_invalid_pipelines_dir(temp_dir):
    invalid_path = temp_dir / "not_a_dir"
    invalid_path.touch()

    with pytest.raises(ValueError) as exc_info:
        AppSettings(pipelines_dir=str(invalid_path))

    assert "exists but is not a directory" in str(exc_info.value)


def test_if_pipelines_dir_does_not_exist_creates_it(temp_dir):
    non_existing_path = temp_dir / "non_existing_dir"

    settings = AppSettings(pipelines_dir=str(non_existing_path))

    assert Path(settings.pipelines_dir).exists()
    assert Path(settings.pipelines_dir).is_dir()


def test_root_path():
    settings = AppSettings(root_path="test_root")
    assert settings.root_path == "test_root"


def test_host():
    settings = AppSettings(host="test_host")
    assert settings.host == "test_host"


def test_port():
    settings = AppSettings(port=1234)
    assert settings.port == 1234
