"""Unsafe uploads must be rejected before touching existing or unrelated data."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hayhooks.server.app import create_app
from hayhooks.server.exceptions import PipelinePathError
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils import deploy_utils
from hayhooks.server.utils.pipeline_paths import validate_pipeline_name


@pytest.fixture
def paths(tmp_path, test_settings, monkeypatch):
    root = tmp_path / "app" / "pipelines"
    root.mkdir(parents=True)
    outside = root.parent / "customer-data"
    outside.mkdir()
    important = outside / "important.txt"
    important.write_text("keep me")
    monkeypatch.setattr(test_settings, "pipelines_dir", str(root))
    registry.clear()
    yield root, outside, important
    registry.clear()


@pytest.mark.parametrize(
    "name", ["", ".", "..", "/app/customer-data", "../customer-data", "a/b", "a\\b", "C:\\data", "C:data", "\0"]
)
def test_unsafe_pipeline_names(name):
    with pytest.raises(PipelinePathError):
        validate_pipeline_name(name)


@pytest.mark.parametrize("endpoint", ["/deploy-yaml", "/deploy_files"])
@pytest.mark.parametrize("name", ["", ".", "..", "../customer-data", "nested/name", "nested\\name", "absolute"])
def test_http_rejects_unsafe_names_before_deployment(paths, endpoint, name):
    root, outside, important = paths
    if name == "absolute":
        name = str(outside)
    body = (
        {"source_code": "not: valid", "save_file": True}
        if endpoint == "/deploy-yaml"
        else {
            "files": {"pipeline_wrapper.py": "not valid Python"},
            "save_files": True,
        }
    )
    with TestClient(create_app()) as client:
        response = client.post(endpoint, json={"name": name, "overwrite": True, **body})
    assert response.status_code == 422
    assert important.read_text() == "keep me"
    assert list(root.iterdir()) == []
    assert registry.get_names() == []


@pytest.mark.parametrize("operation", ["yaml", "files", "save", "remove", "backup", "undeploy"])
def test_programmatic_absolute_name_cannot_touch_customer_data(paths, operation):
    root, outside, important = paths
    name = str(outside)
    with pytest.raises(PipelinePathError):
        if operation == "yaml":
            deploy_utils.deploy_pipeline_yaml(name, "not: valid", overwrite=True, options={"save_file": True})
        elif operation == "files":
            deploy_utils.deploy_pipeline_files(name, {"pipeline_wrapper.py": "invalid"}, overwrite=True)
        elif operation == "save":
            deploy_utils.save_pipeline_files(name, {"file.txt": "invalid"}, str(root))
        elif operation == "remove":
            deploy_utils.remove_pipeline_files(name, str(root))
        elif operation == "backup":
            deploy_utils._backup_pipeline_files(name)
        else:
            deploy_utils.undeploy_pipeline(name)
    assert important.read_text() == "keep me"
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "key",
    [
        "",
        ".",
        "..",
        "../important.txt",
        "nested/../../important.txt",
        "/tmp/file",
        "C:\\file",
        "C:file",
        "nested\\file",
        "file\0",
    ],
)
@pytest.mark.parametrize("save_files", [True, False])
def test_invalid_file_keys_are_rejected_before_overwrite(paths, key, save_files):
    root, _, important = paths
    existing = root / "demo"
    existing.mkdir()
    source = existing / "pipeline_wrapper.py"
    source.write_text("original")
    files = {"pipeline_wrapper.py": "candidate", key: "overwrite"}
    with pytest.raises(PipelinePathError):
        deploy_utils.deploy_pipeline_files("demo", files, overwrite=True, save_files=save_files)
    assert source.read_text() == "original"
    assert important.read_text() == "keep me"
    assert sorted(p.name for p in root.iterdir()) == ["demo"]


def test_file_upload_validation_returns_422(paths):
    with TestClient(create_app()) as client:
        response = client.post(
            "/deploy_files",
            json={
                "name": "demo",
                "files": {"pipeline_wrapper.py": "invalid", "../outside.txt": "overwrite"},
                "overwrite": True,
            },
        )
    assert response.status_code == 422
    assert list(paths[0].iterdir()) == []


def test_nested_files_are_supported(paths):
    root, _, _ = paths
    files = {"pipeline_wrapper.py": "source", "helpers/__init__.py": "", "data/prompts/system.txt": "hello"}
    saved = deploy_utils.save_pipeline_files("demo", files, str(root))
    assert {key: Path(path).read_text() for key, path in saved.items()} == files


@pytest.mark.parametrize("source", ["directory", "yml", "yaml"])
@pytest.mark.parametrize("operation", ["save", "remove", "backup", "yaml", "files"])
def test_symlinked_sources_cannot_escape_pipelines_directory(paths, source, operation):
    root, outside, important = paths
    if source == "directory":
        (root / "demo").symlink_to(outside, target_is_directory=True)
    else:
        (root / f"demo.{source}").symlink_to(important)
    with pytest.raises(PipelinePathError):
        if operation == "save":
            files = {"demo.yml": "new"} if source != "directory" else {"important.txt": "new"}
            deploy_utils.save_pipeline_files("demo", files, str(root))
        elif operation == "remove":
            deploy_utils.remove_pipeline_files("demo", str(root))
        elif operation == "backup":
            deploy_utils._backup_pipeline_files("demo")
        elif operation == "yaml":
            deploy_utils.deploy_pipeline_yaml("demo", "not: valid", overwrite=True)
        else:
            deploy_utils.deploy_pipeline_files("demo", {"pipeline_wrapper.py": "invalid"}, overwrite=True)
    assert important.read_text() == "keep me"
    assert len(list(root.iterdir())) == 1


@pytest.mark.parametrize("link_kind", ["file", "directory", "dangling"])
def test_nested_symlinks_are_checked_before_any_write_or_backup(paths, link_kind):
    root, outside, important = paths
    pipeline_dir = root / "demo"
    pipeline_dir.mkdir()
    original = pipeline_dir / "pipeline_wrapper.py"
    original.write_text("original")
    if link_kind == "directory":
        (pipeline_dir / "data").symlink_to(outside, target_is_directory=True)
        key = "data/important.txt"
    else:
        (pipeline_dir / "data.txt").symlink_to(important if link_kind == "file" else outside / "missing.txt")
        key = "data.txt"
    files = {"pipeline_wrapper.py": "candidate", key: "overwrite"}
    for operation in (
        lambda: deploy_utils.save_pipeline_files("demo", files, str(root)),
        lambda: deploy_utils.deploy_pipeline_files("demo", files, overwrite=True),
    ):
        with pytest.raises(PipelinePathError):
            operation()
    assert original.read_text() == "original"
    assert important.read_text() == "keep me"
    assert not (outside / "missing.txt").exists()
    assert len(list(root.iterdir())) == 1


def test_undeploy_rejects_nested_escape_before_removing_live_pipeline(paths):
    root, outside, important = paths
    sample = (Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml").read_text()
    deploy_utils.deploy_pipeline_yaml("demo", sample, options={"save_file": False})
    wrapper = registry.get("demo")
    pipeline_dir = root / "demo"
    pipeline_dir.mkdir()
    (pipeline_dir / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PipelinePathError):
        deploy_utils.undeploy_pipeline("demo")
    assert registry.get("demo") is wrapper
    assert (pipeline_dir / "data").is_symlink()
    assert important.read_text() == "keep me"
