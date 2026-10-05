"""A failed overwrite restores the pipeline it replaced, or reports that it could not."""

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hayhooks.server.app import create_app
from hayhooks.server.exceptions import PipelineRollbackError
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils import deploy_utils
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.models import PreparedPipeline
from hayhooks.server.utils.module_loader import create_pipeline_wrapper_instance

SOURCE = """
from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    def run_api(self, value: int) -> int:
        \"\"\"Scale a value.\"\"\"
        return value * {factor}
"""
OLD_SOURCE, CANDIDATE_SOURCE = SOURCE.format(factor=2), SOURCE.format(factor=3)
SAMPLE_YAML = (Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml").read_text()


class CandidateWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    def run_api(self, value: int) -> int:
        return value * 3


@pytest.fixture
def pipelines_dir(test_settings) -> Path:
    registry.clear()
    shutil.rmtree(test_settings.pipelines_dir, ignore_errors=True)
    yield Path(test_settings.pipelines_dir)
    registry.clear()
    deploy_utils.unload_pipeline_modules("demo")


@pytest.fixture
def app(pipelines_dir: Path) -> FastAPI:
    app = create_app()
    deploy_utils.deploy_pipeline_files("demo", {"pipeline_wrapper.py": OLD_SOURCE}, app=app)
    return app


def fail_route_additions(monkeypatch, *messages: str) -> None:
    add_route = deploy_utils.add_pipeline_api_route
    failures = [RuntimeError(message) for message in messages]

    def add_route_or_fail(*args, **kwargs):
        if failures:
            raise failures.pop(0)
        return add_route(*args, **kwargs)

    monkeypatch.setattr(deploy_utils, "add_pipeline_api_route", add_route_or_fail)


def assert_old_pipeline_restored(app: FastAPI, pipelines_dir: Path, old_wrapper: BasePipelineWrapper) -> None:
    assert registry.get("demo") is old_wrapper
    assert (pipelines_dir / "demo" / "pipeline_wrapper.py").read_text() == OLD_SOURCE
    assert not [path.name for path in pipelines_dir.iterdir() if path.name.startswith(".demo-")]
    assert TestClient(app).post("/demo/run", json={"value": 5}).json() == {"result": 10}


@pytest.mark.parametrize("save_files", [True, False])
def test_failed_file_overwrite_restores_the_previous_pipeline(app, pipelines_dir, monkeypatch, save_files) -> None:
    old_wrapper = registry.get("demo")
    old_modules = {name: sys.modules[name] for name in ("demo", "demo.pipeline_wrapper")}
    fail_route_additions(monkeypatch, "candidate route failed")

    # A clean rollback re-raises the original error unchanged.
    with pytest.raises(RuntimeError, match=r"^candidate route failed$"):
        deploy_utils.deploy_pipeline_files(
            "demo", {"pipeline_wrapper.py": CANDIDATE_SOURCE}, app=app, save_files=save_files, overwrite=True
        )

    assert {name: sys.modules[name] for name in old_modules} == old_modules
    assert_old_pipeline_restored(app, pipelines_dir, old_wrapper)


def test_file_cleanup_failure_does_not_block_rollback(app, pipelines_dir, monkeypatch) -> None:
    old_wrapper = registry.get("demo")
    remove_files = deploy_utils.remove_pipeline_files

    def remove_then_fail(pipeline_name: str, directory: str) -> None:
        remove_files(pipeline_name, directory)
        message = "file cleanup failed"
        raise OSError(message)

    monkeypatch.setattr(deploy_utils, "remove_pipeline_files", remove_then_fail)
    fail_route_additions(monkeypatch, "candidate route failed")
    candidate = create_pipeline_wrapper_instance(SimpleNamespace(PipelineWrapper=CandidateWrapper))

    with pytest.raises(RuntimeError, match="candidate route failed"):
        deploy_utils.commit_prepared_pipeline(
            PreparedPipeline("demo", candidate),
            app=app,
            overwrite=True,
            source_files={"pipeline_wrapper.py": CANDIDATE_SOURCE},
        )

    assert_old_pipeline_restored(app, pipelines_dir, old_wrapper)


@pytest.mark.parametrize("source", ["files", "yaml"])
def test_partial_backup_failure_restores_only_moved_sources(app, pipelines_dir, monkeypatch, source) -> None:
    old_wrapper = registry.get("demo")
    old_module = sys.modules["demo.pipeline_wrapper"]
    yaml_file = pipelines_dir / "demo.yml"
    yaml_file.write_text(SAMPLE_YAML)
    replace = Path.replace

    def fail_yaml_backup(path, target):
        if path == yaml_file:
            message = "backup rename failed"
            raise OSError(message)
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_yaml_backup)
    with pytest.raises(OSError, match="backup rename failed"):
        if source == "files":
            deploy_utils.deploy_pipeline_files(
                "demo", {"pipeline_wrapper.py": CANDIDATE_SOURCE}, app=app, overwrite=True
            )
        else:
            deploy_utils.deploy_pipeline_yaml("demo", SAMPLE_YAML, app=app, overwrite=True)

    assert sys.modules["demo.pipeline_wrapper"] is old_module
    assert yaml_file.read_text() == SAMPLE_YAML
    assert_old_pipeline_restored(app, pipelines_dir, old_wrapper)


def test_failed_restore_raises_rollback_error_chained_from_the_original(app, monkeypatch) -> None:
    fail_route_additions(monkeypatch, "candidate route failed", "restore route failed")

    with pytest.raises(PipelineRollbackError, match="restore route failed") as error:
        deploy_utils.deploy_pipeline_files("demo", {"pipeline_wrapper.py": CANDIDATE_SOURCE}, app=app, overwrite=True)

    assert str(error.value.__cause__) == "candidate route failed"


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
        pytest.param("/deploy_files", {"files": {"pipeline_wrapper.py": OLD_SOURCE}}, id="files"),
        pytest.param("/deploy-yaml", {"source_code": SAMPLE_YAML}, id="yaml"),
    ],
)
def test_http_failed_rollback_is_a_server_error_with_both_messages(pipelines_dir, monkeypatch, endpoint, body) -> None:
    with TestClient(create_app()) as client:
        assert client.post(endpoint, json={"name": "demo", **body}).status_code == 200
        fail_route_additions(monkeypatch, "candidate route failed", "restore route failed")

        response = client.post(endpoint, json={"name": "demo", **body, "overwrite": True})

    assert response.status_code == 500
    assert "candidate route failed" in response.json()["detail"]
    assert "restore route failed" in response.json()["detail"]


def test_failed_overwrite_restores_a_contained_absolute_yaml_symlink(app, pipelines_dir, monkeypatch) -> None:
    old_wrapper = registry.get("demo")
    shared_yaml = pipelines_dir / "shared.yml"
    shared_yaml.write_text(SAMPLE_YAML)
    yaml_file = pipelines_dir / "demo.yml"
    yaml_file.symlink_to(shared_yaml.resolve())
    fail_route_additions(monkeypatch, "candidate route failed")

    with pytest.raises(RuntimeError, match="candidate route failed"):
        deploy_utils.deploy_pipeline_yaml("demo", SAMPLE_YAML, app=app, overwrite=True)

    assert yaml_file.is_symlink()
    assert yaml_file.resolve() == shared_yaml.resolve()
    assert shared_yaml.read_text() == SAMPLE_YAML
    assert_old_pipeline_restored(app, pipelines_dir, old_wrapper)
