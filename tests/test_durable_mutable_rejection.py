"""Live (mutable) deployment rejects durable wrappers at every admission boundary."""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from hayhooks.durable.context import DurableContext
from hayhooks.server.app import create_app
from hayhooks.server.exceptions import PipelineModeError
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils import deploy_utils
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.models import PreparedPipeline
from hayhooks.server.utils.module_loader import create_pipeline_wrapper_instance
from hayhooks.settings import StartupDeployStrategy

ORDINARY_SOURCE = """
from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    def run_api(self, value: int) -> int:
        \"\"\"Double a value.\"\"\"
        return value * 2
"""

DURABLE_SOURCE = """
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext


class Request(BaseModel):
    value: int


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "v1"

    def setup(self) -> None:
        raise AssertionError("durable setup must not run")

    def run_api(self, value: int) -> int:
        return value

    async def run_durable_async(self, context: DurableContext, request: Request) -> dict:
        return {}
"""


class Request(BaseModel):
    value: int


class DurableOnlyWrapper(BasePipelineWrapper):
    durable_revision = "v1"
    setup_calls = 0

    def setup(self) -> None:
        type(self).setup_calls += 1

    async def run_durable_async(self, context: DurableContext, request: Request) -> dict:
        return {}


class MixedWrapper(DurableOnlyWrapper):
    def run_api(self, value: int) -> int:
        return value


class SetupAssignsDurableWrapper(BasePipelineWrapper):
    setup_calls = 0

    def setup(self) -> None:
        type(self).setup_calls += 1
        self.run_durable = lambda _context, _request: {}

    def run_api(self, value: int) -> int:
        return value


DURABLE_WRAPPERS = [
    pytest.param(DurableOnlyWrapper, id="durable-only"),
    pytest.param(MixedWrapper, id="mixed"),
    pytest.param(SetupAssignsDurableWrapper, id="assigned-in-setup"),
]


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear()
    yield
    registry.clear()


@pytest.mark.parametrize("wrapper_class", DURABLE_WRAPPERS)
def test_legacy_preparation_rejects_durable_wrapper(wrapper_class: type[BasePipelineWrapper]) -> None:
    wrapper_class.setup_calls = 0

    with pytest.raises(PipelineModeError, match="cannot be deployed through live deployment"):
        create_pipeline_wrapper_instance(SimpleNamespace(PipelineWrapper=wrapper_class), allow_durable=False)

    # Durable methods visible on the class are rejected before user setup can run.
    assert wrapper_class.setup_calls == (wrapper_class is SetupAssignsDurableWrapper)


@pytest.mark.parametrize(
    "admit",
    [
        pytest.param(
            lambda wrapper: deploy_utils.commit_prepared_pipeline(PreparedPipeline("demo", wrapper)), id="commit"
        ),
        pytest.param(lambda wrapper: registry.add("demo", wrapper), id="registry"),
    ],
)
@pytest.mark.parametrize("wrapper_class", DURABLE_WRAPPERS)
def test_publication_rejects_constructed_durable_wrapper(wrapper_class: type[BasePipelineWrapper], admit) -> None:
    # Manual construction leaves the capability flags unset, so rejection must inspect the methods.
    wrapper = wrapper_class()
    wrapper.setup()

    with pytest.raises(PipelineModeError):
        admit(wrapper)

    assert registry.get("demo") is None


def test_http_durable_overwrite_is_rejected_and_preserves_ordinary_pipeline(test_settings) -> None:
    with TestClient(create_app()) as client:
        deployed = client.post(
            "/deploy_files", json={"name": "demo", "files": {"pipeline_wrapper.py": ORDINARY_SOURCE}}
        )
        assert deployed.status_code == 200
        old_module = sys.modules["demo.pipeline_wrapper"]
        old_metadata = registry.get_metadata("demo")

        rejected = client.post(
            "/deploy_files",
            json={"name": "demo", "files": {"pipeline_wrapper.py": DURABLE_SOURCE}, "overwrite": True},
        )

        assert rejected.status_code == 422
        assert "cannot be deployed through live deployment" in rejected.json()["detail"]
        assert "HAYHOOKS_DURABLE_MODE=true" in rejected.json()["detail"]
        assert (Path(test_settings.pipelines_dir) / "demo" / "pipeline_wrapper.py").read_text() == ORDINARY_SOURCE
        assert sys.modules["demo.pipeline_wrapper"] is old_module
        assert registry.get_metadata("demo") is old_metadata
        assert client.post("/demo/run", json={"value": 21}).json() == {"result": 42}


def test_commit_rejects_durable_overwrite_before_removing_existing_pipeline(
    test_settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(test_settings, "pipelines_dir", str(tmp_path))
    app = create_app()
    with TestClient(app) as client:
        assert (
            client.post(
                "/deploy_files", json={"name": "demo", "files": {"pipeline_wrapper.py": ORDINARY_SOURCE}}
            ).status_code
            == 200
        )
        old_wrapper = registry.get("demo")
        old_module = sys.modules["demo.pipeline_wrapper"]
        backup = Mock(wraps=deploy_utils._backup_pipeline_files)
        remove = Mock(wraps=registry.remove)
        monkeypatch.setattr(deploy_utils, "_backup_pipeline_files", backup)
        monkeypatch.setattr(registry, "remove", remove)

        with pytest.raises(PipelineModeError):
            deploy_utils.commit_prepared_pipeline(
                PreparedPipeline("demo", MixedWrapper()),
                app=app,
                overwrite=True,
                source_files={"pipeline_wrapper.py": DURABLE_SOURCE},
            )

        backup.assert_not_called()
        remove.assert_not_called()
        assert registry.get("demo") is old_wrapper
        assert sys.modules["demo.pipeline_wrapper"] is old_module
        assert (Path(test_settings.pipelines_dir) / "demo" / "pipeline_wrapper.py").read_text() == ORDINARY_SOURCE
        assert client.post("/demo/run", json={"value": 21}).json() == {"result": 42}


@pytest.mark.parametrize("durable_mode", ["false", "true"])
def test_ordinary_server_loads_no_durable_runtime_redis_or_optional_transports(
    tmp_path: Path, durable_mode: str
) -> None:
    (tmp_path / "ordinary").mkdir()
    (tmp_path / "ordinary" / "pipeline_wrapper.py").write_text(ORDINARY_SOURCE)
    script = textwrap.dedent(
        """
        import sys


        class BlockOptionalTransports:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("mcp", "a2a"):
                    raise ModuleNotFoundError(name)


        sys.meta_path.insert(0, BlockOptionalTransports())

        from fastapi.testclient import TestClient

        from hayhooks.server.app import create_app

        app = create_app()
        with TestClient(app) as client:
            assert client.post("/ordinary/run", json={"value": 2}).json() == {"result": 4}
        assert app.state.durable_runtime is None and app.state.durable_redis_clients == ()

        # The wrapper base class needs DurableContext; nothing may load the runtime, transports, or Redis.
        durable = {name for name in sys.modules if name.startswith("hayhooks.durable")}
        core = ("", "._threading", ".context", ".engine", ".models", ".store")
        allowed = {f"hayhooks.durable{suffix}" for suffix in core}
        assert durable <= allowed, durable - allowed
        assert "redis" not in sys.modules
        """
    )
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "HAYHOOKS_PIPELINES_DIR": str(tmp_path), "HAYHOOKS_DURABLE_MODE": durable_mode},
        timeout=60,
    )


@pytest.fixture
def pipelines_dir(tmp_path: Path, test_settings, monkeypatch) -> Path:
    for name, source in (("ordinary", ORDINARY_SOURCE), ("durable", DURABLE_SOURCE)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pipeline_wrapper.py").write_text(source)
    monkeypatch.setattr(test_settings, "pipelines_dir", str(tmp_path))
    return tmp_path


def _raise_mode_error(*_args, **_kwargs):
    msg = "durable"
    raise PipelineModeError(msg)


@pytest.mark.parametrize(
    ("strategy", "patch"),
    [
        pytest.param(StartupDeployStrategy.SEQUENTIAL, None, id="sequential-directories"),
        pytest.param(StartupDeployStrategy.SEQUENTIAL, "deploy_yaml_pipeline", id="sequential-yaml"),
        pytest.param(StartupDeployStrategy.PARALLEL, None, id="parallel-prepare"),
        pytest.param(StartupDeployStrategy.PARALLEL, "commit_prepared_pipeline", id="parallel-commit"),
    ],
)
def test_server_startup_fails_on_durable_wrapper(
    pipelines_dir: Path, test_settings, monkeypatch, strategy: StartupDeployStrategy, patch: str | None
) -> None:
    monkeypatch.setattr(test_settings, "startup_deploy_strategy", strategy)
    if patch is not None:
        # The patched loop step raises instead, so a skipped error cannot be masked by the real wrapper.
        shutil.rmtree(pipelines_dir / "durable")
        shutil.copy(Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml", pipelines_dir / "calc.yml")
        monkeypatch.setattr(f"hayhooks.server.app.{patch}", _raise_mode_error)
    ready = False

    with pytest.raises(PipelineModeError), TestClient(create_app()):
        ready = True

    assert not ready


def test_standalone_startup_fails_on_durable_wrapper(pipelines_dir: Path) -> None:
    with pytest.raises(PipelineModeError):
        deploy_utils.deploy_pipelines()
