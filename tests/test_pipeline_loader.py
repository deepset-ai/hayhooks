"""The immutable startup loader and the registry it publishes."""

import json
import pickle
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

from hayhooks.server.exceptions import PipelineModuleLoadError
from hayhooks.server.pipelines.loader import REGISTRY_ROOT, load_pipeline_registry, registry_module_name
from hayhooks.server.pipelines.registry import ImmutablePipelineRegistry, PipelineRegistration
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from tests.pipeline_sources import CALC_YAML, DURABLE_WRAPPER, ORDINARY_WRAPPER, write_tree

# Relative imports from __init__.py, setup(), and a request-time helper, plus a source asset.
PACKAGE_WRAPPER = {
    "__init__.py": "from .constants import FACTOR\n",
    "constants.py": "FACTOR = 3\n",
    "deferred.py": "def triple(value):\n    return value * 3\n",
    "prompt.txt": "asset",
    "pipeline_wrapper.py": '''
from pathlib import Path

from hayhooks import BasePipelineWrapper

from . import FACTOR


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from .constants import FACTOR as factor

        self.pipeline = None
        self.factor = factor
        self.asset = (Path(__file__).parent / "prompt.txt").read_text()

    def run_api(self, value: int) -> int:
        """Triple a value."""
        from .deferred import triple

        return triple(value) * FACTOR // self.factor
''',
}

FAILING_SETUP = ORDINARY_WRAPPER.replace("self.pipeline = None", "raise RuntimeError('setup exploded')")


def _registry_modules() -> set[str]:
    return {name for name in sys.modules if name == REGISTRY_ROOT or name.startswith(f"{REGISTRY_ROOT}.")}


def test_loads_yaml_and_wrappers_in_name_order_skipping_non_candidates(durable_pipelines_dir: Path) -> None:
    write_tree(
        durable_pipelines_dir,
        {
            "zeta/pipeline_wrapper.py": ORDINARY_WRAPPER,
            "calc-x/pipeline_wrapper.py": ORDINARY_WRAPPER,  # its directory sorts before calc.yml
            "calc.yml": CALC_YAML,
            "alpha.yaml": CALC_YAML,
            "notes.md": "not a pipeline",
            ".hidden/pipeline_wrapper.py": "raise RuntimeError('hidden')",
            "__pycache__/stale.pyc": "",
            "zeta/nested/pipeline_wrapper.py": "raise RuntimeError('nested directories are not candidates')",
        },
    )

    registry = load_pipeline_registry(durable_pipelines_dir)

    assert registry.get_names() == ["alpha", "calc", "calc-x", "zeta"]
    assert registry.get("zeta").run_api(4) == 8  # ty: ignore[unresolved-attribute]
    assert registry.get_metadata(name="calc")["description"] == "calc"  # ty: ignore[not-subscriptable]
    assert sys.dont_write_bytecode


def test_empty_directory_is_an_empty_registry(durable_pipelines_dir: Path) -> None:
    assert load_pipeline_registry(durable_pipelines_dir).get_names() == []


def test_wrapper_package_imports_and_assets_resolve_from_the_source_directory(durable_pipelines_dir: Path) -> None:
    write_tree(durable_pipelines_dir / "my-pipeline", PACKAGE_WRAPPER)

    wrapper = load_pipeline_registry(durable_pipelines_dir).get("my-pipeline")

    module_name = registry_module_name("my-pipeline")
    assert module_name == f"{REGISTRY_ROOT}.p_{b'my-pipeline'.hex()}.pipeline_wrapper"
    assert type(wrapper).__module__ == module_name
    assert sys.modules[module_name].__file__ == str(
        (durable_pipelines_dir / "my-pipeline/pipeline_wrapper.py").resolve()
    )
    assert wrapper.asset == "asset"  # ty: ignore[unresolved-attribute]
    assert wrapper.run_api(2) == 6  # ty: ignore[unresolved-attribute]
    assert f"{REGISTRY_ROOT}.p_{b'my-pipeline'.hex()}.deferred" in sys.modules
    # Nothing is importable by its public name or through sys.path.
    assert "my-pipeline" not in sys.modules
    assert str(durable_pipelines_dir) not in sys.path


def test_pipeline_named_after_a_real_module_does_not_shadow_it(durable_pipelines_dir: Path) -> None:
    write_tree(durable_pipelines_dir, {"json/pipeline_wrapper.py": ORDINARY_WRAPPER})

    load_pipeline_registry(durable_pipelines_dir)

    assert sys.modules["json"] is json


def test_private_package_supports_standard_imports_and_pickling(durable_pipelines_dir: Path) -> None:
    write_tree(durable_pipelines_dir, {"jobs/pipeline_wrapper.py": DURABLE_WRAPPER})
    load_pipeline_registry(durable_pipelines_dir)
    module_name = registry_module_name("jobs")
    module = sys.modules[module_name]

    root = __import__(module_name)
    package = getattr(root, module_name.split(".")[1])
    assert package.pipeline_wrapper is module
    request = module.Request(value=7)
    restored = pickle.loads(pickle.dumps(request))
    assert type(restored) is module.Request
    assert restored == request


def test_reloading_replaces_the_previous_pipeline_modules(durable_pipelines_dir: Path, tmp_path: Path) -> None:
    write_tree(durable_pipelines_dir, {"first/pipeline_wrapper.py": ORDINARY_WRAPPER})
    load_pipeline_registry(durable_pipelines_dir)
    other = write_tree(tmp_path / "other", {"second/pipeline_wrapper.py": ORDINARY_WRAPPER})

    load_pipeline_registry(other)

    assert _registry_modules() == {
        REGISTRY_ROOT,
        f"{REGISTRY_ROOT}.p_{b'second'.hex()}",
        registry_module_name("second"),
    }


@pytest.mark.parametrize(
    ("files", "match"),
    [
        pytest.param({"broken/helper.py": ""}, "pipeline_wrapper.py' not found", id="directory-without-wrapper"),
        pytest.param({"bad name/pipeline_wrapper.py": ORDINARY_WRAPPER}, "Invalid pipeline name", id="invalid-name"),
        pytest.param({"dup.yml": CALC_YAML, "dup.yaml": CALC_YAML}, "defined twice", id="duplicate-yaml"),
        pytest.param(
            {"dup.yml": CALC_YAML, "dup/pipeline_wrapper.py": ORDINARY_WRAPPER}, "defined twice", id="duplicate-kinds"
        ),
        pytest.param({"invalid.yml": "components: {}\n"}, "Failed to load pipeline 'invalid'", id="invalid-yaml"),
        pytest.param({"setup/pipeline_wrapper.py": FAILING_SETUP}, "setup exploded", id="setup-error"),
        pytest.param(
            {"typed/pipeline_wrapper.py": "class PipelineWrapper:\n    pass\n"},
            "Failed to load pipeline 'typed'",
            id="not-a-wrapper",
        ),
        pytest.param(
            {"norun/pipeline_wrapper.py": ORDINARY_WRAPPER.replace("def run_api", "def other")},
            "At least one of run_api",
            id="no-run-method",
        ),
    ],
)
def test_any_failing_candidate_fails_the_load_and_unloads_its_modules(
    durable_pipelines_dir: Path, files: dict[str, str], match: str
) -> None:
    # A valid pipeline sorted first is loaded, then unloaded with the failure.
    write_tree(durable_pipelines_dir, {"aaa/pipeline_wrapper.py": ORDINARY_WRAPPER, **files})
    unrelated = sys.modules["json"]

    with pytest.raises(PipelineModuleLoadError, match=match) as error:
        load_pipeline_registry(durable_pipelines_dir)

    assert _registry_modules() == set()
    assert sys.modules["json"] is unrelated
    if "Failed to load" in str(error.value):
        assert error.value.__cause__ is not None


def _unreadable(root: Path) -> Path:
    path = root / "unreadable"
    path.mkdir(mode=0)
    return path


@pytest.mark.parametrize(
    "make_path",
    [
        pytest.param(lambda root: root / "missing", id="missing"),
        pytest.param(lambda root: write_tree(root, {"file.yml": CALC_YAML}) / "file.yml", id="file"),
        pytest.param(_unreadable, id="unreadable"),
    ],
)
def test_missing_unreadable_or_non_directory_path_fails_without_creating_it(
    durable_pipelines_dir: Path, make_path
) -> None:
    path = make_path(durable_pipelines_dir)
    existed = path.exists()

    try:
        with pytest.raises(PipelineModuleLoadError, match="not a readable directory"):
            load_pipeline_registry(path)
    finally:
        if path.is_dir():
            path.chmod(0o755)

    assert path.exists() == existed


class _Wrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pass


@pytest.mark.parametrize("read_only", [False, True])
def test_registry_copies_its_input_and_exposes_read_only_metadata(read_only: bool) -> None:
    wrapper = _Wrapper()
    card = {"name": "Demo agent"}
    metadata = {"description": "Demo", "a2a_card": card}
    registration = PipelineRegistration("demo", wrapper, MappingProxyType(metadata) if read_only else metadata)
    registrations = [registration]
    registry = ImmutablePipelineRegistry(registrations)
    registrations.clear()
    metadata["description"] = "changed"

    assert registry.get("demo") is wrapper
    assert registry.get_metadata(name="demo") == {"description": "Demo", "a2a_card": card}
    assert registry.get_metadata("demo")["a2a_card"] is card
    assert registry.get("missing") is None
    assert registry.get_metadata("missing") is None
    names = registry.get_names()
    names.clear()
    assert registry.get_names() == ["demo"]
    with pytest.raises(TypeError):
        registry.get_metadata("demo")["description"] = "changed"  # ty: ignore[invalid-assignment]
    assert not any(hasattr(registry, name) for name in ("add", "remove", "update_metadata", "clear"))
    with pytest.raises(ValueError, match="more than once"):
        ImmutablePipelineRegistry([registration, registration])
