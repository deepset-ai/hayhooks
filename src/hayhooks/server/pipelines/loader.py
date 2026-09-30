"""
Load the fixed pipeline set of durable mode.

Every pipeline loads once, at startup, from the configured directory. Wrapper packages live
under one private import root, so their module paths do not depend on the source location
and cannot shadow real modules. Durable workers serialize these paths into checkpoints, so
the root is part of the durable storage contract.
"""

import re
import sys
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from os import PathLike
from pathlib import Path
from typing import TypeVar

from hayhooks.server.exceptions import PipelineModuleLoadError
from hayhooks.server.logger import log
from hayhooks.server.pipelines.models import create_pipeline_metadata
from hayhooks.server.pipelines.registry import ImmutablePipelineRegistry, PipelineRegistration
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.module_loader import (
    create_pipeline_wrapper_instance,
    load_pipeline_module,
    unload_pipeline_modules,
)
from hayhooks.server.utils.yaml_pipeline_wrapper import YAMLPipelineWrapper

# Renaming the root breaks every persisted checkpoint that references a wrapper-local symbol.
REGISTRY_ROOT = "_hayhooks_registry"

# One URL path segment, as in the /{pipeline_name}/run route.
_PIPELINE_NAME = re.compile(r"[A-Za-z0-9_-]+")
_YAML_SUFFIXES = (".yml", ".yaml")

T = TypeVar("T")


def registry_module_name(pipeline_name: str) -> str:
    """Return the module path of a wrapper pipeline loaded in durable mode, e.g. for deserialization allowlists."""
    return f"{_package_name(pipeline_name)}.pipeline_wrapper"


def _package_name(pipeline_name: str) -> str:
    # Hex keeps names such as "my-pipeline" importable without ambiguous slugging.
    return f"{REGISTRY_ROOT}.p_{pipeline_name.encode().hex()}"


def load_pipeline_registry(pipelines_dir: PathLike | str) -> ImmutablePipelineRegistry:
    """
    Load every pipeline in *pipelines_dir* into an immutable registry, or fail without publishing any.

    Top-level ``.yml``/``.yaml`` files are YAML pipelines and first-level directories are wrapper
    pipelines, which must contain ``pipeline_wrapper.py``. Entries starting with a dot and
    ``__pycache__`` are skipped; other files are ignored. Automatic bytecode writing is disabled
    for the rest of the process, so loading and later imports leave the source tree unchanged.

    Raises:
        PipelineModuleLoadError: If the directory is unreadable, a name is invalid or duplicated,
            or any pipeline fails to load or set up. Modules loaded so far are unloaded.
    """
    candidates = _discover_candidates(Path(pipelines_dir))

    sys.dont_write_bytecode = True
    unload_pipeline_modules(REGISTRY_ROOT)
    try:
        sys.modules[REGISTRY_ROOT] = module_from_spec(ModuleSpec(REGISTRY_ROOT, loader=None, is_package=True))
        registry = ImmutablePipelineRegistry(_load_candidate(name, path) for name, path in candidates.items())
    except BaseException:
        unload_pipeline_modules(REGISTRY_ROOT)
        raise

    for name, path in candidates.items():
        if path.is_dir():
            log.info("Loaded pipeline '{}' from '{}' as module '{}'", name, path, registry_module_name(name))
        else:
            log.info("Loaded YAML pipeline '{}' from '{}'", name, path)
    return registry


def build_immutable_host(pipelines_dir: PathLike | str, build: Callable[[ImmutablePipelineRegistry], T]) -> T:
    """Load the immutable registry and build its host; if either fails, unload the pipeline modules."""
    registry = load_pipeline_registry(pipelines_dir)
    try:
        return build(registry)
    except BaseException:
        unload_pipeline_modules(REGISTRY_ROOT)
        raise


def _discover_candidates(pipelines_dir: Path) -> dict[str, Path]:
    try:
        entries = sorted(pipelines_dir.iterdir())
    except OSError as error:
        msg = f"Pipelines directory '{pipelines_dir}' is not a readable directory"
        raise PipelineModuleLoadError(msg) from error

    candidates: dict[str, Path] = {}
    for entry in entries:
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if entry.is_dir():
            name = entry.name
        elif entry.suffix in _YAML_SUFFIXES:
            name = entry.stem
        else:
            log.debug("Ignoring '{}': not a pipeline definition", entry)
            continue

        if not _PIPELINE_NAME.fullmatch(name):
            msg = f"Invalid pipeline name '{name}' derived from '{entry}': use letters, digits, '_' and '-'"
            raise PipelineModuleLoadError(msg)
        if name in candidates:
            msg = f"Pipeline '{name}' is defined twice: '{candidates[name]}' and '{entry}'"
            raise PipelineModuleLoadError(msg)
        candidates[name] = entry
    return dict(sorted(candidates.items()))


def _load_candidate(name: str, path: Path) -> PipelineRegistration:
    try:
        wrapper: BasePipelineWrapper
        if path.is_dir():
            package_name = _package_name(name)
            module = load_pipeline_module(name, path, package_name=package_name)
            package = sys.modules[package_name]
            package.__dict__["pipeline_wrapper"] = module
            setattr(sys.modules[REGISTRY_ROOT], package_name.rpartition(".")[2], package)
            wrapper = create_pipeline_wrapper_instance(module)
            metadata = create_pipeline_metadata(name, wrapper)
        else:
            wrapper = YAMLPipelineWrapper.from_yaml(path.read_text(encoding="utf-8"))
            wrapper.setup()
            metadata = {**create_pipeline_metadata(name, wrapper), "description": name}
    except Exception as error:
        msg = f"Failed to load pipeline '{name}' from '{path}': {error}"
        raise PipelineModuleLoadError(msg) from error
    return PipelineRegistration(name, wrapper, metadata)
