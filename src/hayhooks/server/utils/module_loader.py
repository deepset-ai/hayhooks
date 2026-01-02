"""
Pipeline module loading utilities.

This module handles loading pipeline wrapper modules as Python packages,
enabling relative imports and proper sys.modules registration for tracing libraries.
"""

import importlib.util
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import NoReturn

from hayhooks.server.exceptions import PipelineModuleLoadError, PipelineWrapperError
from hayhooks.server.logger import log
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings


def load_pipeline_module(pipeline_name: str, dir_path: Path | str) -> ModuleType:
    """
    Load a pipeline module from a directory path.

    The pipeline directory is treated as a Python package, enabling both:
    - Relative imports within the pipeline folder (e.g., `from .helper import func`)
    - Absolute imports from sibling modules (e.g., `from helper import func`)

    The module is also registered in sys.modules to support tracing libraries
    (e.g., Phoenix/OpenInference) that resolve modules by name.

    Args:
        pipeline_name: Name of the pipeline (used as the package name)
        dir_path: Path to the directory containing the pipeline files

    Returns:
        The loaded wrapper module

    Raises:
        PipelineModuleLoadError: If the module cannot be loaded (wraps all errors)
        PipelineWrapperError: If required symbols are missing in the loaded module
    """
    log.trace("Loading pipeline module from '{}'", dir_path)

    dir_path = Path(dir_path)
    loader = _PipelineModuleLoader(pipeline_name, dir_path)
    return loader.load()


def unload_pipeline_modules(pipeline_name: str) -> None:
    """
    Remove pipeline modules from sys.modules.

    This should be called when undeploying a pipeline to clean up.

    Args:
        pipeline_name: Name of the pipeline to unload
    """
    module_names = [pipeline_name, f"{pipeline_name}.pipeline_wrapper"]
    for mod_name in module_names:
        if mod_name in sys.modules:
            log.debug("Removing module '{}' from sys.modules", mod_name)
            del sys.modules[mod_name]


def create_pipeline_wrapper_instance(pipeline_module: ModuleType) -> BasePipelineWrapper:
    """
    Instantiate a `PipelineWrapper` from a loaded module and verify supported methods.

    Args:
        pipeline_module: The loaded module exposing a `PipelineWrapper` class.

    Returns:
        An initialized PipelineWrapper instance with capability flags set.

    Raises:
        PipelineWrapperError: If instantiation or setup fails, or if no supported run methods are implemented.
    """
    try:
        pipeline_wrapper = pipeline_module.PipelineWrapper()
    except Exception as e:
        error_msg = "Failed to create pipeline wrapper instance: " + str(e)
        if settings.show_tracebacks:
            error_msg += f"\n{traceback.format_exc()}"
        raise PipelineWrapperError(error_msg) from e

    try:
        pipeline_wrapper.setup()
    except Exception as e:
        error_msg = "Failed to call setup() on pipeline wrapper instance: " + str(e)
        if settings.show_tracebacks:
            error_msg += f"\n{traceback.format_exc()}"
        raise PipelineWrapperError(error_msg) from e

    # Set implementation flags for each supported method
    _set_method_implementation_flags(pipeline_wrapper)

    # Validate at least one run method is implemented
    _validate_run_methods(pipeline_wrapper)

    return pipeline_wrapper


class _PipelineModuleLoader:
    """
    Internal class to manage the lifecycle of loading a pipeline module.

    Handles sys.path management, module creation, and cleanup on failure.
    """

    def __init__(self, pipeline_name: str, dir_path: Path):
        self.pipeline_name = pipeline_name
        self.dir_path = dir_path.resolve()
        self.dir_path_str = str(self.dir_path)
        self.parent_dir_str = str(self.dir_path.parent)

        # Module names
        self.package_name = pipeline_name
        self.wrapper_module_name = f"{pipeline_name}.pipeline_wrapper"

        # Cleanup tracking
        self._path_added = False
        self._modules_registered: list[str] = []

    def load(self) -> ModuleType:
        try:
            self._check_wrapper_exists()
            self._clear_existing_modules()
            self._add_parent_to_sys_path()
            self._create_package_module()

            wrapper_module = self._load_wrapper_module()
            self._validate_wrapper_module(wrapper_module)

            return wrapper_module

        except Exception as e:
            self._cleanup_on_failure()
            self._raise_load_error(e)

    def _check_wrapper_exists(self) -> None:
        wrapper_path = self.dir_path / "pipeline_wrapper.py"
        if not wrapper_path.exists():
            msg = f"Required file '{wrapper_path}' not found"
            raise PipelineModuleLoadError(msg)

    def _clear_existing_modules(self) -> None:
        unload_pipeline_modules(self.pipeline_name)

    def _add_parent_to_sys_path(self) -> None:
        if self.parent_dir_str not in sys.path:
            sys.path.insert(0, self.parent_dir_str)
            self._path_added = True
            log.debug("Added '{}' to sys.path", self.parent_dir_str)

    def _create_package_module(self) -> None:
        init_path = self.dir_path / "__init__.py"

        if init_path.exists():
            package_module = self._create_package_from_init(init_path)
        else:
            package_module = self._create_namespace_package()

        # Set package attributes
        package_module.__path__ = [self.dir_path_str]
        package_module.__package__ = self.package_name
        package_module.__file__ = str(init_path) if init_path.exists() else None

        # Register in sys.modules
        sys.modules[self.package_name] = package_module
        self._modules_registered.append(self.package_name)
        log.debug("Created package module '{}' with __path__={}", self.package_name, package_module.__path__)

    def _create_package_from_init(self, init_path: Path) -> ModuleType:
        package_spec = importlib.util.spec_from_file_location(
            self.package_name,
            init_path,
            submodule_search_locations=[self.dir_path_str],
        )
        if package_spec is None:
            msg = f"Failed to create package spec for '{self.pipeline_name}'"
            raise PipelineModuleLoadError(msg)

        package_module = importlib.util.module_from_spec(package_spec)

        # Execute __init__.py if it has a loader
        if package_spec.loader is not None:
            package_spec.loader.exec_module(package_module)

        return package_module

    def _create_namespace_package(self) -> ModuleType:
        return ModuleType(self.package_name)

    def _load_wrapper_module(self) -> ModuleType:
        wrapper_path = self.dir_path / "pipeline_wrapper.py"

        wrapper_spec = importlib.util.spec_from_file_location(
            self.wrapper_module_name,
            wrapper_path,
            submodule_search_locations=[],
        )
        if wrapper_spec is None or wrapper_spec.loader is None:
            msg = f"Failed to load pipeline module '{self.pipeline_name}' - module loader not available"
            raise PipelineModuleLoadError(msg)

        wrapper_module = importlib.util.module_from_spec(wrapper_spec)
        wrapper_module.__package__ = self.package_name

        # Register BEFORE exec_module to enable self-references and tracing
        sys.modules[self.wrapper_module_name] = wrapper_module
        self._modules_registered.append(self.wrapper_module_name)

        wrapper_spec.loader.exec_module(wrapper_module)
        log.debug("Loaded module '{}' and registered in sys.modules", self.wrapper_module_name)

        return wrapper_module

    def _validate_wrapper_module(self, wrapper_module: ModuleType) -> None:
        if not hasattr(wrapper_module, "PipelineWrapper"):
            msg = f"Module '{self.pipeline_name}' does not define a 'PipelineWrapper' class"
            raise PipelineWrapperError(msg)

    def _cleanup_on_failure(self) -> None:
        for mod_name in self._modules_registered:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        if self._path_added and self.parent_dir_str in sys.path:
            sys.path.remove(self.parent_dir_str)

    def _raise_load_error(self, original_error: Exception) -> NoReturn:
        log.opt(exception=True).error("Error loading pipeline module: {}", original_error)
        error_msg = f"Failed to load pipeline module '{self.pipeline_name}' - {original_error!s}"

        if settings.show_tracebacks:
            error_msg += f"\n{traceback.format_exc()}"

        raise PipelineModuleLoadError(error_msg) from original_error


def _set_method_implementation_flags(pipeline_wrapper: BasePipelineWrapper) -> None:
    """
    Set implementation flags for all supported run methods.

    Args:
        pipeline_wrapper: The wrapper instance to annotate with flags.
    """
    methods_to_check = [
        ("_is_run_api_implemented", "run_api"),
        ("_is_run_api_async_implemented", "run_api_async"),
        ("_is_run_chat_completion_implemented", "run_chat_completion"),
        ("_is_run_chat_completion_async_implemented", "run_chat_completion_async"),
    ]

    for attr_name, method_name in methods_to_check:
        is_implemented = _is_method_overridden(pipeline_wrapper, method_name)
        setattr(pipeline_wrapper, attr_name, is_implemented)
        log.debug("pipeline_wrapper.{}: {}", attr_name, is_implemented)


def _is_method_overridden(pipeline_wrapper: BasePipelineWrapper, method_name: str) -> bool:
    """
    Check if a method is overridden in the wrapper compared to the base class.

    Args:
        pipeline_wrapper: The wrapper instance to check.
        method_name: The method name to check (e.g., "run_api").

    Returns:
        True if the method is overridden, False otherwise.
    """
    wrapper_method = getattr(pipeline_wrapper, method_name, None)
    base_method = getattr(BasePipelineWrapper, method_name, None)

    if wrapper_method is None or base_method is None:
        return False

    # Compare the underlying function objects
    wrapper_func = getattr(wrapper_method, "__func__", wrapper_method)
    base_func = getattr(base_method, "__func__", base_method)

    return wrapper_func is not base_func


def _validate_run_methods(pipeline_wrapper: BasePipelineWrapper) -> None:
    """
    Validate that at least one run method is implemented.

    Args:
        pipeline_wrapper: The wrapper instance to validate.

    Raises:
        PipelineWrapperError: If no run methods are implemented.
    """
    has_run_method = any(
        [
            pipeline_wrapper._is_run_api_implemented,
            pipeline_wrapper._is_run_api_async_implemented,
            pipeline_wrapper._is_run_chat_completion_implemented,
            pipeline_wrapper._is_run_chat_completion_async_implemented,
        ]
    )

    if not has_run_method:
        msg = (
            "At least one of run_api, run_api_async, run_chat_completion, or run_chat_completion_async "
            "must be implemented"
        )
        raise PipelineWrapperError(msg)
