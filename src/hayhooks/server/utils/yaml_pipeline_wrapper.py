import inspect
from collections import defaultdict
from pathlib import Path
from typing import Any

from haystack import AsyncPipeline

from hayhooks.server.logger import log
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.yaml_utils import (
    InputResolution,
    OutputResolution,
    get_components_from_outputs,
    get_inputs_outputs_from_yaml,
    get_streaming_components_from_yaml,
)


def _set_method_signature(
    func,
    params: list[inspect.Parameter],
    return_annotation: type = inspect.Signature.empty,
) -> None:
    """
    Set a dynamic signature on a function intended to be used as a bound method.

    This is the standard Python pattern for dynamic signatures (used by dataclasses,
    pydantic, attrs, etc.). We include 'self' because inspect.signature() on bound
    methods automatically removes the first parameter.

    Args:
        func: The function to modify (mutates in place).
        params: List of parameters (excluding 'self').
        return_annotation: Return type annotation.
    """
    self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    func.__signature__ = inspect.Signature([self_param, *params], return_annotation=return_annotation)


def _map_flat_inputs_to_components(
    flat_inputs: dict[str, Any], input_resolutions: dict[str, InputResolution]
) -> dict[str, Any]:
    """
    Expand flat input payloads to per-component inputs using resolved IO metadata.

    Args:
        flat_inputs: Payload with logical input names (e.g. {"query": "foo"}).
        input_resolutions: Resolved IO metadata keyed by logical input name.

    Returns:
        Mapping suitable for Haystack Pipeline.run, or the original payload when we cannot resolve it.
    """
    if not input_resolutions or not flat_inputs:
        return flat_inputs

    component_inputs: dict[str, dict[str, Any]] = defaultdict(dict)
    unresolved: dict[str, Any] = {}

    for input_name, value in flat_inputs.items():
        resolution = input_resolutions.get(input_name)
        if resolution is None:
            unresolved[input_name] = value
            continue

        targets = resolution.targets or [resolution.path]
        for target in targets:
            component, field = target.split(".", 1)
            component_inputs[component][field] = value

    if unresolved:
        # If we can't resolve all inputs explicitly, fall back to the original payload.
        return flat_inputs

    return dict(component_inputs)


def _create_dynamic_run_api_async(
    input_resolutions: dict[str, InputResolution],
    include_outputs_from: set[str] | None,
):
    """
    Create a run_api_async method with a dynamic signature based on YAML inputs.

    Args:
        input_resolutions: Resolved input metadata from YAML.
        output_resolutions: Resolved output metadata from YAML.
        include_outputs_from: Components to include outputs from.

    Returns:
        An async function with the appropriate signature for the YAML pipeline inputs.
    """
    # Build parameter list from resolved inputs
    params = []
    for name, resolution in input_resolutions.items():
        param_type = resolution.type if resolution.type is not None else Any
        default = inspect.Parameter.empty if resolution.required else None
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=param_type,
                default=default,
            )
        )

    async def run_api_async(self, **kwargs) -> dict:
        """Execute the YAML pipeline with the provided inputs."""
        pipeline: AsyncPipeline = self.pipeline

        # Map flat inputs to component inputs
        data = _map_flat_inputs_to_components(kwargs, input_resolutions)

        # Build kwargs for pipeline.run_async
        run_kwargs: dict[str, Any] = {"data": data}
        if include_outputs_from is not None:
            run_kwargs["include_outputs_from"] = include_outputs_from

        return await pipeline.run_async(**run_kwargs)

    _set_method_signature(run_api_async, params, return_annotation=dict)
    return run_api_async


class YAMLPipelineWrapper(BasePipelineWrapper):
    """
    Pipeline wrapper for YAML-defined Haystack pipelines.

    This wrapper allows YAML pipelines to be treated uniformly with code-based
    pipeline wrappers, supporting the same API patterns and registry mechanisms.
    """

    def __init__(  # noqa: PLR0913
        self,
        yaml_source: str,
        input_resolutions: dict[str, InputResolution],
        output_resolutions: dict[str, OutputResolution],
        include_outputs_from: set[str] | None = None,
        streaming_components: list[str] | str | None = None,
        description: str | None = None,
    ):
        """
        Initialize a YAML pipeline wrapper.

        This constructor is typically not called directly. Use `from_yaml()` or `from_file()` instead.

        Args:
            yaml_source: The YAML source code for the pipeline.
            input_resolutions: Resolved input metadata from YAML.
            output_resolutions: Resolved output metadata from YAML.
            include_outputs_from: Components to include outputs from.
            streaming_components: Streaming configuration from YAML.
            description: Optional description for the pipeline.
        """
        super().__init__()
        self._yaml_source = yaml_source
        self._input_resolutions = input_resolutions
        self._output_resolutions = output_resolutions
        self._include_outputs_from = include_outputs_from
        self._streaming_components = streaming_components
        self._description = description

        # Create and bind the dynamic run_api_async method
        dynamic_method = _create_dynamic_run_api_async(
            input_resolutions=input_resolutions,
            include_outputs_from=include_outputs_from,
        )
        # Bind the method to this instance
        self.run_api_async = dynamic_method.__get__(self, type(self))

        # Mark that run_api_async is implemented
        self._is_run_api_async_implemented = True

    @classmethod
    def from_yaml(cls, source_code: str, description: str | None = None) -> "YAMLPipelineWrapper":
        """
        Create a YAMLPipelineWrapper from YAML source code.

        Args:
            source_code: YAML pipeline source code.
            description: Optional description for the pipeline.

        Returns:
            A configured YAMLPipelineWrapper instance.

        Raises:
            InvalidYamlIOError: If the YAML is missing inputs/outputs declarations.
        """
        log.debug("Creating YAMLPipelineWrapper from YAML source")

        # Resolve inputs and outputs from YAML
        resolved_io = get_inputs_outputs_from_yaml(source_code)
        input_resolutions = resolved_io["inputs"]
        output_resolutions = resolved_io["outputs"]

        # Extract streaming components configuration if present
        streaming_components = get_streaming_components_from_yaml(source_code)
        if streaming_components:
            log.debug("Found streaming_components in YAML: {}", streaming_components)

        # Automatically derive include_outputs_from from the outputs mapping
        include_outputs_from: set[str] | None = None
        if output_resolutions:
            include_outputs_from = get_components_from_outputs(output_resolutions)
            log.debug("Auto-derived include_outputs_from from outputs: {}", include_outputs_from)

        wrapper = cls(
            yaml_source=source_code,
            input_resolutions=input_resolutions,
            output_resolutions=output_resolutions,
            include_outputs_from=include_outputs_from,
            streaming_components=streaming_components,
            description=description,
        )

        return wrapper

    @classmethod
    def from_file(cls, yaml_path: Path | str, description: str | None = None) -> "YAMLPipelineWrapper":
        """
        Create a YAMLPipelineWrapper from a YAML file path.

        Args:
            yaml_path: Path to the YAML pipeline file.
            description: Optional description for the pipeline.

        Returns:
            A configured YAMLPipelineWrapper instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            InvalidYamlIOError: If the YAML is missing inputs/outputs declarations.
        """
        path = Path(yaml_path)
        if not path.exists():
            msg = f"YAML pipeline file not found: {path}"
            raise FileNotFoundError(msg)

        log.debug("Creating YAMLPipelineWrapper from file: {}", path)
        source_code = path.read_text(encoding="utf-8")

        return cls.from_yaml(source_code, description=description)

    def setup(self) -> None:
        """
        Initialize the Haystack AsyncPipeline from the stored YAML source.

        This method is called during deployment to instantiate the actual pipeline.
        If the pipeline is already initialized, this method does nothing.
        """
        if getattr(self, "pipeline", None) is not None:
            return

        log.debug("Setting up YAMLPipelineWrapper - loading AsyncPipeline from YAML")
        try:
            self.pipeline = AsyncPipeline.loads(self._yaml_source)
            log.debug("AsyncPipeline successfully loaded from YAML")
        except Exception as e:
            msg = f"Failed to load AsyncPipeline from YAML: {e!s}"
            raise ValueError(msg) from e

    @property
    def input_resolutions(self) -> dict[str, InputResolution]:
        """Get the resolved input metadata."""
        return self._input_resolutions

    @property
    def output_resolutions(self) -> dict[str, OutputResolution]:
        """Get the resolved output metadata."""
        return self._output_resolutions

    @property
    def include_outputs_from(self) -> set[str] | None:
        """Get the components to include outputs from."""
        return self._include_outputs_from

    @property
    def streaming_components(self) -> list[str] | str | None:
        """Get the streaming components configuration."""
        return self._streaming_components

    @property
    def description(self) -> str | None:
        """Get the pipeline description."""
        return self._description
