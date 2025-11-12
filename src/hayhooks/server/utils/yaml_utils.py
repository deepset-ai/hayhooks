from typing import Any, TypedDict, Union

import yaml
from pydantic import BaseModel, Field

from hayhooks.server.exceptions import InvalidYamlIOError
from hayhooks.server.logger import log


class InputResolution(BaseModel):
    path: str = Field(
        description="Primary 'component.field' path chosen for type resolution and introspection.",
    )
    component: str = Field(
        description="Component name extracted from the primary path (matches the path used for type resolution)."
    )
    name: str = Field(description="Field name extracted from the primary path.")
    type: Any = Field(
        description="Python type associated with the primary path, as inferred from Haystack pipeline metadata."
    )
    required: bool = Field(
        description="Whether the request model marks this input as mandatory; declared inputs are always required."
    )
    targets: list[str] = Field(
        description="All declared 'component.field' targets that receive the same value when the pipeline is executed."
    )


class OutputResolution(BaseModel):
    path: str
    component: str
    name: str
    type: Any


class ResolvedIO(TypedDict):
    inputs: dict[str, InputResolution]
    outputs: dict[str, OutputResolution]


def _normalize_declared_path(value: Any) -> Union[str, None]:
    """
    Normalize a declared path from YAML to a string.

    A declared IO path in YAML can be provided either as a string (e.g. "comp.field")
    or as a list of strings. This helper normalizes those cases to the first valid
    string entry, or None if the value cannot be normalized.

    Args:
        value: Declared path value from YAML (string or list of strings).

    Returns:
        The normalized "component.field" string, or None when not available.
    """
    if isinstance(value, list):
        string_candidates = [candidate for candidate in value if isinstance(candidate, str)]

        if not string_candidates:
            return None

        path = next((candidate for candidate in string_candidates if "." in candidate), None)
        value = path or string_candidates[0]

    if not isinstance(value, str):
        return None

    return value


def _collect_candidate_paths(value: Any) -> list[str]:
    """
    Collect all candidate "component.field" paths from a declared IO value.

    Args:
        value: Declared path value from YAML (string or list).

    Returns:
        List of normalized "component.field" strings.
    """
    paths: list[str] = []

    if isinstance(value, list):
        candidates = [candidate for candidate in value if isinstance(candidate, str)]
        paths.extend(candidate for candidate in candidates if "." in candidate)
        if paths:
            return paths
        return candidates

    normalized = _normalize_declared_path(value)
    if isinstance(normalized, str) and normalized not in paths:
        paths.append(normalized)

    return [path for path in paths if "." in path]


def _resolve_declared_inputs(
    declared_map: dict[str, Any],
    pipeline_meta: dict[str, dict[str, Any]],
) -> dict[str, InputResolution]:
    """
    Resolve declared input entries using the pipeline metadata.

    Args:
        declared_map: Mapping from declared IO name to path (string or list).
        pipeline_meta: Pipeline inputs metadata as returned by Haystack.

    Returns:
        A mapping from declared IO name to `InputResolution`.
    """
    resolutions: dict[str, InputResolution] = {}
    target_to_declared_input: dict[str, str] = {}

    for io_name, declared_path in declared_map.items():
        candidate_paths = _collect_candidate_paths(declared_path)
        if not candidate_paths:
            continue

        # Deduplicate candidate paths while preserving order
        unique_candidate_paths = list(dict.fromkeys(candidate_paths))

        conflicts = {
            target: owner
            for target in unique_candidate_paths
            if (owner := target_to_declared_input.get(target)) and owner != io_name
        }
        if conflicts:
            conflicting_targets = ", ".join(f"'{target}'" for target in conflicts)
            log.debug(
                "Declared input '{}' reuses targets {} already assigned to {}.",
                io_name,
                conflicting_targets,
                conflicts,
            )

        if conflicts:
            conflict_messages = ", ".join(
                f"'{target}' already targeted by declared input '{owner}'" for target, owner in conflicts.items()
            )
            targets = ", ".join(f"'{target}'" for target in conflicts)
            msg = (
                f"Declared input '{io_name}' targets {targets}; "
                f"{conflict_messages}. Each pipeline input target may be declared only once."
            )
            raise InvalidYamlIOError(msg)

        target_to_declared_input.update(dict.fromkeys(unique_candidate_paths, io_name))

        normalized_path = unique_candidate_paths[0]
        component_name, field_name = normalized_path.split(".", 1)
        meta = (pipeline_meta.get(component_name, {}) or {}).get(field_name, {}) or {}
        resolved_type = meta.get("type")
        is_required = True

        resolutions[io_name] = InputResolution(
            path=f"{component_name}.{field_name}",
            component=component_name,
            name=field_name,
            type=resolved_type,
            required=is_required,
            targets=unique_candidate_paths,
        )

    return resolutions


def _resolve_declared_outputs(
    declared_map: dict[str, Any],
    pipeline_meta: dict[str, dict[str, Any]],
) -> dict[str, OutputResolution]:
    """
    Resolve declared output entries using the pipeline metadata.

    Args:
        declared_map: Mapping from declared IO name to path (string or list).
        pipeline_meta: Pipeline outputs metadata as returned by Haystack.

    Returns:
        A mapping from declared IO name to `OutputResolution`.
    """
    resolutions: dict[str, OutputResolution] = {}
    for io_name, declared_path in declared_map.items():
        normalized_path = _normalize_declared_path(declared_path)
        if not isinstance(normalized_path, str) or "." not in normalized_path:
            continue

        component_name, field_name = normalized_path.split(".", 1)
        meta = (pipeline_meta.get(component_name, {}) or {}).get(field_name, {}) or {}
        resolved_type = meta.get("type")

        resolutions[io_name] = OutputResolution(
            path=f"{component_name}.{field_name}",
            component=component_name,
            name=field_name,
            type=resolved_type,
        )

    return resolutions


def get_inputs_outputs_from_yaml(yaml_source_code: str) -> ResolvedIO:
    """
    Resolve inputs and outputs from a Haystack pipeline YAML.

    This function aligns the YAML-declared inputs and outputs with the pipeline
    metadata returned by Haystack, producing for each declared IO its path,
    component, field name, resolved type, and (for inputs) the required flag.

    Args:
        yaml_source_code: Pipeline YAML source code.

    Returns:
        A dictionary with two keys: "inputs" and "outputs". Each value is a mapping
        from the declared IO name to a resolution model (`InputResolution` for inputs,
        `OutputResolution` for outputs).

    Raises:
        InvalidYamlIOError: If both inputs and outputs are missing from the YAML definition.
    """
    yaml_dict = yaml.safe_load(yaml_source_code) or {}
    declared_inputs = yaml_dict.get("inputs", {}) or {}
    declared_outputs = yaml_dict.get("outputs", {}) or {}

    if not declared_inputs and not declared_outputs:
        msg = "YAML pipeline must declare at least one of 'inputs' or 'outputs'."
        raise InvalidYamlIOError(msg)

    from haystack import Pipeline

    pipeline = Pipeline.loads(yaml_source_code)
    pipeline_inputs = pipeline.inputs()
    pipeline_outputs = pipeline.outputs()

    input_resolutions = _resolve_declared_inputs(declared_inputs, pipeline_inputs)
    output_resolutions = _resolve_declared_outputs(declared_outputs, pipeline_outputs)

    return {"inputs": input_resolutions, "outputs": output_resolutions}


def get_streaming_components_from_yaml(yaml_source_code: str) -> Union[list[str], str, None]:
    """
    Extract streaming components configuration from a Haystack pipeline YAML.

    The streaming_components field is optional and specifies which components should stream.
    By default (when not specified), only the last streaming-capable component will stream.

    Args:
        yaml_source_code: Pipeline YAML source code.

    Returns:
        - None if not specified (use default behavior)
        - "all" if streaming_components is set to "all"
        - list[str] of component names that should stream
        Example: ["llm_1", "llm_2"]
    """
    yaml_dict = yaml.safe_load(yaml_source_code) or {}
    streaming_components = yaml_dict.get("streaming_components")

    if streaming_components is None:
        return None

    # Support "all" keyword
    if isinstance(streaming_components, str) and streaming_components.lower() == "all":
        return "all"

    if not isinstance(streaming_components, list):
        return None

    # Ensure all items are strings
    return [str(item) for item in streaming_components if item]


def get_components_from_outputs(resolved_outputs: dict[str, OutputResolution]) -> set[str]:
    """
    Extract component names from resolved outputs.

    Args:
        resolved_outputs: Resolved outputs from get_inputs_outputs_from_yaml

    Returns:
        Set of component names referenced in the outputs
    """
    return {output.component for output in resolved_outputs.values()}
