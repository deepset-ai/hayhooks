from typing import Any, Union

import yaml
from pydantic import BaseModel


class BaseInputOutputResolution(BaseModel):
    path: str
    component: str
    name: str
    type: Any


class InputResolution(BaseInputOutputResolution):
    required: bool


class OutputResolution(BaseInputOutputResolution):
    pass


def _normalize_declared_path(value: Any) -> Union[str, None]:
    """
    Normalize a declared path value.

    A declared IO path in YAML can be provided either as a string (e.g. "comp.field")
    or as a one-item list of strings. This helper normalizes both cases to a single
    string, or None if the value cannot be normalized.

    Args:
        value: Declared path value from YAML (string or list of strings).

    Returns:
        The normalized "component.field" string, or None when not available.
    """
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _resolve_declared_io(
    declared_map: dict[str, Any],
    pipeline_meta: dict[str, dict[str, Any]],
) -> dict[str, BaseInputOutputResolution]:
    """
    Resolve declared IO entries using the pipeline metadata.

    Auto-detects input vs output based on metadata: inputs generally expose
    "is_mandatory" or "default_value", while outputs do not.

    Args:
        declared_map: Mapping from declared IO name to path (string or list).
        pipeline_meta: Pipeline metadata as returned by Haystack (inputs/outputs).

    Returns:
        A mapping from declared IO name to `InputResolution` or `OutputResolution`.
    """
    resolutions: dict[str, BaseInputOutputResolution] = {}
    for io_name, declared_path in declared_map.items():
        normalized_path = _normalize_declared_path(declared_path)
        if not isinstance(normalized_path, str) or "." not in normalized_path:
            continue

        component_name, field_name = normalized_path.split(".", 1)
        meta = (pipeline_meta.get(component_name, {}) or {}).get(field_name, {}) or {}
        resolved_type = meta.get("type")

        # inputs metadata expose "is_mandatory"/"default_value"
        is_input_like = "is_mandatory" in meta or "default_value" in meta
        if is_input_like:
            resolutions[io_name] = InputResolution(
                path=f"{component_name}.{field_name}",
                component=component_name,
                name=field_name,
                type=resolved_type,
                required=bool(meta.get("is_mandatory", False)),
            )
        else:
            resolutions[io_name] = OutputResolution(
                path=f"{component_name}.{field_name}",
                component=component_name,
                name=field_name,
                type=resolved_type,
            )

    return resolutions


def get_inputs_outputs_from_yaml(yaml_source_code: str) -> dict[str, dict[str, BaseInputOutputResolution]]:
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
        ValueError: If both inputs and outputs are missing from the YAML definition.
    """
    yaml_dict = yaml.safe_load(yaml_source_code) or {}
    declared_inputs = yaml_dict.get("inputs", {}) or {}
    declared_outputs = yaml_dict.get("outputs", {}) or {}

    if not declared_inputs and not declared_outputs:
        msg = "YAML pipeline must declare at least one of 'inputs' or 'outputs'."
        raise ValueError(msg)

    from haystack import Pipeline

    pipeline = Pipeline.loads(yaml_source_code)
    pipeline_inputs = pipeline.inputs()
    pipeline_outputs = pipeline.outputs()

    input_resolutions = _resolve_declared_io(declared_inputs, pipeline_inputs)
    output_resolutions = _resolve_declared_io(declared_outputs, pipeline_outputs)

    return {"inputs": input_resolutions, "outputs": output_resolutions}
