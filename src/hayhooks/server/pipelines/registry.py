from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

from hayhooks.server.exceptions import PipelineModeError, PipelineNotFoundError
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.module_loader import is_durable_wrapper, reject_durable_wrapper
from hayhooks.settings import settings


class _PipelineRegistry:
    """
    Registry for pipeline wrappers.

    All pipelines are stored as BasePipelineWrapper instances (or subclasses like YAMLPipelineWrapper).
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, BasePipelineWrapper] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add(
        self, name: str, pipeline_wrapper: BasePipelineWrapper, metadata: dict[str, Any] | None = None
    ) -> BasePipelineWrapper:
        """
        Add a pipeline wrapper to the registry.

        Args:
            name: Unique name for the pipeline.
            pipeline_wrapper: A BasePipelineWrapper instance (or subclass).
            metadata: Optional metadata to associate with the pipeline.

        Returns:
            The registered pipeline wrapper.

        Raises:
            ValueError: If a pipeline with the same name already exists.
            TypeError: If pipeline_wrapper is not a BasePipelineWrapper instance.
            PipelineModeError: If pipeline_wrapper is durable.
        """
        if metadata is None:
            metadata = {}

        if name in self._pipelines:
            msg = f"A pipeline with name {name} is already in the registry."
            raise ValueError(msg)

        if not isinstance(pipeline_wrapper, BasePipelineWrapper):
            msg = f"Expected BasePipelineWrapper instance, got {type(pipeline_wrapper).__name__}"
            raise TypeError(msg)

        reject_durable_wrapper(pipeline_wrapper)

        self._pipelines[name] = pipeline_wrapper
        self._metadata[name] = metadata

        return pipeline_wrapper

    def remove(self, name: str) -> None:
        if name in self._pipelines:
            del self._pipelines[name]
            del self._metadata[name]

    def get(self, name: str) -> BasePipelineWrapper | None:
        return self._pipelines.get(name)

    def get_metadata(self, name: str) -> dict[str, Any] | None:
        return self._metadata.get(name)

    def update_metadata(self, name: str, metadata: dict[str, Any]) -> None:
        if name not in self._metadata:
            msg = f"Pipeline {name} not found in registry."
            raise PipelineNotFoundError(msg)
        self._metadata[name].update(metadata)

    def get_names(self) -> list[str]:
        return list(self._pipelines.keys())

    def clear(self) -> None:
        self._pipelines.clear()
        self._metadata.clear()


registry = _PipelineRegistry()


@dataclass(frozen=True, slots=True)
class PipelineRegistration:
    """One pipeline of an immutable registry; ``metadata`` is a shallow, read-only snapshot."""

    name: str
    wrapper: BasePipelineWrapper
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class ImmutablePipelineRegistry:
    """Pipelines fixed at startup, with the read methods of the mutable registry."""

    def __init__(self, registrations: Iterable[PipelineRegistration] = ()) -> None:
        members: dict[str, PipelineRegistration] = {}
        for registration in registrations:
            if registration.name in members:
                msg = f"Pipeline '{registration.name}' is registered more than once"
                raise ValueError(msg)
            members[registration.name] = registration
        self._registrations = MappingProxyType(members)

    def get(self, name: str) -> BasePipelineWrapper | None:
        registration = self._registrations.get(name)
        return None if registration is None else registration.wrapper

    def get_metadata(self, name: str) -> Mapping[str, Any] | None:
        registration = self._registrations.get(name)
        return None if registration is None else registration.metadata

    def get_names(self) -> list[str]:
        return list(self._registrations)

    def durable_names(self) -> list[str]:
        return [name for name, registration in self._registrations.items() if is_durable_wrapper(registration.wrapper)]


def require_ordinary_pipelines(pipeline_registry: ImmutablePipelineRegistry, server: str) -> None:
    """
    Reject durable pipelines in a standalone server, which has no durable runtime.

    Raises:
        PipelineModeError: If *pipeline_registry* holds a durable wrapper.
    """
    for name in pipeline_registry.durable_names():
        msg = (
            f"Pipeline '{name}' is durable and cannot be served by the standalone {server} server; "
            "serve durable pipelines with the main HTTP server (hayhooks run)"
        )
        raise PipelineModeError(msg)


PipelineRegistry: TypeAlias = _PipelineRegistry | ImmutablePipelineRegistry


def resolve_registry(pipeline_registry: PipelineRegistry | None) -> PipelineRegistry:
    """Return *pipeline_registry*, defaulting to the mutable singleton outside durable mode."""
    if pipeline_registry is not None:
        return pipeline_registry
    if settings.durable_mode:
        msg = "Durable mode requires an explicit pipeline registry"
        raise PipelineModeError(msg)
    return registry
