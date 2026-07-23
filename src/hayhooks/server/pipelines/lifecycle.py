"""Lifecycle helpers for deployed pipeline wrappers."""

from __future__ import annotations

from hayhooks.durable_runtime import durable_runtime
from hayhooks.server.logger import log
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


def _implements_lifecycle(wrapper: BasePipelineWrapper, method_name: str) -> bool:
    return getattr(type(wrapper), method_name) is not getattr(BasePipelineWrapper, method_name)


async def start_pipeline_wrapper(name: str, wrapper: BasePipelineWrapper) -> None:
    """Start one wrapper if it overrides the optional lifecycle hook."""
    await start_pipeline_wrapper_lifecycle(name, wrapper)
    await durable_runtime.start_wrapper(name, wrapper)


async def start_pipeline_wrapper_lifecycle(name: str, wrapper: BasePipelineWrapper) -> None:
    """Start only the wrapper-owned lifecycle, for staged deployment transactions."""
    if _implements_lifecycle(wrapper, "start"):
        await wrapper.start()
        log.info("Pipeline '{}' lifecycle started", name)


async def close_pipeline_wrapper(name: str, wrapper: BasePipelineWrapper) -> None:
    """Close one wrapper if it overrides the optional lifecycle hook."""
    deployment = durable_runtime.current_deployment(name)
    await durable_runtime.close_wrapper(name)
    if deployment is not None and deployment.manager.draining:

        async def close_after_drain() -> None:
            await deployment.manager.wait_drained()
            await close_pipeline_wrapper_lifecycle(name, wrapper)

        durable_runtime.track_background_task(
            close_after_drain(),
            name=f"durable-shutdown-cleanup:{name}",
        )
    else:
        await close_pipeline_wrapper_lifecycle(name, wrapper)


async def close_pipeline_wrapper_lifecycle(name: str, wrapper: BasePipelineWrapper) -> None:
    """Close only the wrapper-owned lifecycle, for staged deployment transactions."""
    if _implements_lifecycle(wrapper, "close"):
        await wrapper.close()
        log.info("Pipeline '{}' lifecycle stopped", name)


async def start_pipeline_wrappers() -> None:
    """Start every currently registered wrapper, rolling back partial startup."""
    started: list[tuple[str, BasePipelineWrapper]] = []
    try:
        for name in registry.get_names():
            wrapper = registry.get(name)
            if wrapper is None or not _implements_lifecycle(wrapper, "start"):
                continue
            started.append((name, wrapper))
            await start_pipeline_wrapper(name, wrapper)
        await durable_runtime.start()
    except BaseException:
        for name, wrapper in reversed(started):
            try:
                await close_pipeline_wrapper(name, wrapper)
            except Exception as error:
                log.opt(exception=True).warning("Error rolling back pipeline '{}' lifecycle: {}", name, error)
        raise


async def close_pipeline_wrappers() -> None:
    """Close registered wrapper lifecycles without hiding other shutdown work."""
    try:
        wrappers = [(name, registry.get(name)) for name in reversed(registry.get_names())]
        for name, wrapper in wrappers:
            if wrapper is None:
                continue
            try:
                await close_pipeline_wrapper(name, wrapper)
            except Exception as error:
                log.opt(exception=True).warning("Error closing pipeline '{}' lifecycle: {}", name, error)
    finally:
        await durable_runtime.close()


__all__ = [
    "close_pipeline_wrapper",
    "close_pipeline_wrapper_lifecycle",
    "close_pipeline_wrappers",
    "start_pipeline_wrapper",
    "start_pipeline_wrapper_lifecycle",
    "start_pipeline_wrappers",
]
