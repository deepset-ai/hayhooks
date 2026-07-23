import pytest

from hayhooks import BasePipelineWrapper
from hayhooks.server.pipelines.lifecycle import close_pipeline_wrappers, start_pipeline_wrappers
from hayhooks.server.pipelines.registry import registry


class _LifecycleWrapper(BasePipelineWrapper):
    def __init__(self, name: str, events: list[str], *, fail_start: bool = False) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.fail_start = fail_start

    def setup(self) -> None:
        self.pipeline = object()

    async def start(self) -> None:
        self.events.append(f"start:{self.name}")
        if self.fail_start:
            msg = f"cannot start {self.name}"
            raise RuntimeError(msg)

    async def close(self) -> None:
        self.events.append(f"close:{self.name}")


@pytest.fixture(autouse=True)
def _clear_registry():
    registry.clear()
    yield
    registry.clear()


@pytest.mark.asyncio
async def test_pipeline_wrappers_start_in_deployment_order_and_close_in_reverse() -> None:
    events: list[str] = []
    registry.add("first", _LifecycleWrapper("first", events))
    registry.add("second", _LifecycleWrapper("second", events))

    await start_pipeline_wrappers()
    await close_pipeline_wrappers()

    assert events == ["start:first", "start:second", "close:second", "close:first"]


@pytest.mark.asyncio
async def test_pipeline_wrapper_start_failure_closes_partial_startup() -> None:
    events: list[str] = []
    registry.add("first", _LifecycleWrapper("first", events))
    registry.add("broken", _LifecycleWrapper("broken", events, fail_start=True))

    with pytest.raises(RuntimeError, match="cannot start broken"):
        await start_pipeline_wrappers()

    assert events == ["start:first", "start:broken", "close:broken", "close:first"]
